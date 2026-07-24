from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


PROTOCOL_SPLITS = {"reference", "validation", "calibration", "test", "external"}


def row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def row_id(row: dict[str, Any]) -> str:
    value = row.get("sample_id", row.get("id", ""))
    return str(value)


def row_label(row: dict[str, Any]) -> int:
    return int(row["label"])


def protocol_split(row: dict[str, Any]) -> str:
    metadata = row_metadata(row)
    value = str(
        row.get("protocol_split")
        or metadata.get("protocol_split")
        or row.get("evaluation_split")
        or row.get("split")
        or ""
    )
    aliases = {"train": "reference", "val": "validation"}
    value = aliases.get(value, value)
    if value not in PROTOCOL_SPLITS:
        raise ValueError(f"Unknown CNRF protocol split {value!r} for row {row_id(row)!r}")
    return value


def pair_id(row: dict[str, Any]) -> str:
    metadata = row_metadata(row)
    return str(
        row.get("pair_id")
        or metadata.get("pair_id")
        or row.get("pair_key")
        or ""
    )


def pack_id(row: dict[str, Any]) -> str:
    metadata = row_metadata(row)
    return str(
        row.get("pack_id")
        or metadata.get("pack_id")
        or row.get("composition_group")
        or pair_id(row)
    )


def modality(row: dict[str, Any]) -> str:
    metadata = row_metadata(row)
    value = str(
        row.get("modality_branch")
        or row.get("modality")
        or metadata.get("modality")
        or ""
    )
    if value in {"multimodal", "image-text", "image_text"}:
        return "image_text"
    if value == "text":
        return "text"
    return "image_text" if row.get("image_path") else "text"


def semantic_source(row: dict[str, Any]) -> str:
    metadata = row_metadata(row)
    return str(
        metadata.get("semantic_source")
        or row.get("intent_family")
        or row.get("source")
        or "unspecified"
    )


def carrier_source(row: dict[str, Any]) -> str:
    metadata = row_metadata(row)
    return str(
        metadata.get("carrier_source")
        or row.get("condition")
        or row.get("carrier_type")
        or row.get("source")
        or "unspecified"
    )


def protection_group(row: dict[str, Any]) -> str:
    metadata = row_metadata(row)
    return str(
        metadata.get("protection_group")
        or row.get("benign_subtype")
        or row.get("condition")
        or row.get("source")
        or modality(row)
    )


def evaluation_group(row: dict[str, Any]) -> str:
    metadata = row_metadata(row)
    return str(
        metadata.get("evaluation_group")
        or row.get("condition")
        or row.get("source")
        or modality(row)
    )


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    pack_id: str
    split: str
    modality: str
    benign_index: int
    harmful_index: int
    semantic_source: str
    carrier_source: str


def build_pair_records(rows: list[dict[str, Any]]) -> list[PairRecord]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        value = pair_id(row)
        if value:
            grouped[value].append(index)

    records: list[PairRecord] = []
    errors: list[str] = []
    for current_pair, indices in sorted(grouped.items()):
        by_label: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            by_label[row_label(rows[index])].append(index)
        if len(by_label[0]) != 1 or len(by_label[1]) != 1:
            errors.append(
                f"pair {current_pair!r} requires exactly one benign and one harmful endpoint; "
                f"got benign={len(by_label[0])}, harmful={len(by_label[1])}"
            )
            continue
        benign_index, harmful_index = by_label[0][0], by_label[1][0]
        endpoints = [rows[benign_index], rows[harmful_index]]
        values = {
            "pack_id": {pack_id(row) for row in endpoints},
            "split": {protocol_split(row) for row in endpoints},
            "modality": {modality(row) for row in endpoints},
        }
        inconsistent = {name: value for name, value in values.items() if len(value) != 1}
        if inconsistent:
            errors.append(f"pair {current_pair!r} has inconsistent fields: {inconsistent}")
            continue
        benign, harmful = endpoints
        records.append(
            PairRecord(
                pair_id=current_pair,
                pack_id=next(iter(values["pack_id"])),
                split=next(iter(values["split"])),
                modality=next(iter(values["modality"])),
                benign_index=benign_index,
                harmful_index=harmful_index,
                semantic_source=semantic_source(harmful),
                carrier_source=carrier_source(harmful),
            )
        )
    if errors:
        preview = "\n".join(f"  - {error}" for error in errors[:20])
        raise ValueError(f"Invalid counterfactual pairs ({len(errors)} errors):\n{preview}")
    return records


def validate_no_pack_leakage(records: Iterable[PairRecord]) -> None:
    splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        splits[record.pack_id].add(record.split)
    leaking = {key: sorted(value) for key, value in splits.items() if len(value) > 1}
    if leaking:
        preview = list(leaking.items())[:20]
        raise ValueError(f"pack_id leakage across protocol splits: {preview}")


def validate_unique_ids(rows: Iterable[dict[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        value = row_id(row)
        if not value:
            raise ValueError("Manifest contains an empty sample id")
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"Manifest contains duplicate sample ids: {duplicates[:20]}")
