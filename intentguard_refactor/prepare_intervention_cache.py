from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from intentguard.io import read_jsonl, write_json


RETAIN_ROLES = {
    "safe_refusal_teacher",
    "safe_target_control",
    "retain_benign",
    "over_refusal_control",
}


def select_layer(data: Any, layer: int) -> np.ndarray:
    activations = np.asarray(data["activations"])
    if activations.ndim == 2:
        return activations
    if activations.ndim != 3:
        raise ValueError("activations must have shape [n, hidden] or [n, layers, hidden]")
    if "layers" not in data:
        raise KeyError("Layered activations require a layers array")
    layers = np.asarray(data["layers"]).astype(int)
    matches = np.where(layers == int(layer))[0]
    if len(matches) != 1:
        raise ValueError(f"Layer {layer} not found; available={layers.tolist()}")
    return activations[:, int(matches[0]), :]


def rows_by_id(data: Any, vectors: np.ndarray) -> dict[str, np.ndarray]:
    ids = np.asarray(data["ids"]).astype(str)
    if len(ids) != len(vectors):
        raise ValueError("ids and activation rows do not align")
    if len(ids) != len(set(ids.tolist())):
        raise ValueError("Activation cache contains duplicate ids")
    return {sample_id: vectors[index] for index, sample_id in enumerate(ids.tolist())}


def optional_array(path: Path | None, expected_hidden: int, name: str) -> np.ndarray | None:
    if path is None:
        return None
    value = np.asarray(np.load(path), dtype=np.float32)
    if value.ndim == 1:
        if value.shape != (expected_hidden,):
            raise ValueError(f"{name} must have shape [{expected_hidden}]")
    elif value.ndim == 2:
        if value.shape[-1] != expected_hidden:
            raise ValueError(f"{name} must have trailing hidden dimension {expected_hidden}")
    else:
        raise ValueError(f"{name} must be one- or two-dimensional")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the aligned hidden-state cache consumed by train_intervention.py."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-activations", type=Path, required=True)
    parser.add_argument("--teacher-activations", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--refusal-direction", type=Path)
    parser.add_argument("--preserve-basis", type=Path)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    if not manifest:
        raise ValueError("Intervention manifest is empty")
    manifest_ids = [str(row.get("id", "")) for row in manifest]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("Intervention manifest contains duplicate ids")

    base_data = np.load(args.base_activations, allow_pickle=True)
    teacher_data = np.load(args.teacher_activations, allow_pickle=True)
    base_by_id = rows_by_id(base_data, select_layer(base_data, args.layer))
    teacher_by_id = rows_by_id(teacher_data, select_layer(teacher_data, args.layer))
    missing_base = [sample_id for sample_id in manifest_ids if sample_id not in base_by_id]
    if missing_base:
        raise KeyError(f"Base activations missing manifest ids: {missing_base[:10]}")

    hidden = np.stack([base_by_id[sample_id] for sample_id in manifest_ids]).astype(np.float32)
    teacher_hidden = hidden.copy()
    route_mask = np.array(
        [row.get("intervention_role") == "route_positive" for row in manifest],
        dtype=bool,
    )
    retain_mask = np.array(
        [str(row.get("intervention_role")) in RETAIN_ROLES for row in manifest],
        dtype=bool,
    )
    missing_teacher = []
    for index, (sample_id, is_route) in enumerate(zip(manifest_ids, route_mask.tolist())):
        split = str(manifest[index].get("evaluation_split", ""))
        teacher_required = is_route and split in {"train", "validation"}
        if not teacher_required:
            continue
        teacher = teacher_by_id.get(sample_id)
        if teacher is None:
            missing_teacher.append(sample_id)
        else:
            teacher_hidden[index] = teacher
    if missing_teacher:
        raise KeyError(
            "Every train/validation route_positive needs a same-input safe teacher activation; "
            f"missing={missing_teacher[:10]}"
        )
    if not np.any(route_mask):
        raise ValueError("Manifest contains no route_positive rows")
    if not np.any(retain_mask):
        raise ValueError("Manifest contains no retain/control rows")

    payload: dict[str, np.ndarray] = {
        "ids": np.asarray(manifest_ids),
        "hidden": hidden,
        "teacher_hidden": teacher_hidden,
        "splits": np.asarray([str(row.get("evaluation_split", "")) for row in manifest]),
        "route_mask": route_mask,
        "retain_mask": retain_mask,
        "layer": np.asarray([args.layer], dtype=np.int32),
    }
    refusal_direction = optional_array(
        args.refusal_direction, hidden.shape[-1], "refusal_direction"
    )
    preserve_basis = optional_array(
        args.preserve_basis, hidden.shape[-1], "preserve_basis"
    )
    if refusal_direction is not None:
        if refusal_direction.ndim != 1:
            raise ValueError("refusal_direction must be one-dimensional")
        payload["refusal_direction"] = refusal_direction
    if preserve_basis is not None:
        if preserve_basis.ndim != 2:
            raise ValueError("preserve_basis must be two-dimensional")
        payload["preserve_basis"] = preserve_basis

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    split_counts = {
        split: sum(str(row.get("evaluation_split", "")) == split for row in manifest)
        for split in ("train", "validation", "calibration", "test")
    }
    summary = {
        "format_version": "CISR_intervention_cache_v1",
        "out": str(args.out),
        "manifest": str(args.manifest),
        "base_activations": str(args.base_activations),
        "teacher_activations": str(args.teacher_activations),
        "layer": args.layer,
        "n": len(manifest),
        "hidden_size": int(hidden.shape[-1]),
        "route_positive_n": int(route_mask.sum()),
        "retain_n": int(retain_mask.sum()),
        "split_counts": split_counts,
        "has_refusal_direction": refusal_direction is not None,
        "has_preserve_basis": preserve_basis is not None,
        "fit_policy": "train only; validation selects checkpoint; calibration/test excluded from fitting",
    }
    summary_path = args.summary_out or args.out.with_suffix(".json")
    write_json(summary_path, summary)
    print(f"Wrote intervention training cache to {args.out}")
    print(f"Wrote cache summary to {summary_path}")


if __name__ == "__main__":
    main()
