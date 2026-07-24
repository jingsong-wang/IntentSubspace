from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io_utils import canonical_json, read_jsonl, repo_root, sha256_text, write_json_atomic
from .schema import StudySample


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit manifest integrity and major confounds.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-images", action="store_true")
    parser.add_argument("--require-sidecar", action="store_true")
    return parser.parse_args(argv)


def audit(samples: list[StudySample], root: Path, require_images: bool) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    semantic_splits: dict[str, set[str]] = defaultdict(set)
    split_group_splits: dict[str, set[str]] = defaultdict(set)
    external_semantics: set[str] = set()
    standard_semantics: set[str] = set()
    source_labels: dict[str, Counter[int]] = defaultdict(Counter)
    label_modalities: dict[int, Counter[str]] = defaultdict(Counter)
    source_roles: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        if sample.sample_id in seen:
            errors.append(f"duplicate sample_id={sample.sample_id}")
        seen.add(sample.sample_id)
        errors.extend(f"{sample.sample_id}: {value}" for value in sample.validate(root, require_images))
        if sample.is_attack:
            external_semantics.add(sample.semantic_group_id)
        else:
            standard_semantics.add(sample.semantic_group_id)
            group_splits[sample.group_id].add(sample.split)
            semantic_splits[sample.semantic_group_id].add(sample.split)
            split_group_splits[sample.split_group_id].add(sample.split)
        source_labels[sample.source][sample.label] += 1
        label_modalities[sample.label][sample.modality] += 1
        source_roles[sample.source][sample.source_role] += 1
    leaked_groups = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    leaked_semantics = {key: sorted(value) for key, value in semantic_splits.items() if len(value) > 1}
    if leaked_groups:
        errors.append(f"{len(leaked_groups)} group_id values cross splits")
    if leaked_semantics:
        errors.append(f"{len(leaked_semantics)} semantic_group_id values cross splits")
    leaked_split_groups = {
        key: sorted(value) for key, value in split_group_splits.items() if len(value) > 1
    }
    if leaked_split_groups:
        errors.append(f"{len(leaked_split_groups)} split_group_id values cross non-external splits")
    contaminated = sorted(external_semantics & standard_semantics)
    if contaminated:
        warnings.append(
            f"{len(contaminated)} external attack groups overlap standard semantics; inspect contamination."
        )

    benign_image = label_modalities[0]["image_text"]
    harmful_image = label_modalities[1]["image_text"]
    benign_text = label_modalities[0]["text"]
    harmful_text = label_modalities[1]["text"]
    if benign_image == 0 or harmful_image == 0:
        warnings.append("One label has no image-text samples; image presence can become a label shortcut.")
    if benign_text == 0 or harmful_text == 0:
        warnings.append("One label has no text samples; modality and intent are confounded.")
    if source_roles.get("OpenAssistant", Counter()).get("assistant", 0):
        warnings.append("OpenAssistant rows are assistant-role text and can induce role/style leakage.")
    if source_labels.get("DAN-Prompts"):
        warnings.append("DAN-Prompts is labeled by jailbreak-template presence, not confirmed harmful intent.")
    if source_labels.get("VizWiz-VQA"):
        warnings.append(
            "VizWiz-VQA is an assumed-benign visual-QA control, not an explicit safety label; "
            "the local asset directory is historically named VQAv2."
        )
    if any(source.startswith("JailBreakV-28K-") for source in source_labels):
        warnings.append(
            "JailBreakV-28K carrier conditions are frozen external attack holdouts; "
            "they must not be moved into standard train/validation splits."
        )
    if any(sample.source == "CS-DJ" and not sample.image_exists for sample in samples):
        warnings.append("CS-DJ selected rows have missing 12-panel images; inference is not runnable.")

    return {
        "schema_version": "ood_intent_manifest_audit_v1",
        "status": "PASS" if not errors else "FAIL",
        "rows": len(samples),
        "errors": errors[:200],
        "error_count": len(errors),
        "warnings": warnings,
        "leaked_group_examples": dict(list(leaked_groups.items())[:20]),
        "leaked_semantic_examples": dict(list(leaked_semantics.items())[:20]),
        "leaked_split_group_examples": dict(list(leaked_split_groups.items())[:20]),
        "external_semantic_overlap_examples": contaminated[:20],
        "source_labels": {
            source: {str(label): count for label, count in sorted(counts.items())}
            for source, counts in sorted(source_labels.items())
        },
        "label_modalities": {
            str(label): dict(sorted(counts.items())) for label, counts in sorted(label_modalities.items())
        },
        "source_roles": {source: dict(sorted(counts.items())) for source, counts in sorted(source_roles.items())},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    raw_rows = list(read_jsonl(manifest_path))
    samples = [StudySample.from_dict(row) for row in raw_rows]
    result = audit(samples, args.repo_root.expanduser().resolve(), args.require_images)
    calculated = sha256_text("\n".join(canonical_json(row) for row in raw_rows))
    sidecar_path = manifest_path.with_suffix(".manifest.json")
    result["manifest_sha256"] = calculated
    result["manifest_sidecar"] = str(sidecar_path) if sidecar_path.is_file() else None
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        expected = sidecar.get("manifest_sha256")
        if expected != calculated:
            result["errors"].append(
                f"manifest fingerprint differs from sidecar: expected={expected}, calculated={calculated}"
            )
            result["error_count"] += 1
            result["status"] = "FAIL"
    elif args.require_sidecar:
        result["errors"].append(f"required manifest sidecar is missing: {sidecar_path}")
        result["error_count"] += 1
        result["status"] = "FAIL"
    else:
        result["warnings"].append("Manifest sidecar is absent; content fingerprint could not be cross-checked.")
    if args.out:
        write_json_atomic(args.out.expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
