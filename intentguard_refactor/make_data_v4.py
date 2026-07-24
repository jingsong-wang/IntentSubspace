from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from intentguard.dataset_v4 import build_dataset
from intentguard.io import write_json, write_jsonl
from make_data import load_config


def summarize(rows: list[dict]) -> dict:
    branches = Counter(str(row["modality_branch"]) for row in rows)
    semantic_rows = [
        row
        for row in rows
        if row.get("visual_semantic_role") in {"action_semantic", "evidence_semantic"}
    ]
    return {
        "format_version": "CISR_v4_dataset_summary_v1",
        "n": len(rows),
        "pairs": len({str(row["pair_key"]) for row in rows}),
        "label_counts": dict(Counter(str(row["label_name"]) for row in rows)),
        "modality_branch_counts": dict(branches),
        "condition_counts": dict(Counter(str(row["condition"]) for row in rows)),
        "evaluation_split_counts": dict(
            Counter(str(row["evaluation_split"]) for row in rows)
        ),
        "figstep_style_rows": sum(
            row.get("attack_style_family") == "typographic_list_completion"
            for row in rows
        ),
        "image_path_prefix_ok": all(
            not row.get("image_path") or str(row["image_path"]).startswith("imgs/")
            for row in rows
        ),
        "ocr_complete": all(
            row.get("ocr_render_complete") is True
            for row in rows
            if row.get("visual_semantic_role") == "text_carrier_ocr"
        ),
        "semantic_assets_existing_only": all(
            str(row.get("image_source", "")).startswith("existing_")
            for row in semantic_rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CISR_v4 and its isolated text/multimodal manifests."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("intentguard_refactor/configs/cisr_v4_families.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/CISR_v4_probe.jsonl"))
    parser.add_argument(
        "--text-out", type=Path, default=Path("data/CISR_v4_text_probe.jsonl")
    )
    parser.add_argument(
        "--multimodal-out",
        type=Path,
        default=Path("data/CISR_v4_multimodal_probe.jsonl"),
    )
    parser.add_argument(
        "--summary-out", type=Path, default=Path("data/CISR_v4_probe_summary.json")
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--font-path")
    args = parser.parse_args()

    rows = build_dataset(load_config(args.config), args.repo_root.resolve(), args.font_path)
    text_rows = [row for row in rows if row["modality_branch"] == "text"]
    multimodal_rows = [row for row in rows if row["modality_branch"] == "multimodal"]
    if not text_rows or not multimodal_rows:
        raise ValueError("Both CISR_v4 modality branches must be non-empty.")
    for branch, members in (("text", text_rows), ("multimodal", multimodal_rows)):
        if {int(row["label"]) for row in members} != {0, 1}:
            raise ValueError(f"CISR_v4 {branch} branch must contain both labels.")
        if {str(row["evaluation_split"]) for row in members} != {
            "train",
            "validation",
            "calibration",
            "test",
        }:
            raise ValueError(f"CISR_v4 {branch} branch is missing a protocol split.")

    summary = summarize(rows)
    if (
        not summary["image_path_prefix_ok"]
        or not summary["ocr_complete"]
        or not summary["semantic_assets_existing_only"]
    ):
        raise ValueError("CISR_v4 image paths or OCR completeness checks failed.")
    write_jsonl(args.out, rows)
    write_jsonl(args.text_out, text_rows)
    write_jsonl(args.multimodal_out, multimodal_rows)
    write_json(args.summary_out, summary)
    print(
        f"Wrote CISR_v4 rows={len(rows)} text={len(text_rows)} "
        f"multimodal={len(multimodal_rows)}"
    )


if __name__ == "__main__":
    main()
