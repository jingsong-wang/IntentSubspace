from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from intentguard.dataset_v3 import build_dataset
from intentguard.io import write_json, write_jsonl
from make_data import load_config


def summarize(rows: list[dict]) -> dict:
    image_rows = [row for row in rows if row.get("image_path")]
    semantic_rows = [
        row
        for row in image_rows
        if row.get("visual_semantic_role") in {"action_semantic", "evidence_semantic"}
    ]
    ocr_rows = [row for row in image_rows if row.get("visual_semantic_role") == "text_carrier_ocr"]
    view_counts = Counter(row.get("view_group", "") for row in rows if row.get("view_group"))
    unique_images_by_role = {}
    for role in ("action_semantic", "evidence_semantic", "text_carrier_ocr", "irrelevant"):
        unique_images_by_role[role] = len(
            {
                str(row.get("image_path"))
                for row in image_rows
                if row.get("visual_semantic_role") == role
            }
        )
    return {
        "format_version": "CISR_v3_dataset_summary_v1",
        "n": len(rows),
        "pairs": len({row["pair_key"] for row in rows}),
        "label_counts": dict(Counter(row["label_name"] for row in rows)),
        "intent_family_counts": dict(Counter(row["intent_family"] for row in rows)),
        "condition_counts": dict(Counter(row["condition"] for row in rows)),
        "evaluation_split_counts": dict(Counter(row["evaluation_split"] for row in rows)),
        "visual_semantic_role_counts": dict(
            Counter(row.get("visual_semantic_role", "none") for row in rows)
        ),
        "benign_subtype_counts": dict(
            Counter(row.get("benign_subtype", "") for row in rows if int(row["label"]) == 0)
        ),
        "composition_type_counts": dict(Counter(row.get("composition_type", "") for row in rows)),
        "unique_images_by_role": unique_images_by_role,
        "image_rows": len(image_rows),
        "semantic_image_rows": len(semantic_rows),
        "ocr_image_rows": len(ocr_rows),
        "multi_member_view_groups": sum(count > 1 for count in view_counts.values()),
        "max_view_group_size": max(view_counts.values(), default=0),
        "image_path_prefix_ok": all(
            (not row.get("image_path")) or str(row["image_path"]).startswith("imgs/")
            for row in rows
        ),
        "semantic_assets_existing_only": all(
            str(row.get("image_source", "")).startswith("existing_") for row in semantic_rows
        ),
        "contains_generated_semantic_assets": any(
            str(row.get("image_source", "")).startswith(("t2i", "svg", "generated_semantic"))
            for row in semantic_rows
        ),
        "ocr_all_rendered_complete": all(row.get("ocr_render_complete") is True for row in ocr_rows),
        "references_v2_generated_dir": any(
            "/_intentguard_generated/" in str(row.get("image_path", "")) for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the isolated CISR_v3 detection dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("intentguard_refactor/configs/cisr_v3_families.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/CISR_v3_probe.jsonl"))
    parser.add_argument(
        "--summary-out", type=Path, default=Path("data/CISR_v3_probe_summary.json")
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--font-path")
    args = parser.parse_args()

    rows = build_dataset(load_config(args.config), args.repo_root.resolve(), args.font_path)
    summary = summarize(rows)
    if not summary["image_path_prefix_ok"]:
        raise ValueError("CISR_v3 contains an image path outside imgs/.")
    if not summary["semantic_assets_existing_only"] or summary["contains_generated_semantic_assets"]:
        raise ValueError("CISR_v3 MVP must use existing semantic assets only.")
    if not summary["ocr_all_rendered_complete"]:
        raise ValueError("At least one clean OCR image failed the render-completeness check.")
    if summary["references_v2_generated_dir"]:
        raise ValueError("CISR_v3 must not reuse or overwrite the v2 generated directory.")
    write_jsonl(args.out, rows)
    write_json(args.summary_out, summary)
    print(f"Wrote {summary['n']} rows / {summary['pairs']} pairs to {args.out}")
    print(f"Existing semantic assets only: {summary['semantic_assets_existing_only']}")
    print(f"Clean OCR complete: {summary['ocr_all_rendered_complete']}")


if __name__ == "__main__":
    main()
