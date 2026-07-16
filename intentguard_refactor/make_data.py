from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from intentguard.dataset import build_dataset
from intentguard.io import read_json, write_json, write_jsonl


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path, seen: set[Path] | None = None) -> dict:
    resolved = path.resolve()
    visited = set() if seen is None else set(seen)
    if resolved in visited:
        raise ValueError(f"Circular config inheritance detected at {resolved}")
    visited.add(resolved)
    cfg = read_json(resolved)
    parent = cfg.get("extends")
    if not parent:
        return cfg
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    return deep_merge(load_config(parent_path, visited), cfg)


def summarize(rows: list[dict]) -> dict:
    image_rows = [r for r in rows if r.get("image_path")]
    semantic_rows = [r for r in image_rows if r.get("carrier_type") == "semantic_image"]
    return {
        "n": len(rows),
        "pairs": len({r["pair_key"] for r in rows}),
        "label_counts": dict(Counter(r["label_name"] for r in rows)),
        "intent_family_counts": dict(Counter(r["intent_family"] for r in rows)),
        "condition_counts": dict(Counter(r["condition"] for r in rows)),
        "wrapper_family_counts": dict(Counter(r.get("wrapper_family", "") for r in rows)),
        "evaluation_split_counts": dict(Counter(r.get("evaluation_split", "") for r in rows)),
        "split_group_counts": dict(Counter(r.get("evaluation_split", "") for r in {r["template_id"]: r for r in rows}.values())),
        "image_source_counts": dict(Counter(r.get("image_source", "none") for r in rows)),
        "image_path_prefix_ok": all((not r.get("image_path")) or str(r["image_path"]).startswith("imgs/") for r in rows),
        "references_global_generated_dir": any(str(r.get("image_path") or "").startswith("imgs/intentguard_generated/") for r in rows),
        "semantic_images_from_existing_intent_dirs": all(
            str(r.get("image_source", "")).startswith("existing_intent_semantic") for r in semantic_rows
        ),
        "image_rows": len(image_rows),
        "semantic_image_rows": len(semantic_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("intentguard_refactor/configs/intentguard_families.json"))
    parser.add_argument("--out", type=Path, default=Path("data/intentguard_refactor_probe.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("data/intentguard_refactor_probe_summary.json"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--font-path")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    cfg = load_config(args.config)
    rows = build_dataset(cfg, repo_root=repo_root, font_path=args.font_path)
    write_jsonl(args.out, rows)
    summary = summarize(rows)
    write_json(args.summary_out, summary)
    print(f"Wrote {summary['n']} rows / {summary['pairs']} pairs to {args.out}")
    print(f"Image paths aligned to imgs/: {summary['image_path_prefix_ok']}")


if __name__ == "__main__":
    main()
