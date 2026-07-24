from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from counterfactual_risk_field.cnrf.io import (
    manifest_sha256,
    read_jsonl,
    stable_fraction,
    write_json,
    write_jsonl,
)


TRAIN_SEMANTIC = {"AdvBench", "MM-SafetyBench"}
TRAIN_CARRIERS = {"Alpaca", "VizWiz-VQA"}
FROZEN_TEST = {
    "XSTest",
    "MM-Vet",
    "JailBreakV-28K-FigStep",
    "JailBreakV-28K-LLM-Transfer",
    "JailBreakV-28K-Query-Related",
}
FROZEN_EXTERNAL = {"FigStep", "JOOD", "CS-DJ"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the source-role manifest for CNRF v1.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--ood-config",
        type=Path,
        default=Path("ood_intent_study/configs/default.json"),
    )
    parser.add_argument("--max-per-label-source", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help="Reuse an existing normalized ood_intent_study JSONL manifest instead of reading raw sources.",
    )
    parser.add_argument(
        "--smoke-allow-missing-sources",
        action="store_true",
        help="Allow a reused smoke manifest to omit configured sources (never use for a formal run).",
    )
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--allow-source-failures", action="store_true")
    return parser.parse_args()


def _sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("id") or "")


def _select_reused_rows(
    path: Path, *, sources: set[str], maximum: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    loaded = list(read_jsonl(path))
    for row in loaded:
        source = str(row.get("source") or "")
        if source in sources:
            grouped[(source, int(row["label"]))].append(row)
    selected: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda row: (stable_fraction(_sample_id(row), seed), _sample_id(row)))
        selected.extend(values[:maximum])
    sidecar = path.with_suffix(".manifest.json")
    upstream_sidecar: dict[str, Any] | None = None
    if sidecar.exists():
        upstream_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
    return selected, {
        "mode": "reused_normalized_manifest",
        "path": str(path),
        "loaded_rows": len(loaded),
        "selected_rows": len(selected),
        "sidecar": upstream_sidecar,
    }


def main() -> int:
    args = parse_args()
    sources = sorted(TRAIN_SEMANTIC | TRAIN_CARRIERS | FROZEN_TEST | FROZEN_EXTERNAL)
    if args.input_manifest:
        samples, upstream = _select_reused_rows(
            args.input_manifest,
            sources=set(sources),
            maximum=args.max_per_label_source,
            seed=args.seed,
        )
        present = {str(row.get("source") or "") for row in samples}
        missing = sorted(set(sources) - present)
        if missing and not args.smoke_allow_missing_sources:
            raise ValueError(
                f"Reused manifest is missing configured sources: {missing}. "
                "Use --smoke-allow-missing-sources only for a non-formal pipeline test."
            )
        upstream["missing_sources"] = missing
        upstream["smoke_allow_missing_sources"] = bool(args.smoke_allow_missing_sources)
        sample_rows = samples
    else:
        from argparse import Namespace
        from ood_intent_study.build_manifest import build_manifest

        delegated = Namespace(
            config=args.ood_config,
            out=args.out,
            repo_root=Path.cwd(),
            max_per_label_source=args.max_per_label_source,
            seed=args.seed,
            only_sources=",".join(sources),
            image_search_root=[],
            allow_missing_images=args.allow_missing_images,
            drop_missing_images=False,
            allow_source_failures=args.allow_source_failures,
        )
        built_samples, upstream = build_manifest(delegated)
        sample_rows = [sample.to_dict() for sample in built_samples]
    rows = []
    for sample in sample_rows:
        row = dict(sample)
        source = str(row["source"])
        if source in TRAIN_SEMANTIC:
            role, split = "semantic_seed", "seed_pool"
        elif source in TRAIN_CARRIERS:
            role, split = "carrier_donor", "seed_pool"
        elif source in FROZEN_TEST:
            role, split = "frozen_test", "test"
        elif source in FROZEN_EXTERNAL:
            role, split = "frozen_external", "external"
        else:  # pragma: no cover - guarded by only_sources
            continue
        row["metadata"] = {
            **row.get("metadata", {}),
            "protocol_role": role,
            "protocol_split": split,
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["metadata"]["protocol_role"], row["source"], row["sample_id"]))
    write_jsonl(args.out, rows)
    write_json(
        args.out.with_suffix(".manifest.json"),
        {
            "format_version": "cnrf_seed_manifest_v1",
            "manifest_sha256": manifest_sha256(rows),
            "seed": args.seed,
            "formal_eligible": not bool(args.smoke_allow_missing_sources),
            "source_roles": {
                "semantic_seed": sorted(TRAIN_SEMANTIC),
                "carrier_donor": sorted(TRAIN_CARRIERS),
                "frozen_test": sorted(FROZEN_TEST),
                "frozen_external": sorted(FROZEN_EXTERNAL),
            },
            "upstream_manifest": upstream,
        },
    )
    print(f"Wrote {len(rows)} source-role rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
