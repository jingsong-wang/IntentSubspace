from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import (
    assign_component_splits,
    balanced_sample,
    deduplicate,
    load_source_records,
    normalize_record,
    preselect_records,
    summarize_samples,
)
from .io_utils import canonical_json, load_config, repo_root, sha256_text, write_json_atomic, write_jsonl_atomic
from .schema import StudySample


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a leakage-aware unified benchmark manifest.")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "default.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--max-per-label-source", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--only-sources", help="Optional comma-separated source names.")
    parser.add_argument("--image-search-root", type=Path, action="append", default=[])
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--drop-missing-images", action="store_true")
    parser.add_argument("--allow-source-failures", action="store_true")
    return parser.parse_args(argv)


def build_manifest(args: argparse.Namespace) -> tuple[list[StudySample], dict[str, Any]]:
    root = args.repo_root.expanduser().resolve()
    config = load_config(args.config.expanduser().resolve())
    seed = int(args.seed if args.seed is not None else config["seed"])
    maximum = int(
        args.max_per_label_source
        if args.max_per_label_source is not None
        else config["max_samples_per_label_per_source"]
    )
    only = {value.strip() for value in (args.only_sources or "").split(",") if value.strip()}
    asset_dir = args.out.expanduser().resolve().parent / "materialized_images"
    search_dirs = [path.expanduser().resolve() for path in args.image_search_root]

    all_samples: list[StudySample] = []
    source_audit: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    duplicate_count = 0
    for external, specs in ((False, config.get("datasets", [])), (True, config.get("attacks", []))):
        for spec in specs:
            name = str(spec["name"])
            if only and name not in only:
                continue
            try:
                records, provenance = load_source_records(root, spec)
            except Exception as exc:
                failures.append({"source": name, "stage": "load", "error": f"{type(exc).__name__}: {exc}"})
                continue

            selected_records, preselection_duplicates = preselect_records(
                records,
                spec=spec,
                maximum_per_label=maximum,
                seed=seed,
            )
            normalized: list[StudySample] = []
            for record in selected_records:
                try:
                    normalized.append(
                        normalize_record(
                            record,
                            root=root,
                            spec=spec,
                            seed=seed,
                            fractions=config["split_fractions"],
                            asset_dir=asset_dir,
                            search_dirs=search_dirs,
                            external=external,
                        )
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "source": name,
                            "stage": "normalize",
                            "source_row": record.source_row,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            sampled, postselection_duplicates = deduplicate(normalized)
            duplicates = preselection_duplicates + postselection_duplicates
            duplicate_count += duplicates
            if args.drop_missing_images:
                sampled = [
                    sample
                    for sample in sampled
                    if not sample.metadata.get("raw_image_present") or sample.image_exists
                ]
            all_samples.extend(sampled)
            source_audit[name] = {
                **provenance,
                "loaded_rows": len(records),
                "eligible_rows": len(records) - preselection_duplicates,
                "preselected_rows": len(selected_records),
                "normalized_rows": len(normalized),
                "sampled_rows": len(sampled),
                "duplicates_removed": duplicates,
                "summary": summarize_samples(sampled),
            }

    all_samples = assign_component_splits(all_samples, seed=seed, fractions=config["split_fractions"])
    all_samples.sort(key=lambda item: (item.source, item.sample_id))
    for source, audit in source_audit.items():
        source_samples = [sample for sample in all_samples if sample.source == source]
        audit["sampled_rows"] = len(source_samples)
        audit["summary"] = summarize_samples(source_samples)
    ids = [sample.sample_id for sample in all_samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate sample_id values")

    missing = [
        sample
        for sample in all_samples
        if sample.metadata.get("raw_image_present") and not sample.image_exists
    ]
    summary = summarize_samples(all_samples)
    metadata = {
        "schema_version": "ood_intent_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.expanduser().resolve()),
        "config_sha256": sha256_text(canonical_json(config)),
        "repo_root": str(root),
        "seed": seed,
        "max_samples_per_label_per_source": maximum,
        "manifest_sha256": sha256_text("\n".join(canonical_json(item.to_dict()) for item in all_samples)),
        "summary": summary,
        "source_audit": source_audit,
        "failures": failures,
        "duplicates_removed": duplicate_count,
        "missing_image_examples": [item.to_dict() for item in missing[:20]],
        "analysis_contract": {
            "external_attacks_never_train": True,
            "split_key": "connected(group_id, semantic_group_id) -> split_group_id",
            "response_fields_excluded": ["target", "output", "answer", "response"],
            "primary_readout": "last",
        },
    }
    audit_path = args.out.expanduser().resolve().with_suffix(".audit.json")
    if failures and not args.allow_source_failures:
        write_json_atomic(audit_path, metadata)
        raise RuntimeError(
            f"{len(failures)} source rows failed to load or normalize. See {audit_path}; "
            "fix the assets/schema or use --allow-source-failures only for an audit run."
        )
    if missing and not args.allow_missing_images and not args.drop_missing_images:
        write_json_atomic(audit_path, metadata)
        raise FileNotFoundError(
            f"{len(missing)} selected multimodal rows have missing images. "
            f"See {audit_path}; sync/rebuild assets, add --image-search-root, or use "
            "--allow-missing-images only for an audit manifest."
        )
    return all_samples, metadata


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    samples, metadata = build_manifest(args)
    output = args.out.expanduser().resolve()
    write_jsonl_atomic(output, (sample.to_dict() for sample in samples))
    write_json_atomic(output.with_suffix(".manifest.json"), metadata)
    print(f"Wrote {len(samples)} samples to {output}")
    print(f"Manifest fingerprint: {metadata['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
