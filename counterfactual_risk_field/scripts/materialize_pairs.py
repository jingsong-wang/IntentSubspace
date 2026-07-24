from __future__ import annotations

import argparse
from pathlib import Path

from counterfactual_risk_field.cnrf.generation import materialize_approved_pairs
from counterfactual_risk_field.cnrf.io import manifest_sha256, read_jsonl, write_json, write_jsonl
from counterfactual_risk_field.cnrf.schema import build_pair_records, validate_no_pack_leakage, validate_unique_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize approved counterfactual endpoints for extraction.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-unreviewed", action="store_true", help="Development smoke only; forbidden for formal results.")
    args = parser.parse_args()
    rows = materialize_approved_pairs(
        list(read_jsonl(args.candidates)), allow_unreviewed=args.allow_unreviewed
    )
    validate_unique_ids(rows)
    pairs = build_pair_records(rows)
    validate_no_pack_leakage(pairs)
    write_jsonl(args.out, rows)
    write_json(
        args.out.with_suffix(".manifest.json"),
        {
            "format_version": "cnrf_pair_manifest_v1",
            "manifest_sha256": manifest_sha256(rows),
            "rows": len(rows),
            "pairs": len(pairs),
            "packs": len(set(pair.pack_id for pair in pairs)),
            "formal_eligible": not args.allow_unreviewed,
        },
    )
    print(f"Wrote {len(pairs)} pairs ({len(rows)} endpoints) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
