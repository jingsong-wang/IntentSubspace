from __future__ import annotations

import argparse
from pathlib import Path

from counterfactual_risk_field.cnrf.io import manifest_sha256, read_jsonl, write_json, write_jsonl
from counterfactual_risk_field.cnrf.schema import validate_unique_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine approved pairs with frozen test/attack rows.")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = list(read_jsonl(args.pairs))
    for row in read_jsonl(args.seeds):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        role = str(metadata.get("protocol_role") or "")
        if role not in {"frozen_test", "frozen_external"}:
            continue
        split = "test" if role == "frozen_test" else "external"
        row = dict(row)
        row["metadata"] = {**metadata, "protocol_split": split}
        rows.append(row)
    validate_unique_ids(rows)
    write_jsonl(args.out, rows)
    write_json(
        args.out.with_suffix(".manifest.json"),
        {
            "format_version": "cnrf_experiment_manifest_v1",
            "manifest_sha256": manifest_sha256(rows),
            "rows": len(rows),
            "pair_manifest": str(args.pairs),
            "seed_manifest": str(args.seeds),
        },
    )
    print(f"Wrote {len(rows)} experiment rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
