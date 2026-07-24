from __future__ import annotations

import argparse
from pathlib import Path

from counterfactual_risk_field.cnrf.activations import align_rows, load_activation_data
from counterfactual_risk_field.cnrf.io import read_jsonl, write_json
from counterfactual_risk_field.cnrf.schema import build_pair_records, protocol_split, validate_no_pack_leakage
from counterfactual_risk_field.cnrf.source_audit import audit_pair_representations


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit source/carrier readability in endpoints, midpoints, and arrows.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--readout", default="last")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--split", default="reference")
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    rows = list(read_jsonl(args.manifest))
    pairs = build_pair_records(rows)
    validate_no_pack_leakage(pairs)
    pairs = [pair for pair in pairs if pair.split == args.split]
    data = load_activation_data(args.activations)
    order = align_rows(rows, data)
    readout_matches = [index for index, value in enumerate(data.readouts.astype(str)) if value == args.readout]
    layer_matches = [index for index, value in enumerate(data.layers.astype(int)) if value == args.layer]
    if len(readout_matches) != 1 or len(layer_matches) != 1:
        raise ValueError("Requested readout/layer is not present exactly once")
    values = data.activations[order, readout_matches[0], layer_matches[0], :]
    report = {
        "format_version": "cnrf_source_signal_audit_v1",
        "readout": args.readout,
        "layer": args.layer,
        "split": args.split,
        "pairs": len(pairs),
        "semantic_source": audit_pair_representations(values, pairs, source_axis="semantic", seed=args.seed),
        "carrier_source": audit_pair_representations(values, pairs, source_axis="carrier", seed=args.seed),
        "interpretation_gate": (
            "The mechanism hypothesis is supported only if arrow source/carrier macro-F1 is materially "
            "below endpoint and midpoint macro-F1 under the same grouped folds."
        ),
    }
    write_json(args.out, report)
    print(f"Wrote source-signal audit to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
