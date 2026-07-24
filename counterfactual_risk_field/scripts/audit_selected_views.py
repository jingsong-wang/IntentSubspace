from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from counterfactual_risk_field.cnrf.activations import align_rows, load_activation_data
from counterfactual_risk_field.cnrf.io import read_jsonl, write_json
from counterfactual_risk_field.cnrf.schema import build_pair_records, validate_no_pack_leakage
from counterfactual_risk_field.cnrf.source_audit import audit_pair_representations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit every readout/layer selected by a completed CNRF experiment."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="reference")
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    selected: dict[tuple[str, int], set[str]] = defaultdict(set)
    for branch, report in summary.get("branches", {}).items():
        for view in report.get("selected_views", []):
            selected[(str(view["readout"]), int(view["layer"]))].add(str(branch))
    if not selected:
        raise ValueError("Experiment summary contains no selected views")

    rows = list(read_jsonl(args.manifest))
    pairs = build_pair_records(rows)
    validate_no_pack_leakage(pairs)
    data = load_activation_data(args.activations)
    order = align_rows(rows, data)
    values = data.activations[order]
    validity = data.readout_valid[order]
    reports: list[dict[str, Any]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for (readout, layer), branches in sorted(selected.items()):
        readout_matches = [
            index for index, value in enumerate(data.readouts.astype(str)) if value == readout
        ]
        layer_matches = [
            index for index, value in enumerate(data.layers.astype(int)) if value == layer
        ]
        if len(readout_matches) != 1 or len(layer_matches) != 1:
            raise ValueError(f"Selected view {readout}/layer-{layer} is absent from activations")
        readout_index = readout_matches[0]
        candidate_pairs = [
            pair
            for pair in pairs
            if pair.split == args.split and pair.modality in branches
        ]
        valid_pairs = [
            pair
            for pair in candidate_pairs
            if validity[pair.benign_index, readout_index]
            and validity[pair.harmful_index, readout_index]
        ]
        if not valid_pairs:
            raise ValueError(f"Selected view {readout}/layer-{layer} has no valid audited pairs")
        view_values = values[:, readout_index, layer_matches[0], :]
        report = {
            "format_version": "cnrf_selected_view_source_audit_v1",
            "readout": readout,
            "layer": layer,
            "branches": sorted(branches),
            "split": args.split,
            "candidate_pairs": len(candidate_pairs),
            "pairs": len(valid_pairs),
            "excluded_invalid_pairs": len(candidate_pairs) - len(valid_pairs),
            "semantic_source": audit_pair_representations(
                view_values, valid_pairs, source_axis="semantic", seed=args.seed
            ),
            "carrier_source": audit_pair_representations(
                view_values, valid_pairs, source_axis="carrier", seed=args.seed
            ),
            "interpretation_gate": (
                "Support requires arrow source/carrier macro-F1 to be materially below endpoint "
                "and midpoint macro-F1 under identical grouped folds."
            ),
        }
        filename = f"source_audit_{readout}_layer{layer}.json"
        write_json(args.out_dir / filename, report)
        reports.append(
            {
                "readout": readout,
                "layer": layer,
                "branches": sorted(branches),
                "pairs": len(valid_pairs),
                "path": filename,
            }
        )

    write_json(
        args.out_dir / "source_audit_index.json",
        {
            "format_version": "cnrf_selected_view_source_audit_index_v1",
            "summary": str(args.summary),
            "split": args.split,
            "views": reports,
        },
    )
    print(f"Wrote {len(reports)} selected-view source audits to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
