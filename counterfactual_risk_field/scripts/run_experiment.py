from __future__ import annotations

import argparse
import json
from pathlib import Path

from counterfactual_risk_field.cnrf.activations import load_activation_data
from counterfactual_risk_field.cnrf.experiment import Experiment
from counterfactual_risk_field.cnrf.io import read_jsonl, write_json, write_jsonl


def _integers(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit and evaluate CNRF on frozen activations.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("counterfactual_risk_field/configs/protocol_v1.json"))
    parser.add_argument("--readouts", help="Comma separated; defaults to protocol config.")
    parser.add_argument("--layers", help="Optional comma-separated one-based layer numbers.")
    parser.add_argument("--max-reference-packs", type=int)
    parser.add_argument("--fusion-policy", choices=["legacy_max", "supported_max"])
    parser.add_argument("--min-field-pairs", type=int)
    parser.add_argument("--min-field-packs", type=int)
    parser.add_argument("--min-field-retention-fraction", type=float)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    risk = config["risk_field"]
    rows = list(read_jsonl(args.manifest))
    activations = load_activation_data(args.activations)
    requested_readouts = (
        [item.strip() for item in args.readouts.split(",") if item.strip()]
        if args.readouts
        else list(config["readouts"])
    )
    available = set(activations.readouts.astype(str).tolist())
    readouts = [value for value in requested_readouts if value in available]
    if not readouts:
        raise ValueError(f"None of the requested readouts are available: requested={requested_readouts}, available={sorted(available)}")
    experiment = Experiment(
        rows,
        activations,
        k=int(risk["k"]),
        alpha=float(config["calibration"]["primary_alpha"]),
        support_quantile=float(risk["support_quantile"]),
        min_arrow_norm=float(risk["min_arrow_norm"]),
        score_clip=float(risk["score_clip"]),
        min_field_pairs=int(
            args.min_field_pairs
            if args.min_field_pairs is not None
            else risk.get("min_field_pairs", 2)
        ),
        min_field_packs=int(
            args.min_field_packs
            if args.min_field_packs is not None
            else risk.get("min_field_packs", 2)
        ),
        min_field_retention_fraction=float(
            args.min_field_retention_fraction
            if args.min_field_retention_fraction is not None
            else risk.get("min_field_retention_fraction", 0.0)
        ),
        fusion_policy=str(
            args.fusion_policy or risk.get("fusion_policy", "legacy_max")
        ),
        seed=int(config["seed"]),
        max_reference_packs=args.max_reference_packs,
    )
    summary, scores = experiment.run(readouts=readouts, layer_candidates=_integers(args.layers))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "summary.json", summary)
    write_jsonl(args.out_dir / "scores.jsonl", scores)
    print(f"Wrote CNRF experiment to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
