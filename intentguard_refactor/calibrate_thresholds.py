from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from fit_subspaces import load_refusal_labels
from intentguard.io import write_json
from intentguard.subspace import choose_threshold, quantiles, score_activation_file, score_metrics


def threshold_block(
    name: str,
    activations: Path,
    subspace: Path,
    labels: np.ndarray,
    layer: str,
    objective: str,
    target_tpr: float,
    target_fpr: float,
    row_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    scored = score_activation_file(activations, subspace, layer=layer, row_mask=row_mask)
    scores = scored["scores"]
    threshold, metrics = choose_threshold(labels, scores, objective=objective, target_tpr=target_tpr, target_fpr=target_fpr)
    overall = score_metrics(labels, scores, threshold)
    return {
        "name": name,
        "subspace": str(subspace),
        "layer": int(scored["subspace"]["layer"]),
        "objective": objective,
        "target_tpr": target_tpr,
        "target_fpr": target_fpr,
        "threshold": threshold,
        "metrics_at_threshold": metrics,
        "auc": overall["auc"],
        "average_precision": overall["average_precision"],
        "score_distribution": {
            "negative": quantiles(scores[labels == 0]),
            "positive": quantiles(scores[labels == 1]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--intent-subspace", type=Path, required=True)
    parser.add_argument("--refusal-subspace", type=Path, required=True)
    parser.add_argument("--refusal-labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-alias", default="")
    parser.add_argument("--intent-layer", default="selected")
    parser.add_argument("--refusal-layer", default="selected")
    parser.add_argument("--intent-objective", choices=["balanced", "youden", "target_tpr", "target_fpr"], default="target_tpr")
    parser.add_argument("--refusal-objective", choices=["balanced", "youden", "target_tpr", "target_fpr"], default="balanced")
    parser.add_argument("--intent-target-tpr", type=float, default=0.95)
    parser.add_argument("--intent-target-fpr", type=float, default=0.10)
    parser.add_argument("--refusal-target-tpr", type=float, default=0.90)
    parser.add_argument("--refusal-target-fpr", type=float, default=0.10)
    parser.add_argument("--harmful-score-threshold", type=int, default=3)
    args = parser.parse_args()

    data = np.load(args.activations, allow_pickle=True)
    intent_labels = data["labels"].astype(int)
    ids = data["ids"].astype(str)

    refusal_by_id = load_refusal_labels(args.refusal_labels, args.harmful_score_threshold)
    refusal_mask = np.array([sample_id in refusal_by_id for sample_id in ids], dtype=bool)
    refusal_labels = np.array([refusal_by_id[sample_id] for sample_id in ids[refusal_mask]], dtype=int)
    if len(refusal_labels) == 0 or len(set(refusal_labels.tolist())) < 2:
        raise ValueError("Refusal threshold calibration requires both refusal and non-refusal labels.")

    result = {
        "model_alias": args.model_alias,
        "activations": str(args.activations),
        "trigger_rule": "intervene iff intent.score >= intent.threshold and refusal.score < refusal.threshold",
        "intent": threshold_block(
            "intent",
            args.activations,
            args.intent_subspace,
            intent_labels,
            args.intent_layer,
            args.intent_objective,
            args.intent_target_tpr,
            args.intent_target_fpr,
        ),
        "refusal": threshold_block(
            "refusal",
            args.activations,
            args.refusal_subspace,
            refusal_labels,
            args.refusal_layer,
            args.refusal_objective,
            args.refusal_target_tpr,
            args.refusal_target_fpr,
            row_mask=refusal_mask,
        ),
    }
    write_json(args.out, result)
    print(f"Wrote dynamic thresholds to {args.out}")
    print(f"S_I layer={result['intent']['layer']} threshold={result['intent']['threshold']:.6f}")
    print(f"S_R layer={result['refusal']['layer']} threshold={result['refusal']['threshold']:.6f}")


if __name__ == "__main__":
    main()
