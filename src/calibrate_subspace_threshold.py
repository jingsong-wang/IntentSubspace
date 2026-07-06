import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_subspace(path: Path, layer: int | str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    layers = data["layers"].astype(int)
    layer_value = int(layers[-1]) if layer == "last" else int(layer)
    matches = np.where(layers == layer_value)[0]
    if len(matches) != 1:
        raise ValueError(f"Layer {layer_value} is not present in {path}; available layers={layers.tolist()}")
    idx = int(matches[0])
    return {
        "path": str(path),
        "layer": layer_value,
        "basis": data["bases"][idx],
        "center": data["centers"][idx],
    }


def score_activations(activations_path: Path, subspace: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.load(activations_path, allow_pickle=True)
    layers = data["layers"].astype(int)
    matches = np.where(layers == int(subspace["layer"]))[0]
    if len(matches) != 1:
        raise ValueError(
            f"Activations do not contain layer {subspace['layer']}: available layers={layers.tolist()}"
        )
    X = data["activations"][:, int(matches[0]), :]
    labels = data["labels"].astype(int)
    coords = (X - subspace["center"][None, :]) @ subspace["basis"].T
    scores = coords[:, 0].astype(float)
    metadata = {
        "ids": data["ids"].astype(str).tolist() if "ids" in data else [],
        "conditions": data["conditions"].astype(str).tolist() if "conditions" in data else [],
        "intent_families": data["intent_families"].astype(str).tolist() if "intent_families" in data else [],
    }
    return scores, labels, metadata


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for value in pos:
        wins += float(np.sum(value > neg))
        wins += 0.5 * float(np.sum(value == neg))
    return float(wins / (len(pos) * len(neg)))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    tp = 0
    precisions = []
    for i, label in enumerate(y_sorted, start=1):
        if label == 1:
            tp += 1
            precisions.append(tp / i)
    return float(np.mean(precisions)) if precisions else 0.0


def metrics_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = scores >= threshold
    tp = int(np.sum((y_true == 1) & pred))
    tn = int(np.sum((y_true == 0) & (~pred)))
    fp = int(np.sum((y_true == 0) & pred))
    fn = int(np.sum((y_true == 1) & (~pred)))
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    total = int(len(y_true))
    recall = tp / pos if pos else float("nan")
    specificity = tn / neg if neg else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision == precision and (precision + recall) else float("nan")
    return {
        "n": total,
        "positive_n": pos,
        "negative_n": neg,
        "accuracy": (tp + tn) / total if total else float("nan"),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "recall": recall,
        "specificity": specificity,
        "fpr": fp / neg if neg else float("nan"),
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores.astype(float))
    if len(unique) == 1:
        return np.array([float(unique[0])])
    mids = (unique[:-1] + unique[1:]) / 2.0
    margin = max(1.0, float(unique[-1] - unique[0]))
    return np.concatenate([[unique[0] - margin], mids, [unique[-1] + margin]])


def choose_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    objective: str,
    target_recall: float,
    target_fpr: float,
) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.0
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None

    for threshold in candidate_thresholds(scores):
        m = metrics_at(y_true, scores, float(threshold))
        if objective == "balanced":
            key = (float(m["balanced_accuracy"]), float(m["recall"]), float(m["specificity"]), -abs(float(threshold)))
        elif objective == "youden":
            key = (float(m["recall"]) - float(m["fpr"]), float(m["balanced_accuracy"]), -abs(float(threshold)))
        elif objective == "target_recall":
            ok = float(m["recall"]) >= target_recall
            key = (1.0 if ok else 0.0, float(m["specificity"]), float(threshold) if ok else -abs(target_recall - float(m["recall"])))
        elif objective == "target_fpr":
            ok = float(m["fpr"]) <= target_fpr
            key = (1.0 if ok else 0.0, float(m["recall"]), -float(m["fpr"]), float(threshold))
        else:
            raise ValueError(f"Unsupported objective: {objective}")
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = m

    assert best_metrics is not None
    return best_threshold, best_metrics


def quantiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def write_scores_csv(path: Path, scores: np.ndarray, labels: np.ndarray, metadata: dict[str, Any]) -> None:
    ids = metadata.get("ids") or [str(i) for i in range(len(scores))]
    conditions = metadata.get("conditions") or [""] * len(scores)
    families = metadata.get("intent_families") or [""] * len(scores)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "score", "condition", "intent_family"])
        writer.writeheader()
        for i, score in enumerate(scores):
            writer.writerow(
                {
                    "id": ids[i] if i < len(ids) else i,
                    "label": int(labels[i]),
                    "score": float(score),
                    "condition": conditions[i] if i < len(conditions) else "",
                    "intent_family": families[i] if i < len(families) else "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--subspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-layer", default="last")
    parser.add_argument("--objective", choices=["balanced", "youden", "target_recall", "target_fpr"], default="balanced")
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    args = parser.parse_args()

    subspace = load_subspace(args.subspace, args.score_layer)
    scores, labels, metadata = score_activations(args.activations, subspace)
    if set(np.unique(labels).tolist()) - {0, 1}:
        raise ValueError("Labels must be binary 0/1.")
    if np.sum(labels == 0) == 0 or np.sum(labels == 1) == 0:
        raise ValueError("Calibration requires both benign label 0 and harmful label 1.")

    threshold, metrics = choose_threshold(labels, scores, args.objective, args.target_recall, args.target_fpr)
    result = {
        "threshold": threshold,
        "objective": args.objective,
        "target_recall": args.target_recall,
        "target_fpr": args.target_fpr,
        "layer": int(subspace["layer"]),
        "subspace": str(args.subspace),
        "activations": str(args.activations),
        "metrics_at_threshold": metrics,
        "auc": binary_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "score_distribution": {
            "benign": quantiles(scores[labels == 0]),
            "harmful": quantiles(scores[labels == 1]),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_scores_csv(args.out.with_suffix(".scores.csv"), scores, labels, metadata)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
