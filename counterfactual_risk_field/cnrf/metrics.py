from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from .calibration import wilson_interval


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    positive = values[y == 1]
    negative = values[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - len(positive) * (len(positive) + 1) / 2.0) / (len(positive) * len(negative))


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    valid = np.isfinite(values)
    y, values = y[valid], values[valid]
    predicted = values >= float(threshold)
    tp = int(np.sum((y == 1) & predicted))
    tn = int(np.sum((y == 0) & ~predicted))
    fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    positives, negatives = tp + fn, tn + fp
    tpr = tp / positives if positives else float("nan")
    fpr = fp / negatives if negatives else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = (
        2.0 * precision * tpr / (precision + tpr)
        if math.isfinite(precision) and math.isfinite(tpr) and precision + tpr > 0
        else float("nan")
    )
    return {
        "n": int(len(y)),
        "positive_n": positives,
        "negative_n": negatives,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": 0.5 * (tpr + 1.0 - fpr) if positives and negatives else float("nan"),
        "auroc": roc_auc(y, values),
        "tpr_ci95": wilson_interval(tp, positives),
        "fpr_ci95": wilson_interval(fp, negatives),
        "threshold": float(threshold),
    }


def grouped_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    groups: Iterable[str],
) -> dict[str, Any]:
    group_values = np.asarray(list(groups)).astype(str)
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    output: dict[str, Any] = {}
    for group in sorted(set(group_values.tolist())):
        mask = group_values == group
        output[group] = binary_metrics(y[mask], values[mask], threshold)
    harmful_tprs = [
        float(value["tpr"])
        for value in output.values()
        if int(value["positive_n"]) > 0 and math.isfinite(float(value["tpr"]))
    ]
    benign_fprs = [
        float(value["fpr"])
        for value in output.values()
        if int(value["negative_n"]) > 0 and math.isfinite(float(value["fpr"]))
    ]
    return {
        "groups": output,
        "macro_tpr": float(np.mean(harmful_tprs)) if harmful_tprs else None,
        "worst_tpr": min(harmful_tprs) if harmful_tprs else None,
        "macro_fpr": float(np.mean(benign_fprs)) if benign_fprs else None,
        "worst_fpr": max(benign_fprs) if benign_fprs else None,
    }
