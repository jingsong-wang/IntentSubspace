from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .risk_field import l2_normalize
from .schema import PairRecord


def _macro_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    classes = sorted(set(actual.astype(str).tolist()))
    values: list[float] = []
    for value in classes:
        truth = actual == value
        guess = predicted == value
        tp = int(np.sum(truth & guess))
        fp = int(np.sum(~truth & guess))
        fn = int(np.sum(truth & ~guess))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(values)) if values else float("nan")


def _folds(groups: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    unique = sorted(set(groups.astype(str).tolist()))
    if len(unique) < 2:
        return []
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    assignments = {group: index % min(count, len(unique)) for index, group in enumerate(unique)}
    return [np.asarray([assignments[str(group)] == fold for group in groups], dtype=bool) for fold in sorted(set(assignments.values()))]


def centroid_group_cv(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    folds: int = 5,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels).astype(str)
    group_values = np.asarray(groups).astype(str)
    predicted = np.full(len(y), "", dtype=object)
    eligible = np.zeros(len(y), dtype=bool)
    for test in _folds(group_values, folds, seed):
        train = ~test
        train_classes = sorted(set(y[train].tolist()))
        if len(train_classes) < 2:
            continue
        mean = x[train].mean(axis=0)
        scale = x[train].std(axis=0)
        scale[scale < 1e-8] = 1.0
        train_x = (x[train] - mean) / scale
        test_x = (x[test] - mean) / scale
        centroids = np.stack([train_x[y[train] == value].mean(axis=0) for value in train_classes])
        centroids = l2_normalize(centroids)
        similarities = l2_normalize(test_x) @ centroids.T
        predicted[test] = np.asarray(train_classes)[np.argmax(similarities, axis=1)]
        eligible[test] = True
    if not np.any(eligible):
        return {"eligible": False, "reason": "insufficient grouped folds"}
    return {
        "eligible": True,
        "n": int(eligible.sum()),
        "classes": sorted(set(y[eligible].tolist())),
        "accuracy": float(np.mean(predicted[eligible] == y[eligible])),
        "macro_f1": _macro_f1(y[eligible], predicted[eligible].astype(str)),
        "group_count": int(len(set(group_values.tolist()))),
    }


def audit_pair_representations(
    activations: np.ndarray,
    pairs: list[PairRecord],
    *,
    source_axis: str,
    seed: int,
) -> dict[str, Any]:
    if source_axis not in {"semantic", "carrier"}:
        raise ValueError("source_axis must be semantic or carrier")
    safe = l2_normalize(np.asarray([activations[pair.benign_index] for pair in pairs]))
    unsafe = l2_normalize(np.asarray([activations[pair.harmful_index] for pair in pairs]))
    midpoint = 0.5 * (safe + unsafe)
    arrow = unsafe - safe
    pair_labels = np.asarray(
        [pair.semantic_source if source_axis == "semantic" else pair.carrier_source for pair in pairs]
    ).astype(str)
    pair_groups = np.asarray([pair.pack_id for pair in pairs]).astype(str)
    endpoint_features = np.concatenate([safe, unsafe], axis=0)
    endpoint_labels = np.concatenate([pair_labels, pair_labels])
    endpoint_groups = np.concatenate([pair_groups, pair_groups])
    return {
        "source_axis": source_axis,
        "raw_endpoints": centroid_group_cv(
            endpoint_features, endpoint_labels, endpoint_groups, seed=seed
        ),
        "midpoints": centroid_group_cv(midpoint, pair_labels, pair_groups, seed=seed),
        "arrows": centroid_group_cv(arrow, pair_labels, pair_groups, seed=seed),
    }
