from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


EPS = 1e-12


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = scores[y == 1]
    negative = scores[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - len(positive) * (len(positive) + 1) / 2.0) / (len(positive) * len(negative))


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    positive_n = int(y.sum())
    if positive_n == 0:
        return float("nan")
    values = np.asarray(scores, dtype=float)
    order = np.argsort(-values, kind="mergesort")
    sorted_y = y[order]
    sorted_scores = values[order]
    true_positives = 0
    previous_recall = 0.0
    result = 0.0
    start = 0
    while start < len(y):
        stop = start + 1
        while stop < len(y) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        true_positives += int(sorted_y[start:stop].sum())
        recall = true_positives / positive_n
        precision = true_positives / stop
        result += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return float(result)


def confusion_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    predictions = np.asarray(scores, dtype=float) >= float(threshold)
    tp = int(np.sum((y == 1) & predictions))
    tn = int(np.sum((y == 0) & ~predictions))
    fp = int(np.sum((y == 0) & predictions))
    fn = int(np.sum((y == 1) & ~predictions))
    positive_n = tp + fn
    negative_n = tn + fp
    tpr = tp / positive_n if positive_n else float("nan")
    fpr = fp / negative_n if negative_n else float("nan")
    tnr = 1.0 - fpr if negative_n else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = (
        2.0 * precision * tpr / (precision + tpr)
        if math.isfinite(precision) and math.isfinite(tpr) and precision + tpr > 0
        else float("nan")
    )
    balanced = 0.5 * (tpr + tnr) if math.isfinite(tpr) and math.isfinite(tnr) else float("nan")
    return {
        "n": len(y),
        "positive_n": positive_n,
        "negative_n": negative_n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tpr": tpr,
        "fpr": fpr,
        "tnr": tnr,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced,
    }


def score_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    return {
        "auroc": roc_auc(y, scores),
        "auprc": average_precision(y, scores),
        "threshold": float(threshold),
        **confusion_metrics(y, scores, threshold),
    }


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(scores, dtype=float))
    if len(values) == 1:
        return np.array([values[0]])
    midpoints = 0.5 * (values[:-1] + values[1:])
    margin = max(1.0, float(values[-1] - values[0]))
    return np.concatenate(([values[0] - margin], midpoints, [values[-1] + margin]))


def balanced_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    best_threshold = 0.0
    best_key = (-math.inf, -math.inf, -math.inf)
    for threshold in candidate_thresholds(scores):
        metrics = confusion_metrics(y, scores, float(threshold))
        balanced = float(metrics["balanced_accuracy"])
        key = (
            balanced if math.isfinite(balanced) else -math.inf,
            float(metrics["tpr"]) if math.isfinite(float(metrics["tpr"])) else -math.inf,
            -abs(float(threshold)),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def threshold_at_fpr(y: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    y = np.asarray(y, dtype=int)
    negatives = np.asarray(scores, dtype=float)[y == 0]
    if len(negatives) == 0:
        return float("nan")
    best = float(np.nextafter(np.max(negatives), math.inf))
    best_tpr = -math.inf
    for threshold in candidate_thresholds(scores):
        metrics = confusion_metrics(y, scores, float(threshold))
        fpr = float(metrics["fpr"])
        tpr = float(metrics["tpr"])
        if math.isfinite(fpr) and fpr <= target_fpr and math.isfinite(tpr) and tpr > best_tpr:
            best = float(threshold)
            best_tpr = tpr
    return best


def fit_logistic(X: np.ndarray, y: np.ndarray, seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y)) != 2:
        raise ValueError("Linear probe training requires both labels")
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="liblinear",
        ),
    ).fit(X, y)


def probe_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(model.decision_function(X), dtype=float)


def standardized_centroid_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale < EPS] = 1.0
    train = (X_train - mean) / scale
    test = (X_test - mean) / scale
    direction = train[y_train == 1].mean(axis=0) - train[y_train == 0].mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm > EPS:
        direction = direction / norm
    return train @ direction, test @ direction, direction, scale


def stratified_group_folds(y: np.ndarray, groups: np.ndarray, seed: int, maximum: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    per_class_groups = [len(set(groups[y == label].tolist())) for label in (0, 1)]
    folds = min(maximum, min(per_class_groups))
    if folds < 2:
        return []
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def centroid_domain_auc(
    reference: np.ndarray,
    shifted: np.ndarray,
    reference_groups: np.ndarray,
    shifted_groups: np.ndarray,
    seed: int,
) -> float:
    X = np.concatenate([reference, shifted], axis=0)
    y = np.concatenate([np.zeros(len(reference), dtype=int), np.ones(len(shifted), dtype=int)])
    groups = np.concatenate([reference_groups.astype(str), shifted_groups.astype(str)])
    predictions = np.full(len(y), np.nan, dtype=float)
    folds = stratified_group_folds(y, groups, seed)
    if not folds:
        return float("nan")
    for train, test in folds:
        _, scores, _, _ = standardized_centroid_probe(X[train], y[train], X[test])
        predictions[test] = scores
    valid = np.isfinite(predictions)
    return roc_auc(y[valid], predictions[valid])


def multiclass_centroid_macro_f1(X: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int) -> float:
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold

    labels = np.asarray(labels).astype(str)
    groups = np.asarray(groups).astype(str)
    classes = np.unique(labels)
    if len(classes) < 2:
        return float("nan")
    class_group_counts = [len(np.unique(groups[labels == value])) for value in classes]
    folds = min(5, min(class_group_counts))
    if folds < 2:
        return float("nan")
    predictions = np.full(len(labels), "", dtype=object)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(X, labels, groups):
        if set(np.unique(labels[train]).tolist()) != set(classes.tolist()):
            return float("nan")
        mean = X[train].mean(axis=0)
        scale = X[train].std(axis=0)
        scale[scale < EPS] = 1.0
        train_x = (X[train] - mean) / scale
        test_x = (X[test] - mean) / scale
        train_labels = labels[train]
        fold_classes = np.unique(train_labels)
        centroids = np.stack([train_x[train_labels == value].mean(axis=0) for value in fold_classes])
        train_norm = np.linalg.norm(centroids, axis=1)
        train_norm[train_norm < EPS] = 1.0
        centroids = centroids / train_norm[:, None]
        test_norm = np.linalg.norm(test_x, axis=1)
        test_norm[test_norm < EPS] = 1.0
        similarities = (test_x / test_norm[:, None]) @ centroids.T
        predictions[test] = fold_classes[np.argmax(similarities, axis=1)]
    valid = predictions != ""
    if not bool(valid.all()):
        return float("nan")
    return float(f1_score(labels[valid], predictions[valid], average="macro"))


def fisher_ratio(X: np.ndarray, y: np.ndarray) -> float:
    positive = X[y == 1]
    negative = X[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    delta = positive.mean(axis=0) - negative.mean(axis=0)
    within = float(np.sum(np.var(positive, axis=0)) + np.sum(np.var(negative, axis=0)))
    return float(delta @ delta) / max(within, EPS)


def centroid_shift(
    X_train: np.ndarray,
    y_train: np.ndarray,
    standard_harmful: np.ndarray,
    attack_harmful: np.ndarray,
) -> dict[str, float]:
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale < EPS] = 1.0
    train = (X_train - mean) / scale
    standard = (standard_harmful - mean) / scale
    attack = (attack_harmful - mean) / scale
    intent = train[y_train == 1].mean(axis=0) - train[y_train == 0].mean(axis=0)
    displacement = attack.mean(axis=0) - standard.mean(axis=0)
    intent_norm = float(np.linalg.norm(intent))
    displacement_norm = float(np.linalg.norm(displacement))
    cosine = (
        float(intent @ displacement) / (intent_norm * displacement_norm)
        if intent_norm > EPS and displacement_norm > EPS
        else float("nan")
    )
    return {
        "centroid_shift_l2": displacement_norm,
        "shift_intent_cosine": cosine,
    }


def rbf_mmd_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    values = np.concatenate([x, y], axis=0)
    distances = np.abs(values - values.T)
    nonzero = distances[distances > 0]
    bandwidth = float(np.median(nonzero)) if len(nonzero) else 1.0
    gamma = 1.0 / max(2.0 * bandwidth * bandwidth, EPS)
    kxx = np.exp(-gamma * (x - x.T) ** 2)
    kyy = np.exp(-gamma * (y - y.T) ** 2)
    kxy = np.exp(-gamma * (x - y.T) ** 2)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def cluster_bootstrap_intervals(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    if iterations <= 0:
        return {}
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    if len(unique) < 2:
        return {}
    indices = {group: np.where(groups == group)[0] for group in unique}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {key: [] for key in ("auroc", "balanced_accuracy", "tpr", "fpr")}
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        sample_indices = np.concatenate([indices[group] for group in sampled])
        metrics = score_metrics(y[sample_indices], scores[sample_indices], threshold)
        for key in values:
            value = float(metrics[key])
            if math.isfinite(value):
                values[key].append(value)
    output: dict[str, float] = {
        "bootstrap_cluster_n": int(len(unique)),
        "bootstrap_requested_b": int(iterations),
    }
    for key, observations in values.items():
        output[f"{key}_bootstrap_valid_b"] = int(len(observations))
        if observations:
            output[f"{key}_conditional_cluster_ci_low"] = float(np.quantile(observations, 0.025))
            output[f"{key}_conditional_cluster_ci_high"] = float(np.quantile(observations, 0.975))
    return output


def finite_or_none(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def json_clean(row: dict[str, Any]) -> dict[str, Any]:
    return {key: finite_or_none(value) for key, value in row.items()}
