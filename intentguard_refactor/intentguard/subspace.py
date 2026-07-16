from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    SKLEARN_AVAILABLE = False


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v if norm < eps else v / norm


def orthonormalize(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return rows
    q, _ = np.linalg.qr(rows.T)
    return q.T[: rows.shape[0]]


def paired_deltas(X: np.ndarray, y: np.ndarray, pair_keys: np.ndarray) -> np.ndarray:
    by_pair: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for i, key in enumerate(pair_keys):
        by_pair[str(key)][int(y[i])] = X[i]
    deltas = [items[1] - items[0] for items in by_pair.values() if 0 in items and 1 in items]
    if not deltas:
        raise ValueError("No matched label 1/0 pairs were found for paired subspace fitting.")
    return np.stack(deltas, axis=0)


def fit_paired_basis(X: np.ndarray, y: np.ndarray, pair_keys: np.ndarray, rank: int) -> tuple[np.ndarray, dict[str, Any]]:
    deltas = paired_deltas(X, y, pair_keys)
    mean_delta = deltas.mean(axis=0)
    rows = [normalize(mean_delta)]
    if rank > 1 and deltas.shape[0] > 1:
        residual = deltas - mean_delta[None, :]
        _, singular_values, vt = np.linalg.svd(residual, full_matrices=False)
        rows.extend(vt[: rank - 1])
    else:
        singular_values = np.array([], dtype=float)
    basis = orthonormalize(np.stack(rows, axis=0))
    if float(basis[0] @ mean_delta) < 0:
        basis[0] *= -1.0
    diagnostics = {
        "fit_mode": "paired_delta",
        "n_pairs": int(deltas.shape[0]),
        "mean_delta_norm": float(np.linalg.norm(mean_delta)),
        "singular_values": [float(v) for v in singular_values[:10]],
    }
    return basis, diagnostics


def fit_binary_basis(X: np.ndarray, y: np.ndarray, rank: int, positive_label_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    pos = X[y == 1]
    neg = X[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(f"{positive_label_name} subspace requires both positive and negative labels.")
    mean_delta = pos.mean(axis=0) - neg.mean(axis=0)
    rows = [normalize(mean_delta)]
    centered_parts = []
    if len(pos) > 1:
        centered_parts.append(pos - pos.mean(axis=0, keepdims=True))
    if len(neg) > 1:
        centered_parts.append(neg - neg.mean(axis=0, keepdims=True))
    if rank > 1 and centered_parts:
        centered = np.concatenate(centered_parts, axis=0)
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        rows.extend(vt[: rank - 1])
    else:
        singular_values = np.array([], dtype=float)
    basis = orthonormalize(np.stack(rows, axis=0))
    if float(basis[0] @ mean_delta) < 0:
        basis[0] *= -1.0
    diagnostics = {
        "fit_mode": "binary_mean_delta",
        "positive_label_name": positive_label_name,
        "positive_n": int(len(pos)),
        "negative_n": int(len(neg)),
        "mean_delta_norm": float(np.linalg.norm(mean_delta)),
        "singular_values": [float(v) for v in singular_values[:10]],
    }
    return basis, diagnostics


def project_scores(X: np.ndarray, center: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = (X - center[None, :]) @ basis.T
    return coords, coords[:, 0].astype(float)


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


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores.astype(float))
    if len(unique) == 1:
        return np.array([float(unique[0])])
    mids = (unique[:-1] + unique[1:]) / 2.0
    margin = max(1.0, float(unique[-1] - unique[0]))
    return np.concatenate([[unique[0] - margin], mids, [unique[-1] + margin]])


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = scores >= threshold
    tp = int(np.sum((y_true == 1) & pred))
    tn = int(np.sum((y_true == 0) & (~pred)))
    fp = int(np.sum((y_true == 0) & pred))
    fn = int(np.sum((y_true == 1) & (~pred)))
    pos_n = int(np.sum(y_true == 1))
    neg_n = int(np.sum(y_true == 0))
    total = int(len(y_true))
    tpr = tp / pos_n if pos_n else float("nan")
    tnr = tn / neg_n if neg_n else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * precision * tpr / (precision + tpr) if precision == precision and precision + tpr else float("nan")
    return {
        "n": total,
        "positive_n": pos_n,
        "negative_n": neg_n,
        "accuracy": (tp + tn) / total if total else float("nan"),
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fp / neg_n if neg_n else float("nan"),
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, objective: str = "balanced", target_tpr: float = 0.95, target_fpr: float = 0.10) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.0
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    for threshold in candidate_thresholds(scores):
        m = metrics_at_threshold(y_true, scores, float(threshold))
        if objective == "balanced":
            key = (float(m["balanced_accuracy"]), float(m["tpr"]), float(m["tnr"]), -abs(float(threshold)))
        elif objective == "youden":
            key = (float(m["tpr"]) - float(m["fpr"]), float(m["balanced_accuracy"]), -abs(float(threshold)))
        elif objective == "target_tpr":
            ok = float(m["tpr"]) >= target_tpr
            key = (1.0 if ok else 0.0, float(m["tnr"]), -float(m["fpr"]), float(threshold))
        elif objective == "target_fpr":
            ok = float(m["fpr"]) <= target_fpr
            key = (1.0 if ok else 0.0, float(m["tpr"]), -float(m["fpr"]), float(threshold))
        else:
            raise ValueError(f"Unsupported threshold objective: {objective}")
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = m
    assert best_metrics is not None
    return best_threshold, best_metrics


def score_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
    if threshold is None:
        threshold, threshold_metrics = choose_threshold(y_true, scores, objective="balanced")
    else:
        threshold_metrics = metrics_at_threshold(y_true, scores, threshold)
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        ap = float("nan")
    elif SKLEARN_AVAILABLE:
        auc = float(roc_auc_score(y_true, scores))
        ap = float(average_precision_score(y_true, scores))
    else:
        auc = binary_auc(y_true, scores)
        ap = average_precision(y_true, scores)
    return {
        "auc": auc,
        "average_precision": ap,
        "threshold": threshold,
        "metrics_at_threshold": threshold_metrics,
    }


def make_splits(y: np.ndarray, groups: np.ndarray | None, seed: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
    if groups is not None and len(set(groups.tolist())) > 1:
        if SKLEARN_AVAILABLE:
            splits = LeaveOneGroupOut().split(np.zeros(len(y)), y, groups)
        else:
            unique_groups = sorted(set(groups.tolist()))
            splits = ((np.where(groups != g)[0], np.where(groups == g)[0]) for g in unique_groups)
        return [(train, test, str(groups[test][0])) for train, test in splits]
    if SKLEARN_AVAILABLE and len(np.unique(y)) == 2 and min(np.bincount(y.astype(int))) >= 2:
        k = min(5, int(min(np.bincount(y.astype(int)))))
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return [(train, test, f"fold_{i}") for i, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y))]
    idx = np.arange(len(y))
    return [(idx, idx, "resubstitution")]


def cross_validated_layer_score(
    X: np.ndarray,
    y: np.ndarray,
    rank: int,
    mode: str,
    groups: np.ndarray | None,
    pair_keys: np.ndarray | None,
    seed: int,
) -> dict[str, Any]:
    y_all: list[int] = []
    score_all: list[float] = []
    pred_all: list[int] = []
    folds = []
    for train_idx, test_idx, fold_name in make_splits(y, groups, seed):
        X_train = X[train_idx]
        y_train = y[train_idx]
        center = X_train.mean(axis=0)
        if mode == "intent":
            if pair_keys is None:
                raise ValueError("Intent subspace selection requires pair_keys.")
            basis, _ = fit_paired_basis(X_train, y_train, pair_keys[train_idx], rank)
        elif mode == "refusal":
            basis, _ = fit_binary_basis(X_train, y_train, rank, "refusal")
        else:
            raise ValueError(f"Unsupported subspace mode: {mode}")
        _, train_scores = project_scores(X_train, center, basis)
        _, test_scores = project_scores(X[test_idx], center, basis)
        threshold, _ = choose_threshold(y_train, train_scores, objective="balanced")
        preds = (test_scores >= threshold).astype(int)
        y_all.extend(y[test_idx].astype(int).tolist())
        score_all.extend(test_scores.astype(float).tolist())
        pred_all.extend(preds.astype(int).tolist())
        fold_metrics = score_metrics(y[test_idx], test_scores, threshold)
        folds.append({"fold": fold_name, "n_test": int(len(test_idx)), **fold_metrics})

    y_arr = np.array(y_all, dtype=int)
    score_arr = np.array(score_all, dtype=float)
    pred_arr = np.array(pred_all, dtype=int)
    overall = score_metrics(y_arr, score_arr)
    if SKLEARN_AVAILABLE and len(np.unique(y_arr)) == 2:
        bal_acc = float(balanced_accuracy_score(y_arr, pred_arr))
    else:
        m = metrics_at_threshold(y_arr, pred_arr.astype(float), 0.5)
        bal_acc = float(m["balanced_accuracy"])
    overall["cross_validated_balanced_accuracy"] = bal_acc
    overall["folds"] = folds
    return overall


def select_best_layer(layer_results: list[dict[str, Any]]) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[float, float, float]:
        auc = row.get("cv", {}).get("auc", float("nan"))
        bal = row.get("cv", {}).get("cross_validated_balanced_accuracy", float("nan"))
        ap = row.get("cv", {}).get("average_precision", float("nan"))
        auc = -math.inf if auc != auc else float(auc)
        bal = -math.inf if bal != bal else float(bal)
        ap = -math.inf if ap != ap else float(ap)
        return auc, bal, ap

    return max(layer_results, key=key)


def fit_all_layers(
    activations: np.ndarray,
    layers: np.ndarray,
    y: np.ndarray,
    rank: int,
    mode: str,
    groups: np.ndarray | None,
    pair_keys: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    bases = []
    centers = []
    layer_results = []
    for layer_index, layer in enumerate(layers.astype(int).tolist()):
        X = activations[:, layer_index, :]
        center = X.mean(axis=0)
        if mode == "intent":
            if pair_keys is None:
                raise ValueError("Intent subspace fitting requires pair_keys.")
            basis, diagnostics = fit_paired_basis(X, y, pair_keys, rank)
        elif mode == "refusal":
            basis, diagnostics = fit_binary_basis(X, y, rank, "refusal")
        else:
            raise ValueError(f"Unsupported subspace mode: {mode}")
        _, full_scores = project_scores(X, center, basis)
        full_metrics = score_metrics(y, full_scores)
        cv = cross_validated_layer_score(X, y, rank, mode, groups, pair_keys, seed)
        bases.append(basis)
        centers.append(center)
        layer_results.append(
            {
                "layer": int(layer),
                "layer_index": int(layer_index),
                "rank": int(rank),
                "full": full_metrics,
                "cv": cv,
                "diagnostics": diagnostics,
            }
        )
    best = select_best_layer(layer_results)
    return np.stack(bases, axis=0), np.stack(centers, axis=0), layer_results, best


def save_subspace(
    path: Path,
    layers: np.ndarray,
    bases: np.ndarray,
    centers: np.ndarray,
    rank: int,
    mode: str,
    selected: dict[str, Any],
    source_activations: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        layers=layers.astype(np.int32),
        bases=bases,
        centers=centers,
        rank=np.array([rank], dtype=np.int32),
        mode=np.array([mode]),
        selected_layer=np.array([int(selected["layer"])], dtype=np.int32),
        selected_layer_index=np.array([int(selected["layer_index"])], dtype=np.int32),
        selected_auc=np.array([float(selected["cv"].get("auc", float("nan")))]),
        selected_balanced_accuracy=np.array([float(selected["cv"].get("cross_validated_balanced_accuracy", float("nan")))]),
        source_activations=np.array([str(source_activations)]),
    )


def load_subspace(path: Path, layer: str | int = "selected") -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    layers = data["layers"].astype(int)
    if str(layer) == "selected" and "selected_layer" in data:
        layer_value = int(data["selected_layer"][0])
    elif str(layer) == "last":
        layer_value = int(layers[-1])
    else:
        layer_value = int(layer)
    matches = np.where(layers == layer_value)[0]
    if len(matches) != 1:
        raise ValueError(f"Layer {layer_value} not found in {path}; available={layers.tolist()}")
    idx = int(matches[0])
    return {
        "path": str(path),
        "mode": str(data["mode"][0]) if "mode" in data else "",
        "layers": layers,
        "layer": layer_value,
        "layer_index": idx,
        "basis": data["bases"][idx],
        "center": data["centers"][idx],
        "rank": int(data["rank"][0]) if "rank" in data else int(data["bases"][idx].shape[0]),
    }


def score_activation_file(activations_path: Path, subspace_path: Path, layer: str | int = "selected", row_mask: np.ndarray | None = None) -> dict[str, Any]:
    data = np.load(activations_path, allow_pickle=True)
    subspace = load_subspace(subspace_path, layer)
    layers = data["layers"].astype(int)
    matches = np.where(layers == subspace["layer"])[0]
    if len(matches) != 1:
        raise ValueError(f"Activations do not contain layer {subspace['layer']}; available={layers.tolist()}")
    X = data["activations"][:, int(matches[0]), :]
    if row_mask is not None:
        X = X[row_mask]
    coords, scores = project_scores(X, subspace["center"], subspace["basis"])
    return {"subspace": subspace, "coords": coords, "scores": scores}


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
