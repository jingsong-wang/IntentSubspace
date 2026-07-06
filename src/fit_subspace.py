import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def orthonormalize(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return rows
    q, _ = np.linalg.qr(rows.T)
    return q.T[: rows.shape[0]]


def paired_deltas(X: np.ndarray, y: np.ndarray, pair_keys: np.ndarray) -> np.ndarray:
    by_pair = defaultdict(dict)
    for i, key in enumerate(pair_keys):
        by_pair[str(key)][int(y[i])] = X[i]

    deltas = []
    missing = []
    for key, items in by_pair.items():
        if 0 in items and 1 in items:
            deltas.append(items[1] - items[0])
        else:
            missing.append(key)
    if not deltas:
        raise ValueError("No positive/negative matched pairs found.")
    return np.stack(deltas, axis=0)


def fit_basis_from_pairs(X: np.ndarray, y: np.ndarray, pair_keys: np.ndarray, rank: int) -> np.ndarray:
    deltas = paired_deltas(X, y, pair_keys)
    mean_delta = deltas.mean(axis=0)
    rows = [normalize(mean_delta)]

    if rank > 1 and deltas.shape[0] > 1:
        resid = deltas - mean_delta[None, :]
        _, _, vt = np.linalg.svd(resid, full_matrices=False)
        rows.extend(vt[: rank - 1])

    basis = orthonormalize(np.stack(rows, axis=0))
    if basis.shape[0] > 0 and float(basis[0] @ mean_delta) < 0:
        basis[0] *= -1.0
    return basis


def delta_diagnostics(X: np.ndarray, y: np.ndarray, groups: np.ndarray, pair_keys: np.ndarray) -> dict:
    deltas = paired_deltas(X, y, pair_keys)
    mean_delta = deltas.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_delta))
    centered = deltas - mean_delta[None, :]
    if centered.shape[0] > 1:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        var = s**2
        explained = var / max(float(var.sum()), 1e-12)
    else:
        s = np.array([], dtype=np.float32)
        explained = np.array([], dtype=np.float32)

    global_dir = normalize(mean_delta)
    by_pair_group = {}
    for i, key in enumerate(pair_keys):
        by_pair_group.setdefault(str(key), str(groups[i]))

    condition_rows = []
    pair_order = []
    seen = set()
    for key in pair_keys:
        key = str(key)
        if key not in seen:
            seen.add(key)
            pair_order.append(key)

    for condition in sorted(set(groups.tolist())):
        idxs = [i for i, key in enumerate(pair_order) if by_pair_group.get(key) == condition]
        if not idxs:
            continue
        condition_delta = deltas[idxs].mean(axis=0)
        condition_rows.append(
            {
                "condition": str(condition),
                "n_pairs": int(len(idxs)),
                "delta_norm": float(np.linalg.norm(condition_delta)),
                "cos_to_global_delta": float(normalize(condition_delta) @ global_dir),
            }
        )

    return {
        "n_pairs": int(deltas.shape[0]),
        "mean_delta_norm": mean_norm,
        "singular_values": [float(v) for v in s[:10]],
        "explained_variance_ratio": [float(v) for v in explained[:10]],
        "condition_delta_alignment": condition_rows,
    }


def subspace_features(X: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coords = (X - center[None, :]) @ basis.T
    norm = np.linalg.norm(coords, axis=1, keepdims=True)
    return np.concatenate([coords, norm], axis=1)


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if SKLEARN_AVAILABLE:
        return float(roc_auc_score(y_true, scores))
    return float(binary_auc(y_true, scores))


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        wins += np.sum(p > neg)
        wins += 0.5 * np.sum(p == neg)
    return wins / total


def binary_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return float("nan")
    tp = 0
    precisions = []
    for i, label in enumerate(y_sorted, start=1):
        if label == 1:
            tp += 1
            precisions.append(tp / i)
    return float(np.mean(precisions)) if precisions else 0.0


def binary_balanced_accuracy(y_true: np.ndarray, pred: np.ndarray) -> float:
    vals = []
    for label in [0, 1]:
        m = y_true == label
        if np.any(m):
            vals.append(float(np.mean(pred[m] == label)))
    return float(np.mean(vals)) if vals else float("nan")


def binary_accuracy(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(y_true == pred))


def fit_threshold_classifier(train_scores: np.ndarray, y_train: np.ndarray) -> tuple[float, int]:
    pos_mean = float(train_scores[y_train == 1].mean())
    neg_mean = float(train_scores[y_train == 0].mean())
    direction = 1 if pos_mean >= neg_mean else -1
    threshold = 0.5 * (pos_mean + neg_mean)
    return threshold, direction


def evaluate_loco(X: np.ndarray, y: np.ndarray, groups: np.ndarray, pair_keys: np.ndarray, rank: int) -> dict:
    if SKLEARN_AVAILABLE:
        splits = LeaveOneGroupOut().split(X, y, groups)
    else:
        unique_groups = sorted(set(groups.tolist()))
        splits = ((np.where(groups != g)[0], np.where(groups == g)[0]) for g in unique_groups)

    y_all = []
    prob_all = []
    pred_all = []
    folds = []

    for train_idx, test_idx in splits:
        center = X[train_idx].mean(axis=0)
        basis = fit_basis_from_pairs(X[train_idx], y[train_idx], pair_keys[train_idx], rank)
        z_train = subspace_features(X[train_idx], center, basis)
        z_test = subspace_features(X[test_idx], center, basis)

        if SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            z_train = scaler.fit_transform(z_train)
            z_test = scaler.transform(z_test)

            clf = LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear")
            clf.fit(z_train, y[train_idx])
            prob = clf.predict_proba(z_test)[:, 1]
            pred = (prob >= 0.5).astype(int)
            bal_acc = float(balanced_accuracy_score(y[test_idx], pred))
        else:
            train_scores = z_train[:, 0]
            test_scores = z_test[:, 0]
            threshold, direction = fit_threshold_classifier(train_scores, y[train_idx])
            oriented = direction * (test_scores - threshold)
            prob = 1.0 / (1.0 + np.exp(-oriented))
            pred = (oriented >= 0).astype(int)
            bal_acc = binary_balanced_accuracy(y[test_idx], pred)

        y_all.extend(y[test_idx].tolist())
        prob_all.extend(prob.tolist())
        pred_all.extend(pred.tolist())
        folds.append(
            {
                "held_out_condition": str(groups[test_idx][0]),
                "n_test": int(len(test_idx)),
                "auc": safe_auc(y[test_idx], prob),
                "balanced_acc": bal_acc,
            }
        )

    y_all = np.array(y_all)
    prob_all = np.array(prob_all)
    pred_all = np.array(pred_all)
    return {
        "auc": safe_auc(y_all, prob_all),
        "average_precision": float(average_precision_score(y_all, prob_all)) if SKLEARN_AVAILABLE else binary_average_precision(y_all, prob_all),
        "accuracy": float(accuracy_score(y_all, pred_all)) if SKLEARN_AVAILABLE else binary_accuracy(y_all, pred_all),
        "balanced_accuracy": float(balanced_accuracy_score(y_all, pred_all)) if SKLEARN_AVAILABLE else binary_balanced_accuracy(y_all, pred_all),
        "folds": folds,
    }


def condition_score_gaps(X: np.ndarray, y: np.ndarray, groups: np.ndarray, pair_keys: np.ndarray, rank: int) -> dict:
    basis = fit_basis_from_pairs(X, y, pair_keys, rank)
    center = X.mean(axis=0)
    coords = (X - center[None, :]) @ basis.T
    score = coords[:, 0]
    gaps = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        pos = score[m & (y == 1)]
        neg = score[m & (y == 0)]
        gaps[str(g)] = {
            "pos_mean": float(pos.mean()),
            "neg_mean": float(neg.mean()),
            "gap": float(pos.mean() - neg.mean()),
        }
    return gaps


def build_groups(kind: str, conditions: np.ndarray, intent_ids: np.ndarray) -> np.ndarray:
    if kind == "condition":
        return conditions
    if kind == "intent":
        return intent_ids
    if kind == "condition_intent":
        return np.array([f"{c}::{i}" for c, i in zip(conditions.tolist(), intent_ids.tolist())])
    raise ValueError(f"Unsupported group-by mode: {kind}")


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    out = []
    out.append("| " + " | ".join(columns) + " |")
    out.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.4f}"
            values.append(str(val))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--group-by", choices=["condition", "intent", "condition_intent"], default="condition")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.activations, allow_pickle=True)
    A = data["activations"]
    layers = data["layers"]
    y = data["labels"].astype(int)
    conditions = data["conditions"].astype(str)
    intent_ids = data["intent_ids"].astype(str) if "intent_ids" in data else np.array(["unknown_intent"] * len(y))
    groups = build_groups(args.group_by, conditions, intent_ids)
    pair_keys = data["pair_keys"].astype(str)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_results = {}
    bases = []
    centers = []
    for li, layer in enumerate(layers):
        X = A[:, li, :]
        eval_result = evaluate_loco(X, y, groups, pair_keys, args.rank)
        gaps = condition_score_gaps(X, y, groups, pair_keys, args.rank)
        diagnostics = delta_diagnostics(X, y, groups, pair_keys)
        basis = fit_basis_from_pairs(X, y, pair_keys, args.rank)
        center = X.mean(axis=0)
        bases.append(basis)
        centers.append(center)

        key = f"layer_{int(layer)}"
        all_results[key] = {"loco": eval_result, "condition_score_gaps": gaps, "delta_diagnostics": diagnostics}
        summary_rows.append(
            {
                "layer": int(layer),
                "rank": args.rank,
                "LOCO AUROC": eval_result["auc"],
                "LOCO AP": eval_result["average_precision"],
                "LOCO acc": eval_result["accuracy"],
                "LOCO bal_acc": eval_result["balanced_accuracy"],
            }
        )

    np.savez_compressed(
        args.out_dir / "intent_subspace.npz",
        layers=layers,
        bases=np.stack(bases, axis=0),
        centers=np.stack(centers, axis=0),
        rank=np.array([args.rank], dtype=np.int32),
        group_by=np.array([args.group_by]),
    )
    with (args.out_dir / "subspace_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    report_lines = [
        "# Intent Subspace Probe Report",
        "",
        f"Activations: `{args.activations}`",
        f"Rank: `{args.rank}`",
        "",
        f"## Leave-One-{args.group_by}-Out Summary",
        "",
        f"Group-by: `{args.group_by}`",
        "",
        markdown_table(summary_rows, ["layer", "rank", "LOCO AUROC", "LOCO AP", "LOCO acc", "LOCO bal_acc"]),
        "",
        "## Per-Layer Fold Results",
        "",
    ]
    for key, result in all_results.items():
        report_lines.append(f"### {key}")
        report_lines.append("")
        report_lines.append(markdown_table(result["loco"]["folds"], ["held_out_condition", "n_test", "auc", "balanced_acc"]))
        report_lines.append("")
        gap_rows = [{"condition": k, **v} for k, v in result["condition_score_gaps"].items()]
        report_lines.append("Condition score gaps on first subspace coordinate:")
        report_lines.append("")
        report_lines.append(markdown_table(gap_rows, ["condition", "pos_mean", "neg_mean", "gap"]))
        report_lines.append("")
        diag = result["delta_diagnostics"]
        report_lines.append("Delta-space diagnostics:")
        report_lines.append("")
        report_lines.append(
            markdown_table(
                [
                    {
                        "n_pairs": diag["n_pairs"],
                        "mean_delta_norm": diag["mean_delta_norm"],
                        "evr_1": diag["explained_variance_ratio"][0] if diag["explained_variance_ratio"] else float("nan"),
                        "evr_2": diag["explained_variance_ratio"][1] if len(diag["explained_variance_ratio"]) > 1 else float("nan"),
                        "evr_3": diag["explained_variance_ratio"][2] if len(diag["explained_variance_ratio"]) > 2 else float("nan"),
                    }
                ],
                ["n_pairs", "mean_delta_norm", "evr_1", "evr_2", "evr_3"],
            )
        )
        report_lines.append("")
        report_lines.append("Condition delta alignment to global mean delta:")
        report_lines.append("")
        report_lines.append(markdown_table(diag["condition_delta_alignment"], ["condition", "n_pairs", "delta_norm", "cos_to_global_delta"]))
        report_lines.append("")

    report_path = args.out_dir / "subspace_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote report to {report_path}")
    print(f"Wrote basis to {args.out_dir / 'intent_subspace.npz'}")


if __name__ == "__main__":
    main()
