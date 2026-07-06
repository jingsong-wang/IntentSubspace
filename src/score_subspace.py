import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False


def summarize(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    out = []
    out.append("| " + " | ".join(columns) + " |")
    out.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.4f}"
            vals.append(str(val))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


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
    return float(wins / total)


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


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if SKLEARN_AVAILABLE:
        return float(roc_auc_score(y_true, scores))
    return binary_auc(y_true, scores)


def safe_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if SKLEARN_AVAILABLE:
        return float(average_precision_score(y_true, scores))
    return binary_average_precision(y_true, scores)


def midpoint_threshold_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"threshold": float("nan"), "balanced_acc": float("nan")}
    pos_mean = float(scores[y_true == 1].mean())
    neg_mean = float(scores[y_true == 0].mean())
    threshold = 0.5 * (pos_mean + neg_mean)
    pred = (scores >= threshold).astype(int)
    if SKLEARN_AVAILABLE:
        bal_acc = float(balanced_accuracy_score(y_true, pred))
    else:
        vals = []
        for label in [0, 1]:
            mask = y_true == label
            vals.append(float(np.mean(pred[mask] == label)))
        bal_acc = float(np.mean(vals))
    return {"threshold": threshold, "balanced_acc": bal_acc}


def score_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "auc": safe_auc(y_true, scores),
        "average_precision": safe_ap(y_true, scores),
        **midpoint_threshold_metrics(y_true, scores),
    }


def condition_gaps(conditions: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> list[dict]:
    rows = []
    for condition in sorted(set(conditions.tolist())):
        mask = conditions == condition
        pos = scores[mask & (labels == 1)]
        neg = scores[mask & (labels == 0)]
        if len(pos) == 0 or len(neg) == 0:
            continue
        rows.append(
            {
                "condition": condition,
                "n_pos": int(len(pos)),
                "n_neg": int(len(neg)),
                "pos_mean": float(pos.mean()),
                "neg_mean": float(neg.mean()),
                "gap": float(pos.mean() - neg.mean()),
                "auc": safe_auc(labels[mask], scores[mask]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--subspace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.activations, allow_pickle=True)
    subspace = np.load(args.subspace, allow_pickle=True)

    A = data["activations"]
    act_layers = data["layers"].astype(int)
    sub_layers = subspace["layers"].astype(int)
    bases = subspace["bases"]
    centers = subspace["centers"]

    ids = data["ids"].astype(str)
    labels = data["labels"].astype(int) if "labels" in data else np.full(len(ids), -1)
    conditions = data["conditions"].astype(str)
    intent_texts = data["intent_texts"].astype(str) if "intent_texts" in data else np.array([""] * len(ids))
    intent_ids = data["intent_ids"].astype(str) if "intent_ids" in data else np.array([""] * len(ids))
    intent_families = data["intent_families"].astype(str) if "intent_families" in data else np.array([""] * len(ids))
    image_paths = data["image_paths"].astype(str) if "image_paths" in data else np.array([""] * len(ids))
    image_roles = data["image_roles"].astype(str) if "image_roles" in data else np.array([""] * len(ids))
    sources = data["sources"].astype(str) if "sources" in data else np.array([""] * len(ids))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    sample_rows = []
    metric_rows = []
    gap_rows_by_layer = {}
    intent_gap_rows_by_layer = {}
    json_summary = {}
    has_binary_labels = set(labels.tolist()) == {0, 1}

    for sub_i, layer in enumerate(sub_layers):
        matches = np.where(act_layers == layer)[0]
        if len(matches) != 1:
            continue
        act_i = int(matches[0])
        X = A[:, act_i, :]
        basis = bases[sub_i]
        center = centers[sub_i]
        coords = (X - center[None, :]) @ basis.T
        primary_score = coords[:, 0]
        subspace_norm = np.linalg.norm(coords, axis=1)

        layer_key = f"layer_{int(layer)}"
        json_summary[layer_key] = {"condition_label_summary": {}}

        if has_binary_labels:
            primary_metrics = score_metrics(labels, primary_score)
            norm_metrics = score_metrics(labels, subspace_norm)
            json_summary[layer_key]["label_metrics"] = {
                "primary_score": primary_metrics,
                "subspace_norm": norm_metrics,
            }
            metric_rows.extend(
                [
                    {"layer": int(layer), "score": "primary", **primary_metrics},
                    {"layer": int(layer), "score": "norm", **norm_metrics},
                ]
            )
            gap_rows_by_layer[layer_key] = condition_gaps(conditions, labels, primary_score)
            intent_gap_rows_by_layer[layer_key] = condition_gaps(intent_ids, labels, primary_score)
            json_summary[layer_key]["condition_score_gaps"] = gap_rows_by_layer[layer_key]
            json_summary[layer_key]["intent_score_gaps"] = intent_gap_rows_by_layer[layer_key]

        for condition in sorted(set(conditions.tolist())):
            for label in sorted(set(labels.tolist())):
                mask = (conditions == condition) & (labels == label)
                if not np.any(mask):
                    continue
                score_stats = summarize(primary_score[mask])
                norm_stats = summarize(subspace_norm[mask])
                row = {
                    "layer": int(layer),
                    "condition": condition,
                    "label": int(label),
                    "n": score_stats["n"],
                    "primary_mean": score_stats["mean"],
                    "primary_std": score_stats["std"],
                    "primary_median": score_stats["median"],
                    "norm_mean": norm_stats["mean"],
                    "norm_std": norm_stats["std"],
                }
                summary_rows.append(row)
                json_summary[layer_key]["condition_label_summary"][f"{condition}__label_{int(label)}"] = {
                    "primary_score": score_stats,
                    "subspace_norm": norm_stats,
                }

        for i, sample_id in enumerate(ids):
            sample_rows.append(
                {
                    "layer": int(layer),
                    "id": sample_id,
                    "condition": conditions[i],
                    "label": int(labels[i]),
                    "intent_text": intent_texts[i],
                    "intent_id": intent_ids[i],
                    "intent_family": intent_families[i],
                    "image_role": image_roles[i],
                    "image_path": image_paths[i],
                    "source": sources[i],
                    "primary_score": float(primary_score[i]),
                    "subspace_norm": float(subspace_norm[i]),
                }
            )

    with (args.out_dir / "subspace_scores_by_sample.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "id",
                "condition",
                "label",
                "intent_text",
                "intent_id",
                "intent_family",
                "image_role",
                "image_path",
                "source",
                "primary_score",
                "subspace_norm",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    with (args.out_dir / "subspace_score_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)

    report = [
        "# Subspace Score Report",
        "",
        f"Activations: `{args.activations}`",
        f"Subspace: `{args.subspace}`",
        "",
        "Scores are projections onto an already fitted intent subspace.",
        "",
    ]
    if metric_rows:
        report.extend(
            [
                "## Label Metrics",
                "",
                "Metrics are reported because this activation file contains matched target and benign labels.",
                "",
                markdown_table(metric_rows, ["layer", "score", "auc", "average_precision", "balanced_acc", "threshold"]),
                "",
                "## Condition Gaps",
                "",
            ]
        )
        for layer_key, rows in gap_rows_by_layer.items():
            report.append(f"### {layer_key}")
            report.append("")
            report.append(markdown_table(rows, ["condition", "n_pos", "n_neg", "pos_mean", "neg_mean", "gap", "auc"]))
            report.append("")
        report.append("## Intent Gaps")
        report.append("")
        for layer_key, rows in intent_gap_rows_by_layer.items():
            report.append(f"### {layer_key}")
            report.append("")
            report.append(markdown_table(rows, ["condition", "n_pos", "n_neg", "pos_mean", "neg_mean", "gap", "auc"]))
            report.append("")
    else:
        report.extend(
            [
                "No binary labels were found, so this report is distributional only.",
                "",
            ]
        )

    report.extend(
        [
            "## Condition/Label Summary",
            "",
            markdown_table(
                summary_rows,
                ["layer", "condition", "label", "n", "primary_mean", "primary_std", "primary_median", "norm_mean", "norm_std"],
            ),
            "",
        ]
    )
    (args.out_dir / "subspace_score_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote score report to {args.out_dir / 'subspace_score_report.md'}")


if __name__ == "__main__":
    main()
