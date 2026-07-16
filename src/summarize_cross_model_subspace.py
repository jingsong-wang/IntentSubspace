import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def collect_model_result(model_dir: Path, visualization_subdir: str = "visualizations") -> dict[str, Any]:
    metrics_path = model_dir / "fit_by_condition" / "subspace_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = read_json(metrics_path)
    layer_keys = sorted(metrics.keys(), key=lambda item: int(re.search(r"\d+", item).group(0)) if re.search(r"\d+", item) else 0)
    layer_key = layer_keys[-1]
    layer_metrics = metrics[layer_key]
    loco = layer_metrics["loco"]
    folds = loco.get("folds", [])
    diag = layer_metrics.get("delta_diagnostics", {})

    weak_conditions = sorted(folds, key=lambda row: float(row.get("auc", 0.0)))[:3]
    alignments = diag.get("condition_delta_alignment", [])
    weak_alignments = sorted(alignments, key=lambda row: float(row.get("cos_to_global_delta", 0.0)))[:3]

    score_summary_path = model_dir / "score_train" / "subspace_score_summary.json"
    score_auc = None
    score_bal_acc = None
    if score_summary_path.exists():
        score_summary = read_json(score_summary_path)
        score_layer_key = sorted(score_summary.keys())[-1]
        label_metrics = score_summary[score_layer_key].get("label_metrics", {})
        primary = label_metrics.get("primary_score", {})
        score_auc = primary.get("auc")
        score_bal_acc = primary.get("balanced_acc")

    visual_path = model_dir / visualization_subdir / "visualization_summary.json"
    if not visual_path.exists() and visualization_subdir != "visualizations":
        visual_path = model_dir / "visualizations" / "visualization_summary.json"
    visual_metrics = {}
    if visual_path.exists():
        visual = read_json(visual_path)
        for space in ["raw", "subspace_coords", "residual_without_subspace"]:
            row = visual.get("spaces", {}).get(space, {})
            visual_metrics[f"{space}_fisher"] = row.get("label_fisher_ratio")
            visual_metrics[f"{space}_silhouette"] = row.get("label_silhouette")
            cats = row.get("categorical_separation", {})
            for field in [
                "condition",
                "image_role",
                "prompt_form",
                "prompt_strategy",
                "prompt_category",
                "carrier_type",
                "intent_family",
                "response_outcome",
                "refusal_state",
                "judge_score_label",
                "label_response_outcome",
            ]:
                visual_metrics[f"{space}_{field}_dispersion"] = cats.get(field, {}).get("dispersion_ratio")
        raw_label = visual_metrics.get("raw_fisher")
        sub_label = visual_metrics.get("subspace_coords_fisher")
        if raw_label not in (None, 0) and sub_label is not None:
            visual_metrics["label_fisher_gain"] = sub_label / raw_label
        for field in ["condition", "image_role", "prompt_form", "prompt_strategy", "carrier_type"]:
            raw_value = visual_metrics.get(f"raw_{field}_dispersion")
            sub_value = visual_metrics.get(f"subspace_coords_{field}_dispersion")
            if raw_value not in (None, 0) and sub_value is not None:
                visual_metrics[f"{field}_suppression"] = 1.0 - (sub_value / raw_value)

    return {
        "model_dir": model_dir.name,
        "layer": layer_key.replace("layer_", ""),
        "loco_auc": loco.get("auc"),
        "loco_ap": loco.get("average_precision"),
        "loco_acc": loco.get("accuracy"),
        "loco_bal_acc": loco.get("balanced_accuracy"),
        "score_auc": score_auc,
        "score_bal_acc": score_bal_acc,
        "n_pairs": diag.get("n_pairs"),
        "mean_delta_norm": diag.get("mean_delta_norm"),
        "weak_conditions": weak_conditions,
        "weak_alignments": weak_alignments,
        **visual_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--visualization-subdir", default="visualizations")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    rows = []
    for child in sorted(args.run_dir.iterdir()):
        if child.is_dir() and (child / "fit_by_condition" / "subspace_metrics.json").exists():
            rows.append(collect_model_result(child, visualization_subdir=args.visualization_subdir))

    out_json = args.out_json or (args.run_dir / "cross_model_subspace_summary.json")
    out_md = args.out_md or (args.run_dir / "cross_model_subspace_summary.md")
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Cross-Model Intent Subspace Summary",
        "",
        "## Overall",
        "",
        markdown_table(
            rows,
            [
                "model_dir",
                "layer",
                "loco_auc",
                "loco_ap",
                "loco_bal_acc",
                "score_auc",
                "score_bal_acc",
                "n_pairs",
                "mean_delta_norm",
                "raw_fisher",
                "subspace_coords_fisher",
                "residual_without_subspace_fisher",
                "label_fisher_gain",
                "condition_suppression",
                "image_role_suppression",
                "prompt_form_suppression",
                "prompt_strategy_suppression",
                "carrier_type_suppression",
            ],
        ),
        "",
        "## Nuisance Dispersion",
        "",
        "Dispersion is between-group / within-group variance after standardization. "
        "Suppression is `1 - subspace_coords/raw`; positive values mean the nuisance variable "
        "is less dominant inside the intent coordinates.",
        "",
        markdown_table(
            rows,
            [
                "model_dir",
                "raw_condition_dispersion",
                "subspace_coords_condition_dispersion",
                "raw_image_role_dispersion",
                "subspace_coords_image_role_dispersion",
                "raw_prompt_form_dispersion",
                "subspace_coords_prompt_form_dispersion",
                "raw_prompt_strategy_dispersion",
                "subspace_coords_prompt_strategy_dispersion",
                "raw_carrier_type_dispersion",
                "subspace_coords_carrier_type_dispersion",
            ],
        ),
        "",
        "## Response Behavior Dispersion",
        "",
        "These fields are populated when visualization is run with `--sample-results` from generation/judge outputs. "
        "A strong response-outcome cluster inside `subspace_coords` suggests refusal/compliance behavior may be mixed "
        "with the harmful-intent direction.",
        "",
        markdown_table(
            rows,
            [
                "model_dir",
                "raw_response_outcome_dispersion",
                "subspace_coords_response_outcome_dispersion",
                "residual_without_subspace_response_outcome_dispersion",
                "raw_refusal_state_dispersion",
                "subspace_coords_refusal_state_dispersion",
                "raw_judge_score_label_dispersion",
                "subspace_coords_judge_score_label_dispersion",
                "raw_label_response_outcome_dispersion",
                "subspace_coords_label_response_outcome_dispersion",
            ],
        ),
        "",
        "## Weak Conditions",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['model_dir']}",
                "",
                markdown_table(row["weak_conditions"], ["held_out_condition", "n_test", "auc", "balanced_acc"]),
                "",
                "Weakest condition alignments:",
                "",
                markdown_table(row["weak_alignments"], ["condition", "n_pairs", "delta_norm", "cos_to_global_delta"]),
                "",
            ]
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote cross-model summary to {out_json} and {out_md}")


if __name__ == "__main__":
    main()
