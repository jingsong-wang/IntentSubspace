from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from intentguard.detector import (
    CISRDetector,
    build_detector_features,
    role_categories_from_train,
    standardize_fit,
    train_tiny_mlp,
)
from intentguard.io import read_jsonl, write_json, write_jsonl
from intentguard.subspace import (
    candidate_thresholds,
    fit_paired_basis,
    metrics_at_threshold,
    project_scores,
    score_metrics,
)


SPLITS = ("train", "validation", "calibration", "test")


def safe_float(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def choose_coverage_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_tpr: float,
    target_fpr: float,
    confidence: float,
) -> tuple[float, dict[str, Any]]:
    """Choose the highest-specificity threshold meeting conservative recall coverage."""
    positive_n = int(np.sum(labels == 1))
    if positive_n == 0 or int(np.sum(labels == 0)) == 0:
        raise ValueError("Threshold calibration requires both target and benign samples.")
    z = NormalDist().inv_cdf(confidence)
    best: tuple[tuple[float, ...], float, dict[str, Any]] | None = None
    for threshold in candidate_thresholds(probabilities):
        metrics = metrics_at_threshold(labels, probabilities, float(threshold))
        lower = wilson_interval(int(metrics["tp"]), positive_n, z=z)[0]
        coverage_met = lower >= target_tpr
        empirical_met = float(metrics["tpr"]) >= target_tpr
        fpr_met = float(metrics["fpr"]) <= target_fpr
        if coverage_met and fpr_met:
            tier = 3.0
        elif coverage_met:
            tier = 2.0
        elif empirical_met:
            tier = 1.0
        else:
            tier = 0.0
        key = (
            tier,
            -float(metrics["fpr"]),
            float(metrics["tpr"]),
            float(threshold),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    assert best is not None
    _, threshold, metrics = best
    lower = wilson_interval(int(metrics["tp"]), positive_n, z=z)[0]
    return threshold, {
        **metrics,
        "target_tpr": target_tpr,
        "target_fpr": target_fpr,
        "coverage_confidence": confidence,
        "tpr_lower_confidence_bound": lower,
        "coverage_constraint_met": bool(lower >= target_tpr),
        "fpr_constraint_met": bool(float(metrics["fpr"]) <= target_fpr),
        "fallback_to_empirical_coverage": bool(
            lower < target_tpr and float(metrics["tpr"]) >= target_tpr
        ),
    }


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    result = score_metrics(labels.astype(int), probabilities.astype(float), threshold)
    metrics = dict(result["metrics_at_threshold"])
    metrics["auc"] = safe_float(result["auc"])
    metrics["average_precision"] = safe_float(result["average_precision"])
    metrics["tpr_ci95"] = wilson_interval(int(metrics["tp"]), int(metrics["positive_n"]))
    metrics["fpr_ci95"] = wilson_interval(int(metrics["fp"]), int(metrics["negative_n"]))
    return metrics


def group_report(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    indices: np.ndarray,
    key: str,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in indices.tolist():
        grouped[str(rows[index].get(key, ""))].append(index)
    report = {}
    for group, group_indices in sorted(grouped.items()):
        idx = np.asarray(group_indices, dtype=int)
        report[group] = classification_metrics(labels[idx], probabilities[idx], threshold)
    return report


def response_label_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {str(row.get("id")): row for row in read_jsonl(path)}


def successful_attack_report(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    responses: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not responses:
        return None
    successful = []
    for index in indices.tolist():
        row = rows[index]
        judged = responses.get(str(row.get("id")))
        if int(row.get("label", 0)) == 1 and judged and bool(judged.get("judge_harmful")):
            successful.append(index)
    detected = sum(float(probabilities[index]) >= threshold for index in successful)
    return {
        "successful_attack_n": len(successful),
        "detected_n": int(detected),
        "recall": detected / len(successful) if successful else None,
        "recall_ci95": wilson_interval(int(detected), len(successful)),
    }


def validate_protocol(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset contains duplicate sample ids.")
    pair_splits: dict[str, set[str]] = defaultdict(set)
    template_splits: dict[str, set[str]] = defaultdict(set)
    pair_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split = str(row.get("evaluation_split", ""))
        if split not in SPLITS:
            raise ValueError(f"Sample {row['id']} has invalid evaluation_split={split!r}")
        pair_key = str(row["pair_key"])
        pair_splits[pair_key].add(split)
        pair_rows[pair_key].append(row)
        template_splits[str(row["template_id"])].add(split)
    leaking_pairs = [key for key, values in pair_splits.items() if len(values) != 1]
    leaking_templates = [key for key, values in template_splits.items() if len(values) != 1]
    if leaking_pairs or leaking_templates:
        raise ValueError(
            f"Split leakage detected: pairs={leaking_pairs[:5]} templates={leaking_templates[:5]}"
        )
    malformed_pairs = []
    paired_fields = (
        "intent_family",
        "condition",
        "template_id",
        "wrapper_family",
        "carrier_type",
        "image_role",
        "evaluation_split",
    )
    for pair_key, items in pair_rows.items():
        labels = sorted(int(item["label"]) for item in items)
        inconsistent = [
            field for field in paired_fields if len({str(item.get(field, "")) for item in items}) != 1
        ]
        if len(items) != 2 or labels != [0, 1] or inconsistent:
            malformed_pairs.append(
                {
                    "pair_key": pair_key,
                    "row_count": len(items),
                    "labels": labels,
                    "inconsistent_fields": inconsistent,
                }
            )
    if malformed_pairs:
        raise ValueError(f"Malformed counterfactual pairs: {malformed_pairs[:5]}")

    sample_counts = Counter(str(row["evaluation_split"]) for row in rows)
    pair_counts = Counter(next(iter(splits)) for splits in pair_splits.values())
    template_counts = Counter(next(iter(splits)) for splits in template_splits.values())
    return {
        "pair_count": len(pair_splits),
        "template_count": len(template_splits),
        "pair_leakage_count": 0,
        "template_leakage_count": 0,
        "malformed_pair_count": 0,
        "split_sample_counts": {split: sample_counts[split] for split in SPLITS},
        "split_pair_counts": {split: pair_counts[split] for split in SPLITS},
        "split_template_counts": {split: template_counts[split] for split in SPLITS},
    }


def layer_features(
    activations: np.ndarray,
    anchor_activations: np.ndarray,
    has_anchor: np.ndarray,
    image_roles: np.ndarray,
    labels: np.ndarray,
    pair_keys: np.ndarray,
    train_mask: np.ndarray,
    layer_index: int,
    rank: int,
    role_categories: list[str],
) -> dict[str, Any]:
    raw = activations[:, layer_index, :].astype(np.float64)
    anchor = anchor_activations[:, layer_index, :].astype(np.float64)
    center = raw[train_mask].mean(axis=0)
    basis, diagnostics = fit_paired_basis(
        raw[train_mask],
        labels[train_mask],
        pair_keys[train_mask],
        rank,
    )
    raw_coords, _ = project_scores(raw, center, basis)
    residual_vectors = np.zeros_like(raw)
    residual_vectors[has_anchor] = raw[has_anchor] - anchor[has_anchor]
    anchored_train = train_mask & has_anchor
    residual_center = (
        residual_vectors[anchored_train].mean(axis=0)
        if np.any(anchored_train)
        else np.zeros(raw.shape[1], dtype=np.float64)
    )
    residual_coords = np.zeros((len(raw), rank), dtype=np.float64)
    if np.any(has_anchor):
        residual_coords[has_anchor] = (residual_vectors[has_anchor] - residual_center[None, :]) @ basis.T
    features = build_detector_features(
        raw_coords,
        residual_coords,
        has_anchor,
        image_roles,
        role_categories,
    )
    standardized_train, feature_mean, feature_std = standardize_fit(features[train_mask])
    standardized = (features - feature_mean[None, :]) / feature_std[None, :]
    return {
        "basis": basis,
        "center": center,
        "residual_center": residual_center,
        "raw_coords": raw_coords,
        "residual_coords": residual_coords,
        "features": standardized,
        "train_features": standardized_train,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "diagnostics": diagnostics,
    }


def ridge_validation_auc(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    l2: float = 1e-2,
) -> float:
    augmented = np.concatenate([train_features, np.ones((len(train_features), 1))], axis=1)
    identity = np.eye(augmented.shape[1])
    identity[-1, -1] = 0.0
    weights = np.linalg.pinv(augmented.T @ augmented + l2 * identity) @ (
        augmented.T @ (train_labels.astype(float) - 0.5)
    )
    validation_augmented = np.concatenate(
        [validation_features, np.ones((len(validation_features), 1))], axis=1
    )
    scores = validation_augmented @ weights
    return float(score_metrics(validation_labels, scores)["auc"])


def save_detector(
    path: Path,
    selected: dict[str, Any],
    layer: int,
    rank: int,
    role_categories: list[str],
    threshold: float,
    metadata: dict[str, Any],
    model_alias: str,
    anchor_prompt: str,
    uses_anchor: bool,
    target_tpr: float,
    target_fpr: float,
    coverage_confidence: float,
) -> None:
    network = selected["network"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.array(["CISR_v2_detector_v2"]),
        model_id=np.array([str(metadata.get("model", ""))]),
        model_alias=np.array([model_alias]),
        layer=np.array([layer], dtype=np.int32),
        rank=np.array([rank], dtype=np.int32),
        pooling=np.array([str(metadata.get("pooling", "last"))]),
        basis=selected["basis"],
        center=selected["center"],
        residual_center=selected["residual_center"],
        feature_mean=selected["feature_mean"],
        feature_std=selected["feature_std"],
        role_categories=np.array(role_categories),
        weight_1=network.weight_1,
        bias_1=network.bias_1,
        weight_2=network.weight_2,
        bias_2=network.bias_2,
        threshold=np.array([threshold], dtype=np.float64),
        anchor_prompt=np.array([anchor_prompt]),
        uses_anchor=np.array([uses_anchor], dtype=bool),
        calibration_target_tpr=np.array([target_tpr], dtype=np.float64),
        calibration_target_fpr=np.array([target_fpr], dtype=np.float64),
        coverage_confidence=np.array([coverage_confidence], dtype=np.float64),
    )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# CISR_v2 Detection Report",
        "",
        f"Model: `{summary['model_alias']}`",
        f"Selected layer: `{summary['selected_layer']}`",
        f"Rank: `{summary['rank']}`",
        f"Calibration threshold: `{summary['threshold']:.6f}`",
        f"Recall coverage confidence: `{summary['coverage_confidence']:.3f}`",
        "",
        "| split | n | AUROC | TPR | FPR | balanced accuracy | successful-attack recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        metrics = summary["splits"][split]
        successful = metrics.get("successful_attacks") or {}
        recall = successful.get("recall")
        recall_text = "n/a" if recall is None else f"{recall:.4f}"
        lines.append(
            f"| {split} | {metrics['n']} | {metrics['auc']:.4f} | {metrics['tpr']:.4f} | "
            f"{metrics['fpr']:.4f} | {metrics['balanced_accuracy']:.4f} | {recall_text} |"
        )
    lines.extend(
        [
            "",
            "The train split fits the basis and MLP, validation selects the layer, and calibration selects the threshold. Test data never participates in fitting or selection.",
            "",
            "## Pre-specified Calibration Operating Points",
            "",
            "| target TPR | threshold | calibration TPR | TPR lower bound | calibration FPR | coverage met | test TPR | test FPR |",
            "| ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |",
        ]
    )
    for point in summary["operating_points"]:
        lines.append(
            f"| {point['target_calibration_tpr']:.3f} | {point['threshold']:.6f} | "
            f"{point['calibration']['tpr']:.4f} | "
            f"{point['calibration_selection']['tpr_lower_confidence_bound']:.4f} | "
            f"{point['calibration']['fpr']:.4f} | "
            f"{'yes' if point['calibration_selection']['coverage_constraint_met'] else 'no'} | "
            f"{point['test']['tpr']:.4f} | {point['test']['fpr']:.4f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-alias", default="")
    parser.add_argument("--response-labels", type=Path)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--layer-candidates", type=int, default=6)
    parser.add_argument("--target-tpr", type=float, default=0.95)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--coverage-confidence", type=float, default=0.95)
    parser.add_argument("--hard-positive-weight", type=float, default=2.0)
    args = parser.parse_args()
    if args.rank <= 0:
        raise ValueError("--rank must be positive.")
    if not 0.0 <= args.target_tpr <= 1.0 or not 0.0 <= args.target_fpr <= 1.0:
        raise ValueError("--target-tpr and --target-fpr must be in [0, 1].")
    if not 0.5 < args.coverage_confidence < 1.0:
        raise ValueError("--coverage-confidence must be in (0.5, 1).")

    rows = read_jsonl(args.data)
    protocol = validate_protocol(rows)
    data = np.load(args.activations, allow_pickle=True)
    ids = data["ids"].astype(str)
    if ids.tolist() != [str(row["id"]) for row in rows]:
        raise ValueError("Activation ids do not exactly match dataset order.")
    activations = data["activations"]
    anchor_activations = data["anchor_activations"] if "anchor_activations" in data else np.zeros_like(activations)
    has_anchor = data["has_anchor"].astype(bool) if "has_anchor" in data else np.zeros(len(rows), dtype=bool)
    layers = data["layers"].astype(int)
    labels = data["labels"].astype(int)
    pair_keys = data["pair_keys"].astype(str)
    image_roles = data["image_roles"].astype(str)
    split_names = np.array([str(row["evaluation_split"]) for row in rows])
    split_masks = {split: split_names == split for split in SPLITS}
    for split, mask in split_masks.items():
        if not np.any(mask) or len(np.unique(labels[mask])) != 2:
            raise ValueError(f"Split {split} must contain both labels.")

    train_mask = split_masks["train"]
    validation_mask = split_masks["validation"]
    calibration_mask = split_masks["calibration"]
    role_categories = role_categories_from_train(image_roles[train_mask])
    responses = response_label_map(args.response_labels)
    sample_weight = np.ones(int(train_mask.sum()), dtype=np.float64)
    train_indices = np.where(train_mask)[0]
    if responses and args.hard_positive_weight != 1.0:
        for local_index, global_index in enumerate(train_indices.tolist()):
            row = rows[global_index]
            judged = responses.get(str(row["id"]))
            if int(row["label"]) == 1 and judged and bool(judged.get("judge_harmful")):
                sample_weight[local_index] *= args.hard_positive_weight

    prepared_layers = []
    for layer_index, layer in enumerate(layers.tolist()):
        prepared = layer_features(
            activations,
            anchor_activations,
            has_anchor,
            image_roles,
            labels,
            pair_keys,
            train_mask,
            layer_index,
            args.rank,
            role_categories,
        )
        prepared["layer"] = int(layer)
        prepared["layer_index"] = int(layer_index)
        prepared["ridge_validation_auc"] = ridge_validation_auc(
            prepared["features"][train_mask],
            labels[train_mask],
            prepared["features"][validation_mask],
            labels[validation_mask],
        )
        prepared_layers.append(prepared)

    candidate_count = max(1, min(args.layer_candidates, len(prepared_layers)))
    candidates = sorted(
        prepared_layers,
        key=lambda item: item["ridge_validation_auc"],
        reverse=True,
    )[:candidate_count]
    candidate_reports = []
    for candidate_index, candidate in enumerate(candidates):
        network, training = train_tiny_mlp(
            candidate["features"][train_mask],
            labels[train_mask],
            sample_weight=sample_weight,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            seed=args.seed + candidate_index,
        )
        probabilities = network.predict_proba(candidate["features"])
        validation_metrics = classification_metrics(
            labels[validation_mask], probabilities[validation_mask], 0.5
        )
        candidate["network"] = network
        candidate["probabilities"] = probabilities
        candidate["training"] = training
        candidate["validation_metrics"] = validation_metrics
        candidate_reports.append(
            {
                "layer": candidate["layer"],
                "ridge_validation_auc": candidate["ridge_validation_auc"],
                "mlp_validation_auc": validation_metrics["auc"],
                "mlp_validation_average_precision": validation_metrics["average_precision"],
                "mlp_validation_balanced_accuracy_at_0_5": validation_metrics["balanced_accuracy"],
                "training": training,
            }
        )

    selected = max(
        candidates,
        key=lambda item: (
            -math.inf if item["validation_metrics"]["auc"] is None else item["validation_metrics"]["auc"],
            -math.inf
            if item["validation_metrics"]["average_precision"] is None
            else item["validation_metrics"]["average_precision"],
            -item["training"]["best_weighted_bce"],
        ),
    )
    probabilities = selected["probabilities"]
    threshold, calibration_selection = choose_coverage_threshold(
        labels[calibration_mask],
        probabilities[calibration_mask],
        target_tpr=args.target_tpr,
        target_fpr=args.target_fpr,
        confidence=args.coverage_confidence,
    )
    threshold = float(threshold)
    selected["threshold"] = threshold
    metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
    anchor_prompt = str(metadata.get("multimodal_anchor_prompt", ""))
    detector_path = args.out_dir / "detector.npz"
    save_detector(
        detector_path,
        selected,
        selected["layer"],
        args.rank,
        role_categories,
        threshold,
        metadata,
        args.model_alias,
        anchor_prompt,
        bool(np.any(has_anchor)),
        args.target_tpr,
        args.target_fpr,
        args.coverage_confidence,
    )

    split_reports = {}
    for split in SPLITS:
        indices = np.where(split_masks[split])[0]
        report = classification_metrics(labels[indices], probabilities[indices], threshold)
        report["successful_attacks"] = successful_attack_report(
            rows, indices, probabilities, threshold, responses
        )
        split_reports[split] = report

    operating_points = []
    for target_recall in sorted({0.90, 0.95, 0.975, 0.99, float(args.target_tpr)}):
        point_threshold, point_selection = choose_coverage_threshold(
            labels[calibration_mask],
            probabilities[calibration_mask],
            target_tpr=target_recall,
            target_fpr=args.target_fpr,
            confidence=args.coverage_confidence,
        )
        calibration_point = classification_metrics(
            labels[calibration_mask], probabilities[calibration_mask], point_threshold
        )
        test_point = classification_metrics(
            labels[split_masks["test"]], probabilities[split_masks["test"]], point_threshold
        )
        operating_points.append(
            {
                "target_calibration_tpr": target_recall,
                "threshold": float(point_threshold),
                "calibration": calibration_point,
                "calibration_selection": point_selection,
                "test": test_point,
            }
        )

    test_indices = np.where(split_masks["test"])[0]
    summary = {
        "format_version": "CISR_v2_detection_protocol_v2",
        "model_alias": args.model_alias,
        "model_id": str(metadata.get("model", "")),
        "activations": str(args.activations),
        "data": str(args.data),
        "detector": str(detector_path),
        "rank": args.rank,
        "selected_layer": int(selected["layer"]),
        "threshold": threshold,
        "threshold_objective": "calibration_lower_confidence_tpr_coverage",
        "target_tpr": args.target_tpr,
        "target_fpr": args.target_fpr,
        "coverage_confidence": args.coverage_confidence,
        "calibration_selection": calibration_selection,
        "protocol": protocol,
        "role_categories": role_categories,
        "uses_multimodal_anchor": bool(np.any(has_anchor)),
        "hard_positive_weight": args.hard_positive_weight,
        "candidate_layers": candidate_reports,
        "operating_points": operating_points,
        "splits": split_reports,
        "test_by_condition": group_report(rows, labels, probabilities, threshold, test_indices, "condition"),
        "test_by_intent_family": group_report(rows, labels, probabilities, threshold, test_indices, "intent_family"),
        "test_by_image_role": group_report(rows, labels, probabilities, threshold, test_indices, "image_role"),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "detection_summary.json", summary)
    write_report(args.out_dir / "detection_report.md", summary)

    detection_rows = []
    for index, row in enumerate(rows):
        judged = responses.get(str(row["id"]), {})
        detection_rows.append(
            {
                **row,
                "cisr_probability": float(probabilities[index]),
                "cisr_threshold": threshold,
                "cisr_detected": bool(probabilities[index] >= threshold),
                "cisr_layer": int(selected["layer"]),
                "cisr_coordinates": selected["raw_coords"][index].astype(float).tolist(),
                "cisr_residual_coordinates": selected["residual_coords"][index].astype(float).tolist(),
                "cisr_has_anchor": bool(has_anchor[index]),
                "original_judge_harmful": judged.get("judge_harmful"),
                "original_judge_harmfulness_score": judged.get("judge_harmfulness_score"),
                "original_response_outcome": judged.get("response_outcome"),
            }
        )
    write_jsonl(args.out_dir / "detection_results.jsonl", detection_rows)

    # Load once to verify that the serialized artifact is self-contained.
    loaded = CISRDetector.load(detector_path)
    if loaded.layer != int(selected["layer"]) or loaded.rank != args.rank:
        raise RuntimeError("Serialized detector failed round-trip validation.")
    print(f"Wrote CISR_v2 detector to {detector_path}")
    print(
        f"Selected layer={selected['layer']} validation_auc={split_reports['validation']['auc']:.4f} "
        f"calibration_tpr={split_reports['calibration']['tpr']:.4f} "
        f"calibration_fpr={split_reports['calibration']['fpr']:.4f} "
        f"test_auc={split_reports['test']['auc']:.4f}"
    )


if __name__ == "__main__":
    main()
