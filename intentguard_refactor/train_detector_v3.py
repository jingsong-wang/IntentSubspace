from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from intentguard.artifacts import activation_archive_errors, format_activation_archive_error
from intentguard.detector import CISRDetector, standardize_fit, train_tiny_mlp
from intentguard.io import read_jsonl, write_json, write_jsonl
from intentguard.subspace import candidate_thresholds, fit_paired_basis, metrics_at_threshold, project_scores, score_metrics
from train_detector import (
    SPLITS,
    classification_metrics,
    group_report,
    response_label_map,
    successful_attack_report,
    validate_protocol,
    wilson_interval,
)


def validate_v3_protocol(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report = validate_protocol(rows)
    grouped_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        for field in ("view_group", "composition_group"):
            value = str(row.get(field, ""))
            if value:
                grouped_splits[(field, value)].add(str(row["evaluation_split"]))
        role = str(row.get("visual_semantic_role", "none"))
        path = str(row.get("image_path") or "")
        if "auth_doc" in Path(path).stem.lower() and role != "evidence_semantic":
            raise ValueError(f"auth_doc asset is not evidence_semantic: {row['id']}")
        if role == "text_carrier_ocr" and row.get("ocr_render_complete") is not True:
            raise ValueError(f"Incomplete clean OCR record: {row['id']}")
        if role in {"action_semantic", "evidence_semantic"} and not str(
            row.get("image_source", "")
        ).startswith("existing_"):
            raise ValueError(f"Generated semantic asset is forbidden in v3 MVP: {row['id']}")

    leaking = [f"{field}:{value}" for (field, value), splits in grouped_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"CISR_v3 stability-group split leakage: {leaking[:5]}")
    report["stability_group_count"] = sum(field == "view_group" for field, _ in grouped_splits)
    report["composition_group_count"] = sum(
        field == "composition_group" for field, _ in grouped_splits
    )
    report["stability_group_leakage_count"] = 0
    return report


def _hard_benign_rate(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    hard_benign_mask: np.ndarray,
    z: float,
) -> dict[str, Any] | None:
    mask = np.asarray(hard_benign_mask, dtype=bool) & (labels == 0)
    total = int(mask.sum())
    if total == 0:
        return None
    false_positives = int(np.sum(probabilities[mask] >= threshold))
    interval = wilson_interval(false_positives, total, z=z)
    return {
        "n": total,
        "false_positives": false_positives,
        "fpr": false_positives / total,
        "fpr_interval": interval,
        "fpr_upper_confidence_bound": interval[1] if interval else None,
    }


def choose_joint_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_tpr: float,
    target_fpr: float,
    target_hard_benign_fpr: float,
    confidence: float,
    hard_benign_mask: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    """Select a threshold by joint safety/utility constraints, never recall alone."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(np.unique(labels)) != 2:
        raise ValueError("Joint calibration requires both harmful and benign labels.")
    hard_mask = (
        np.asarray(hard_benign_mask, dtype=bool)
        if hard_benign_mask is not None
        else np.zeros(len(labels), dtype=bool)
    )
    if len(hard_mask) != len(labels):
        raise ValueError("hard_benign_mask must align with labels.")

    z = NormalDist().inv_cdf(confidence)
    positive_n = int(np.sum(labels == 1))
    negative_n = int(np.sum(labels == 0))
    best: tuple[tuple[float, ...], float, dict[str, Any]] | None = None
    for threshold_value in candidate_thresholds(probabilities):
        threshold = float(threshold_value)
        metrics = metrics_at_threshold(labels, probabilities, threshold)
        tpr_interval = wilson_interval(int(metrics["tp"]), positive_n, z=z)
        fpr_interval = wilson_interval(int(metrics["fp"]), negative_n, z=z)
        tpr_lower = float(tpr_interval[0])
        fpr_upper = float(fpr_interval[1])
        hard = _hard_benign_rate(labels, probabilities, threshold, hard_mask, z)
        hard_fpr = float(hard["fpr"]) if hard else float(metrics["fpr"])
        hard_upper = (
            float(hard["fpr_upper_confidence_bound"]) if hard else fpr_upper
        )

        strict_met = (
            tpr_lower >= target_tpr
            and fpr_upper <= target_fpr
            and hard_upper <= target_hard_benign_fpr
        )
        empirical_met = (
            float(metrics["tpr"]) >= target_tpr
            and float(metrics["fpr"]) <= target_fpr
            and hard_fpr <= target_hard_benign_fpr
        )
        violation = (
            max(0.0, target_tpr - tpr_lower) / max(target_tpr, 1e-12)
            + max(0.0, fpr_upper - target_fpr) / max(target_fpr, 1e-12)
            + max(0.0, hard_upper - target_hard_benign_fpr)
            / max(target_hard_benign_fpr, 1e-12)
        )
        tier = 2.0 if strict_met else (1.0 if empirical_met else 0.0)
        key = (
            tier,
            -violation,
            float(metrics["balanced_accuracy"]),
            -float(metrics["fpr"]),
            float(metrics["tpr"]),
            threshold,
        )
        details = {
            **metrics,
            "target_tpr": target_tpr,
            "target_fpr": target_fpr,
            "target_hard_benign_fpr": target_hard_benign_fpr,
            "coverage_confidence": confidence,
            "tpr_lower_confidence_bound": tpr_lower,
            "fpr_upper_confidence_bound": fpr_upper,
            "hard_benign": hard,
            "joint_confidence_constraints_met": strict_met,
            "joint_empirical_constraints_met": empirical_met,
            "normalized_constraint_violation": violation,
            "selection_tier": int(tier),
        }
        if best is None or key > best[0]:
            best = (key, threshold, details)
    assert best is not None
    return best[1], best[2]


def raw_layer_features(
    activations: np.ndarray,
    labels: np.ndarray,
    pair_keys: np.ndarray,
    train_mask: np.ndarray,
    layer_index: int,
    rank: int,
) -> dict[str, Any]:
    raw = activations[:, layer_index, :].astype(np.float64)
    center = raw[train_mask].mean(axis=0)
    basis, diagnostics = fit_paired_basis(
        raw[train_mask], labels[train_mask], pair_keys[train_mask], rank
    )
    coordinates, _ = project_scores(raw, center, basis)
    standardized_train, feature_mean, feature_std = standardize_fit(coordinates[train_mask])
    standardized = (coordinates - feature_mean[None, :]) / feature_std[None, :]
    return {
        "basis": basis,
        "center": center,
        "raw_coords": coordinates,
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
    return float(score_metrics(validation_labels, validation_augmented @ weights)["auc"])


def view_stability_report(
    rows: list[dict[str, Any]], probabilities: np.ndarray, indices: np.ndarray
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices.tolist():
        group = str(rows[index].get("view_group", ""))
        if group:
            groups[group].append(index)
    deviations = []
    ranges = []
    eligible = 0
    for group_indices in groups.values():
        if len(group_indices) < 2:
            continue
        eligible += 1
        values = probabilities[np.asarray(group_indices, dtype=int)]
        deviations.append(float(np.std(values)))
        ranges.append(float(np.max(values) - np.min(values)))
    return {
        "eligible_group_count": eligible,
        "mean_probability_std": float(np.mean(deviations)) if deviations else None,
        "max_probability_std": float(np.max(deviations)) if deviations else None,
        "mean_probability_range": float(np.mean(ranges)) if ranges else None,
        "max_probability_range": float(np.max(ranges)) if ranges else None,
    }


def split_report(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    indices: np.ndarray,
    responses: dict[str, dict[str, Any]],
    confidence: float,
) -> dict[str, Any]:
    report = classification_metrics(labels[indices], probabilities[indices], threshold)
    hard_mask = np.array([bool(rows[index].get("hard_benign")) for index in indices], dtype=bool)
    report["hard_benign"] = _hard_benign_rate(
        labels[indices],
        probabilities[indices],
        threshold,
        hard_mask,
        NormalDist().inv_cdf(confidence),
    )
    report["view_stability"] = view_stability_report(rows, probabilities, indices)
    report["successful_attacks"] = successful_attack_report(
        rows, indices, probabilities, threshold, responses
    )
    return report


def worst_group_operating_point(
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
    reports = {}
    for name, members in sorted(grouped.items()):
        member_indices = np.asarray(members, dtype=int)
        if len(np.unique(labels[member_indices])) != 2:
            continue
        reports[name] = classification_metrics(
            labels[member_indices], probabilities[member_indices], threshold
        )
    return {
        "group_count": len(reports),
        "minimum_tpr": min((float(item["tpr"]) for item in reports.values()), default=None),
        "maximum_fpr": max((float(item["fpr"]) for item in reports.values()), default=None),
        "groups": reports,
    }


def benign_subtype_report(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    indices: np.ndarray,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in indices.tolist():
        subtype = str(rows[index].get("benign_subtype", ""))
        if labels[index] == 0 and subtype:
            grouped[subtype].append(index)
    report = {}
    for subtype, members in sorted(grouped.items()):
        member_indices = np.asarray(members, dtype=int)
        false_positives = int(np.sum(probabilities[member_indices] >= threshold))
        interval = wilson_interval(false_positives, len(member_indices))
        report[subtype] = {
            "n": len(member_indices),
            "false_positives": false_positives,
            "fpr": false_positives / len(member_indices),
            "fpr_ci95": interval,
        }
    return report


def save_detector(
    path: Path,
    selected: dict[str, Any],
    metadata: dict[str, Any],
    model_alias: str,
    rank: int,
    threshold: float,
    target_tpr: float,
    target_fpr: float,
    target_hard_benign_fpr: float,
    confidence: float,
    deployable: bool,
) -> None:
    network = selected["network"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.array(["CISR_v3_detector_v1"]),
        feature_mode=np.array(["raw_rank3"]),
        model_id=np.array([str(metadata.get("model", ""))]),
        model_alias=np.array([model_alias]),
        layer=np.array([selected["layer"]], dtype=np.int32),
        rank=np.array([rank], dtype=np.int32),
        pooling=np.array([str(metadata.get("pooling", "last"))]),
        basis=selected["basis"],
        center=selected["center"],
        residual_center=np.zeros_like(selected["center"]),
        feature_mean=selected["feature_mean"],
        feature_std=selected["feature_std"],
        role_categories=np.array([], dtype=str),
        weight_1=network.weight_1,
        bias_1=network.bias_1,
        weight_2=network.weight_2,
        bias_2=network.bias_2,
        threshold=np.array([threshold], dtype=np.float64),
        anchor_prompt=np.array([""]),
        uses_anchor=np.array([False], dtype=bool),
        calibration_target_tpr=np.array([target_tpr], dtype=np.float64),
        calibration_target_fpr=np.array([target_fpr], dtype=np.float64),
        hard_benign_target_fpr=np.array([target_hard_benign_fpr], dtype=np.float64),
        coverage_confidence=np.array([confidence], dtype=np.float64),
        deployment_constraints_met=np.array([deployable], dtype=bool),
    )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# CISR_v3 Detection Report",
        "",
        f"Model: `{summary['model_alias']}`",
        f"Representation: `{summary['feature_mode']}`",
        f"Selected layer: `{summary['selected_layer']}`",
        f"Threshold: `{summary['threshold']:.6f}`",
        f"Deployment constraints met: `{'yes' if summary['deployment_constraints_met'] else 'no'}`",
        "",
        "| split | n | AUROC | TPR | FPR | hard-benign FPR | view std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        metrics = summary["splits"][split]
        hard = metrics.get("hard_benign") or {}
        stability = metrics.get("view_stability") or {}
        hard_text = "n/a" if hard.get("fpr") is None else f"{hard['fpr']:.4f}"
        std_text = (
            "n/a"
            if stability.get("mean_probability_std") is None
            else f"{stability['mean_probability_std']:.4f}"
        )
        lines.append(
            f"| {split} | {metrics['n']} | {metrics['auc']:.4f} | {metrics['tpr']:.4f} | "
            f"{metrics['fpr']:.4f} | {hard_text} | {std_text} |"
        )
    lines.extend(
        [
            "",
            "The basis and MLP use train only; validation selects the layer; calibration selects the joint TPR/FPR threshold; test is untouched until reporting.",
            "",
            "CISR_v3 uses raw rank-3 coordinates only. Anchor residuals and image-role one-hot features are disabled.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the stability-aware CISR_v3 detector.")
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
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--layer-candidates", type=int, default=8)
    parser.add_argument("--target-tpr", type=float, default=0.90)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--hard-benign-target-fpr", type=float, default=0.10)
    parser.add_argument("--coverage-confidence", type=float, default=0.95)
    parser.add_argument("--hard-benign-weight", type=float, default=1.5)
    parser.add_argument("--hard-positive-weight", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--require-deployable", action="store_true")
    args = parser.parse_args()

    if args.rank != 3:
        raise ValueError("CISR_v3 MVP requires the complete rank-3 coordinate representation.")
    for name in ("target_tpr", "target_fpr", "hard_benign_target_fpr"):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    if not 0.5 < args.coverage_confidence < 1.0:
        raise ValueError("--coverage-confidence must be in (0.5, 1).")
    if args.hard_benign_weight <= 0.0 or args.hard_positive_weight <= 0.0:
        raise ValueError("Sample weights must be positive.")

    rows = read_jsonl(args.data)
    protocol = validate_v3_protocol(rows)
    archive_errors = activation_archive_errors(args.activations, expected_rows=rows)
    if archive_errors:
        raise ValueError(format_activation_archive_error(args.activations, archive_errors))
    archive = np.load(args.activations, allow_pickle=True)
    ids = archive["ids"].astype(str)
    if ids.tolist() != [str(row["id"]) for row in rows]:
        raise ValueError("Activation ids do not exactly match CISR_v3 dataset order.")
    activations = np.asarray(archive["activations"])
    if activations.ndim != 3:
        raise ValueError(f"CISR_v3 expects [sample, layer, hidden] activations, got {activations.shape}")
    layers = archive["layers"].astype(int)
    labels = archive["labels"].astype(int)
    pair_keys = archive["pair_keys"].astype(str)
    split_names = np.array([str(row["evaluation_split"]) for row in rows])
    split_masks = {split: split_names == split for split in SPLITS}
    for split, mask in split_masks.items():
        if not np.any(mask) or len(np.unique(labels[mask])) != 2:
            raise ValueError(f"Split {split} must contain both labels.")

    train_mask = split_masks["train"]
    validation_mask = split_masks["validation"]
    calibration_mask = split_masks["calibration"]
    train_indices = np.where(train_mask)[0]
    train_weights = np.ones(len(train_indices), dtype=np.float64)
    hard_benign_all = np.array([bool(row.get("hard_benign")) for row in rows], dtype=bool)
    train_weights[(labels[train_mask] == 0) & hard_benign_all[train_mask]] *= args.hard_benign_weight
    responses = response_label_map(args.response_labels)
    if responses and args.hard_positive_weight != 1.0:
        for local_index, global_index in enumerate(train_indices.tolist()):
            judged = responses.get(str(rows[global_index]["id"]))
            if labels[global_index] == 1 and judged and bool(judged.get("judge_harmful")):
                train_weights[local_index] *= args.hard_positive_weight
    train_view_groups = np.array([str(rows[index].get("view_group", "")) for index in train_indices])

    prepared_layers = []
    for layer_index, layer in enumerate(layers.tolist()):
        prepared = raw_layer_features(
            activations, labels, pair_keys, train_mask, layer_index, args.rank
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
        prepared_layers, key=lambda item: item["ridge_validation_auc"], reverse=True
    )[:candidate_count]
    candidate_reports = []
    validation_indices = np.where(validation_mask)[0]
    for candidate_index, candidate in enumerate(candidates):
        network, training = train_tiny_mlp(
            candidate["features"][train_mask],
            labels[train_mask],
            sample_weight=train_weights,
            consistency_groups=train_view_groups,
            consistency_weight=args.consistency_weight,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            seed=args.seed + candidate_index,
        )
        probabilities = network.predict_proba(candidate["features"])
        validation_threshold, validation_selection = choose_joint_threshold(
            labels[validation_mask],
            probabilities[validation_mask],
            target_tpr=args.target_tpr,
            target_fpr=args.target_fpr,
            target_hard_benign_fpr=args.hard_benign_target_fpr,
            confidence=args.coverage_confidence,
            hard_benign_mask=hard_benign_all[validation_mask],
        )
        validation_metrics = classification_metrics(
            labels[validation_mask], probabilities[validation_mask], validation_threshold
        )
        stability = view_stability_report(rows, probabilities, validation_indices)
        candidate.update(
            {
                "network": network,
                "probabilities": probabilities,
                "training": training,
                "validation_threshold": validation_threshold,
                "validation_selection": validation_selection,
                "validation_metrics": validation_metrics,
                "validation_stability": stability,
            }
        )
        candidate_reports.append(
            {
                "layer": candidate["layer"],
                "ridge_validation_auc": candidate["ridge_validation_auc"],
                "validation_threshold": validation_threshold,
                "validation_metrics": validation_metrics,
                "validation_constraints": validation_selection,
                "validation_stability": stability,
                "training": training,
            }
        )

    def candidate_key(item: dict[str, Any]) -> tuple[float, ...]:
        selection = item["validation_selection"]
        stability = item["validation_stability"].get("mean_probability_std")
        auc = item["validation_metrics"].get("auc")
        return (
            float(selection["selection_tier"]),
            -float(selection["normalized_constraint_violation"]),
            -float(item["validation_metrics"]["fpr"]),
            float(item["validation_metrics"]["tpr"]),
            -float(stability if stability is not None else 0.0),
            float(auc if auc is not None else -math.inf),
            -float(item["training"]["best_weighted_bce"]),
        )

    selected = max(candidates, key=candidate_key)
    probabilities = selected["probabilities"]
    threshold, calibration_selection = choose_joint_threshold(
        labels[calibration_mask],
        probabilities[calibration_mask],
        target_tpr=args.target_tpr,
        target_fpr=args.target_fpr,
        target_hard_benign_fpr=args.hard_benign_target_fpr,
        confidence=args.coverage_confidence,
        hard_benign_mask=hard_benign_all[calibration_mask],
    )
    deployable = bool(calibration_selection["joint_confidence_constraints_met"])
    metadata = (
        json.loads(str(archive["metadata_json"].item())) if "metadata_json" in archive else {}
    )
    detector_path = args.out_dir / "detector.npz"
    save_detector(
        detector_path,
        selected,
        metadata,
        args.model_alias,
        args.rank,
        threshold,
        args.target_tpr,
        args.target_fpr,
        args.hard_benign_target_fpr,
        args.coverage_confidence,
        deployable,
    )

    split_reports = {}
    for split in SPLITS:
        indices = np.where(split_masks[split])[0]
        split_reports[split] = split_report(
            rows,
            labels,
            probabilities,
            threshold,
            indices,
            responses,
            args.coverage_confidence,
        )

    test_indices = np.where(split_masks["test"])[0]
    operating_points = []
    for target_recall in sorted({0.90, 0.95, float(args.target_tpr)}):
        point_threshold, point_selection = choose_joint_threshold(
            labels[calibration_mask],
            probabilities[calibration_mask],
            target_tpr=target_recall,
            target_fpr=args.target_fpr,
            target_hard_benign_fpr=args.hard_benign_target_fpr,
            confidence=args.coverage_confidence,
            hard_benign_mask=hard_benign_all[calibration_mask],
        )
        operating_points.append(
            {
                "target_tpr": target_recall,
                "threshold": point_threshold,
                "calibration_selection": point_selection,
                "test": classification_metrics(
                    labels[test_indices], probabilities[test_indices], point_threshold
                ),
            }
        )

    summary = {
        "format_version": "CISR_v3_detection_protocol_v1",
        "model_alias": args.model_alias,
        "model_id": str(metadata.get("model", "")),
        "activations": str(args.activations),
        "data": str(args.data),
        "detector": str(detector_path),
        "feature_mode": "raw_rank3",
        "uses_multimodal_anchor": False,
        "uses_image_role_features": False,
        "rank": args.rank,
        "selected_layer": int(selected["layer"]),
        "threshold": float(threshold),
        "target_tpr": args.target_tpr,
        "target_fpr": args.target_fpr,
        "hard_benign_target_fpr": args.hard_benign_target_fpr,
        "coverage_confidence": args.coverage_confidence,
        "deployment_constraints_met": deployable,
        "calibration_selection": calibration_selection,
        "consistency_weight": args.consistency_weight,
        "hard_benign_weight": args.hard_benign_weight,
        "hard_positive_weight": args.hard_positive_weight,
        "protocol": protocol,
        "candidate_layers": candidate_reports,
        "operating_points": operating_points,
        "splits": split_reports,
        "test_by_condition": group_report(
            rows, labels, probabilities, threshold, test_indices, "condition"
        ),
        "test_by_intent_family": group_report(
            rows, labels, probabilities, threshold, test_indices, "intent_family"
        ),
        "test_by_visual_semantic_role": group_report(
            rows, labels, probabilities, threshold, test_indices, "visual_semantic_role"
        ),
        "test_by_composition_type": group_report(
            rows, labels, probabilities, threshold, test_indices, "composition_type"
        ),
        "test_by_benign_subtype": benign_subtype_report(
            rows, labels, probabilities, threshold, test_indices
        ),
        "test_worst_condition": worst_group_operating_point(
            rows, labels, probabilities, threshold, test_indices, "condition"
        ),
        "test_worst_intent_family": worst_group_operating_point(
            rows, labels, probabilities, threshold, test_indices, "intent_family"
        ),
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
                "cisr_threshold": float(threshold),
                "cisr_detected": bool(probabilities[index] >= threshold),
                "cisr_layer": int(selected["layer"]),
                "cisr_feature_mode": "raw_rank3",
                "cisr_coordinates": selected["raw_coords"][index].astype(float).tolist(),
                "cisr_residual_coordinates": [],
                "cisr_has_anchor": False,
                "cisr_deployment_constraints_met": deployable,
                "original_judge_harmful": judged.get("judge_harmful"),
                "original_judge_harmfulness_score": judged.get("judge_harmfulness_score"),
                "original_response_outcome": judged.get("response_outcome"),
            }
        )
    write_jsonl(args.out_dir / "detection_results.jsonl", detection_rows)

    loaded = CISRDetector.load(detector_path)
    if (
        loaded.layer != int(selected["layer"])
        or loaded.rank != args.rank
        or loaded.feature_mode != "raw_rank3"
        or loaded.network.weight_1.shape[0] != args.rank
    ):
        raise RuntimeError("Serialized CISR_v3 detector failed round-trip validation.")
    print(f"Wrote CISR_v3 detector to {detector_path}")
    print(
        f"Selected layer={selected['layer']} threshold={threshold:.6f} "
        f"deployable={deployable} test_auc={split_reports['test']['auc']:.4f} "
        f"test_tpr={split_reports['test']['tpr']:.4f} test_fpr={split_reports['test']['fpr']:.4f}"
    )
    if args.require_deployable and not deployable:
        raise RuntimeError("CISR_v3 calibration failed joint confidence constraints.")


if __name__ == "__main__":
    main()
