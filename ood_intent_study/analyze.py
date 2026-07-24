from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import RESULT_VERSION
from .artifacts import ActivationTable, load_activation_table
from .io_utils import canonical_json, read_jsonl, sha256_text, write_json_atomic
from .metrics import (
    balanced_threshold,
    centroid_domain_auc,
    centroid_shift,
    cluster_bootstrap_intervals,
    confusion_metrics,
    fisher_ratio,
    fit_logistic,
    json_clean,
    multiclass_centroid_macro_f1,
    probe_scores,
    rbf_mmd_1d,
    roc_auc,
    score_metrics,
    standardized_centroid_probe,
    threshold_at_fpr,
)
from .panels import MODALITY_PANELS, modality_panel_mask, source_display_name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze layer-wise intent separability and OOD shift.")
    parser.add_argument("--activations", type=Path, required=True, help="Directory containing shard_*.npz.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--primary-readout", default="last")
    parser.add_argument("--domain-probe-readout", default="last")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--skip-loso", action="store_true")
    parser.add_argument("--strong-label-sensitivity", action="store_true")
    parser.add_argument(
        "--modality-panel",
        choices=MODALITY_PANELS,
        default="all",
        help="Analyze all rows, text-only rows, or multimodal rows. Applied before fitting.",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def _manifest_frame(path: Path, table: ActivationTable) -> pd.DataFrame:
    rows = list(read_jsonl(path))
    fingerprint = sha256_text("\n".join(canonical_json(row) for row in rows))
    expected = table.metadata.get("manifest_sha256")
    if expected and expected != fingerprint:
        raise ValueError(
            f"Activation/manifest fingerprint mismatch: activation={expected}, manifest={fingerprint}"
        )
    by_id = {str(row["sample_id"]): row for row in rows}
    missing = [sample_id for sample_id in table.sample_ids.tolist() if sample_id not in by_id]
    if missing:
        raise ValueError(f"Activation IDs are missing from the manifest: {missing[:5]}")
    selected = [by_id[sample_id] for sample_id in table.sample_ids.tolist()]
    frame = pd.DataFrame(selected)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Joined manifest contains duplicate sample IDs")
    frame["label"] = frame["label"].astype(int)
    frame["is_attack"] = frame["is_attack"].astype(bool)
    return frame


def _subset_activation_table(table: ActivationTable, keep: np.ndarray) -> ActivationTable:
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (len(table.sample_ids),):
        raise ValueError(
            f"Activation filter has shape {keep.shape}; expected {(len(table.sample_ids),)}"
        )
    return ActivationTable(
        sample_ids=table.sample_ids[keep],
        activations=table.activations[keep],
        readout_valid=table.readout_valid[keep],
        layers=table.layers,
        readouts=table.readouts,
        sequence_lengths=table.sequence_lengths[keep],
        image_token_counts=table.image_token_counts[keep],
        text_token_counts=table.text_token_counts[keep],
        image_widths=table.image_widths[keep],
        image_heights=table.image_heights[keep],
        rendered_prompt_sha256=table.rendered_prompt_sha256[keep],
        metadata=table.metadata,
    )


def _filter_analysis_rows(
    table: ActivationTable,
    frame: pd.DataFrame,
    modality_panel: str,
    strong_label_sensitivity: bool,
) -> tuple[ActivationTable, pd.DataFrame]:
    keep = modality_panel_mask(frame["modality"].tolist(), modality_panel)
    if strong_label_sensitivity:
        keep &= frame["label_confidence"].eq("strong").to_numpy()
    filtered_table = _subset_activation_table(table, keep)
    filtered_frame = frame.loc[keep].reset_index(drop=True)
    if filtered_frame.empty:
        strong_note = " with --strong-label-sensitivity" if strong_label_sensitivity else ""
        raise ValueError(
            f"Modality panel {modality_panel!r}{strong_note} selected zero activation rows"
        )
    return filtered_table, filtered_frame


def _standard_split_label_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    standard = ~frame["is_attack"].to_numpy(dtype=bool)
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        mask = standard & frame["split"].eq(split).to_numpy()
        values = frame.loc[mask, "label"].value_counts().astype(int).to_dict()
        counts[split] = {str(label): int(values.get(label, 0)) for label in (0, 1)}
    return counts


def _cross_counts(
    frame: pd.DataFrame,
    row_column: str,
    column_column: str,
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    counts = frame.groupby([row_column, column_column], dropna=False).size()
    for (row_value, column_value), count in counts.items():
        output.setdefault(str(row_value), {})[str(column_value)] = int(count)
    return output


def _validate_standard_split_labels(
    frame: pd.DataFrame,
    modality_panel: str,
    strong_label_sensitivity: bool,
) -> dict[str, dict[str, int]]:
    counts = _standard_split_label_counts(frame)
    missing = [
        split
        for split, split_counts in counts.items()
        if split_counts["0"] == 0 or split_counts["1"] == 0
    ]
    if missing:
        strong_note = " after --strong-label-sensitivity" if strong_label_sensitivity else ""
        raise ValueError(
            f"Modality panel {modality_panel!r}{strong_note} cannot support the standard "
            "train/validation/test protocol: every non-attack split must contain both "
            f"labels 0 and 1; missing both-class coverage in {missing}; observed counts={counts}"
        )
    return counts


def _safe_fit(X: np.ndarray, y: np.ndarray, seed: int) -> Any | None:
    if len(X) < 4 or len(np.unique(y)) != 2:
        return None
    return fit_logistic(X.astype(np.float32, copy=False), y, seed)


def _mask(frame: pd.DataFrame, valid: np.ndarray, expression: np.ndarray) -> np.ndarray:
    return np.asarray(valid, dtype=bool) & np.asarray(expression, dtype=bool)


def _metric_row(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    metrics = score_metrics(y, scores, threshold)
    metrics.update(cluster_bootstrap_intervals(y, scores, groups, threshold, bootstrap, seed))
    return metrics


def _thresholds(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "balanced": balanced_threshold(y, scores),
        "fpr_1pct": threshold_at_fpr(y, scores, 0.01),
        "fpr_5pct": threshold_at_fpr(y, scores, 0.05),
    }


def _standard_masks(frame: pd.DataFrame, valid: np.ndarray) -> dict[str, np.ndarray]:
    standard = ~frame["is_attack"].to_numpy(dtype=bool)
    return {
        split: _mask(frame, valid, standard & frame["split"].eq(split).to_numpy())
        for split in ("train", "validation", "test")
    }


def _analyze_layer(
    X: np.ndarray,
    frame: pd.DataFrame,
    valid: np.ndarray,
    layer: int,
    readout: str,
    args: argparse.Namespace,
    include_source_diagnostics: bool = True,
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    labels = frame["label"].to_numpy(dtype=int)
    groups = frame["split_group_id"].astype(str).to_numpy()
    masks = _standard_masks(frame, valid)
    train, validation, test = masks["train"], masks["validation"], masks["test"]
    if (
        len(np.unique(labels[train])) != 2
        or len(np.unique(labels[validation])) != 2
        or len(np.unique(labels[test])) != 2
    ):
        return None, [], [], [], {}
    model = _safe_fit(X[train], labels[train], args.seed + layer)
    if model is None:
        return None, [], [], [], {}
    validation_scores = probe_scores(model, X[validation])
    test_scores = probe_scores(model, X[test])
    thresholds = _thresholds(labels[validation], validation_scores)
    threshold_source = "validation"

    validation_metrics = (
        score_metrics(labels[validation], validation_scores, thresholds["balanced"])
        if validation.any()
        else {}
    )
    test_metrics = _metric_row(
        labels[test],
        test_scores,
        groups[test],
        thresholds["balanced"],
        args.bootstrap,
        args.seed + layer,
    )
    fpr1 = confusion_metrics(labels[test], test_scores, thresholds["fpr_1pct"])
    fpr5 = confusion_metrics(labels[test], test_scores, thresholds["fpr_5pct"])
    probe_row = {
        "layer": layer,
        "normalized_depth": 0.0,
        "readout": readout,
        "train_n": int(train.sum()),
        "validation_n": int(validation.sum()),
        "test_n": int(test.sum()),
        "threshold_source": threshold_source,
        "balanced_threshold": thresholds["balanced"],
        "fpr_1pct_threshold": thresholds["fpr_1pct"],
        "fpr_5pct_threshold": thresholds["fpr_5pct"],
        "validation_auroc": validation_metrics.get("auroc", float("nan")),
        "validation_auprc": validation_metrics.get("auprc", float("nan")),
        "fisher_ratio_test": fisher_ratio(X[test].astype(np.float32, copy=False), labels[test]),
        "tpr_at_validation_fpr_1pct": fpr1["tpr"],
        "observed_fpr_at_validation_fpr_1pct": fpr1["fpr"],
        "tpr_at_validation_fpr_5pct": fpr5["tpr"],
        "observed_fpr_at_validation_fpr_5pct": fpr5["fpr"],
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }

    source_rows: list[dict[str, Any]] = []
    source_label_rows: list[dict[str, Any]] = []
    standard_test = test
    diagnostic_sources = (
        sorted(frame.loc[standard_test, "source"].unique().tolist())
        if include_source_diagnostics
        else []
    )
    for source in diagnostic_sources:
        source_mask = standard_test & frame["source"].eq(source).to_numpy()
        source_scores = probe_scores(model, X[source_mask])
        metrics = _metric_row(
            labels[source_mask],
            source_scores,
            groups[source_mask],
            thresholds["balanced"],
            args.bootstrap,
            args.seed + layer + len(source_rows),
        )
        source_rows.append(
            {
                "layer": layer,
                "readout": readout,
                "source": source,
                "source_kind": "benchmark",
                **metrics,
            }
        )
        for label in sorted(np.unique(labels[source_mask]).tolist()):
            label_mask = source_mask & (labels == int(label))
            label_scores = probe_scores(model, X[label_mask])
            # These rows drive an all-layer diagnostic heatmap. Source-level rows
            # retain clustered CIs; repeating bootstrap for every label cell is
            # prohibitively expensive and does not change the plotted estimate.
            label_metrics = score_metrics(
                labels[label_mask], label_scores, thresholds["balanced"]
            )
            source_label_rows.append(
                {
                    "layer": layer,
                    "readout": readout,
                    "source": source,
                    "source_label": source_display_name(source, label),
                    "label": int(label),
                    "label_name": "harmful" if int(label) == 1 else "benign",
                    "source_kind": "benchmark",
                    "label_recall": label_metrics["tpr"] if int(label) == 1 else label_metrics["tnr"],
                    **label_metrics,
                }
            )

    attack_rows: list[dict[str, Any]] = []
    benign_panel = test & (labels == 0)
    standard_harmful = test & (labels == 1)
    if not benign_panel.any() or not standard_harmful.any():
        return (
            probe_row,
            source_rows,
            source_label_rows,
            [],
            {"model": model, "thresholds": thresholds},
        )
    standard_harmful_scores = probe_scores(model, X[standard_harmful])
    for attack in sorted(frame.loc[frame["is_attack"], "source"].unique().tolist()):
        attack_mask = valid & frame["source"].eq(attack).to_numpy()
        if not attack_mask.any():
            continue
        attack_scores = probe_scores(model, X[attack_mask])
        combined_y = np.concatenate(
            [np.zeros(int(benign_panel.sum()), dtype=int), np.ones(int(attack_mask.sum()), dtype=int)]
        )
        combined_scores = np.concatenate([probe_scores(model, X[benign_panel]), attack_scores])
        combined_groups = np.concatenate([groups[benign_panel], groups[attack_mask]])
        metrics = _metric_row(
            combined_y,
            combined_scores,
            combined_groups,
            thresholds["balanced"],
            args.bootstrap,
            args.seed + layer + len(attack_rows) * 17,
        )
        standard_std = float(np.std(standard_harmful_scores))
        shift = float(np.mean(attack_scores) - np.mean(standard_harmful_scores))
        raw_shift = centroid_shift(
            X[train].astype(np.float32, copy=False),
            labels[train],
            X[standard_harmful].astype(np.float32, copy=False),
            X[attack_mask].astype(np.float32, copy=False),
        )
        domain_auc = float("nan")
        matched_domain_auc = float("nan")
        matched_goal_n = 0
        matched_standard_sources = "{}"
        matched_standard_modalities = "{}"
        if readout == args.domain_probe_readout:
            semantic_groups = frame["semantic_group_id"].astype(str).to_numpy()
            domain_auc = centroid_domain_auc(
                X[standard_harmful].astype(np.float32, copy=False),
                X[attack_mask].astype(np.float32, copy=False),
                semantic_groups[standard_harmful],
                semantic_groups[attack_mask],
                args.seed + layer,
            )
            all_standard_harmful = (
                valid
                & ~frame["is_attack"].to_numpy(dtype=bool)
                & (labels == 1)
            )
            shared_goals = set(semantic_groups[all_standard_harmful]) & set(semantic_groups[attack_mask])
            if shared_goals:
                matched_standard = all_standard_harmful & np.isin(semantic_groups, list(shared_goals))
                matched_attack = attack_mask & np.isin(semantic_groups, list(shared_goals))
                matched_goal_n = len(shared_goals)
                matched_standard_sources = canonical_json(
                    frame.loc[matched_standard, "source"].value_counts().astype(int).to_dict()
                )
                matched_standard_modalities = canonical_json(
                    frame.loc[matched_standard, "modality"].value_counts().astype(int).to_dict()
                )
                matched_domain_auc = centroid_domain_auc(
                    X[matched_standard].astype(np.float32, copy=False),
                    X[matched_attack].astype(np.float32, copy=False),
                    semantic_groups[matched_standard],
                    semantic_groups[matched_attack],
                    args.seed + layer + 7919,
                )
        attack_rows.append(
            {
                "layer": layer,
                "readout": readout,
                "attack": attack,
                "attack_n": int(attack_mask.sum()),
                "benign_panel_n": int(benign_panel.sum()),
                "standard_harmful_n": int(standard_harmful.sum()),
                "attack_score_mean": float(np.mean(attack_scores)),
                "standard_harmful_score_mean": float(np.mean(standard_harmful_scores)),
                "score_location_shift": shift,
                "standardized_score_shift": shift / max(standard_std, 1e-12),
                "intent_score_mmd2": rbf_mmd_1d(standard_harmful_scores, attack_scores),
                "group_cv_harmful_domain_auroc": domain_auc,
                "matched_goal_group_cv_harmful_domain_auroc": matched_domain_auc,
                "matched_goal_n": matched_goal_n,
                "matched_goal_reference_scope": "all_standard_splits",
                "matched_goal_standard_sources": matched_standard_sources,
                "matched_goal_standard_modalities": matched_standard_modalities,
                **raw_shift,
                **metrics,
            }
        )
    return (
        probe_row,
        source_rows,
        source_label_rows,
        attack_rows,
        {"model": model, "thresholds": thresholds},
    )


def _loso_rows(
    X: np.ndarray,
    frame: pd.DataFrame,
    valid: np.ndarray,
    layer: int,
    readout: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = frame["label"].to_numpy(dtype=int)
    standard = ~frame["is_attack"].to_numpy(dtype=bool)
    output: list[dict[str, Any]] = []
    label_output: list[dict[str, Any]] = []
    for source in sorted(frame.loc[standard & valid, "source"].unique().tolist()):
        excluded = frame["source"].eq(source).to_numpy()
        holdout_groups = set(frame.loc[excluded, "split_group_id"].astype(str).tolist())
        group_related = frame["split_group_id"].astype(str).isin(holdout_groups).to_numpy()
        train = valid & standard & ~group_related & frame["split"].eq("train").to_numpy()
        calibration = valid & standard & ~group_related & frame["split"].eq("validation").to_numpy()
        holdout = valid & standard & excluded
        if len(np.unique(labels[train])) != 2 or not holdout.any():
            reason = "training_missing_class_or_empty_holdout"
            output.append(
                {
                    "layer": layer,
                    "readout": readout,
                    "held_out_source": source,
                    "probe": "standardized_centroid",
                    "eligible": False,
                    "ineligible_reason": reason,
                }
            )
            for label in sorted(np.unique(labels[holdout]).tolist()):
                label_output.append(
                    {
                        "layer": layer,
                        "readout": readout,
                        "held_out_source": source,
                        "held_out_source_label": source_display_name(source, label),
                        "held_out_label": int(label),
                        "label_name": "harmful" if int(label) == 1 else "benign",
                        "probe": "standardized_centroid",
                        "eligible": False,
                        "ineligible_reason": reason,
                    }
                )
            continue
        if len(np.unique(labels[calibration])) != 2:
            reason = "validation_missing_class_after_group_exclusion"
            output.append(
                {
                    "layer": layer,
                    "readout": readout,
                    "held_out_source": source,
                    "probe": "standardized_centroid",
                    "eligible": False,
                    "ineligible_reason": reason,
                }
            )
            for label in sorted(np.unique(labels[holdout]).tolist()):
                label_output.append(
                    {
                        "layer": layer,
                        "readout": readout,
                        "held_out_source": source,
                        "held_out_source_label": source_display_name(source, label),
                        "held_out_label": int(label),
                        "label_name": "harmful" if int(label) == 1 else "benign",
                        "probe": "standardized_centroid",
                        "eligible": False,
                        "ineligible_reason": reason,
                    }
                )
            continue
        train_scores, calibration_scores, direction, scale = standardized_centroid_probe(
            X[train].astype(np.float32, copy=False),
            labels[train],
            X[calibration].astype(np.float32, copy=False),
        )
        mean = X[train].astype(np.float32, copy=False).mean(axis=0)
        holdout_scaled = (X[holdout].astype(np.float32, copy=False) - mean) / scale
        holdout_scores = holdout_scaled @ direction
        threshold = balanced_threshold(labels[calibration], calibration_scores)
        threshold_source = "validation"
        metrics = score_metrics(labels[holdout], holdout_scores, threshold)
        output.append(
            {
                "layer": layer,
                "readout": readout,
                "held_out_source": source,
                "probe": "standardized_centroid",
                "eligible": True,
                "ineligible_reason": "",
                "threshold_source": threshold_source,
                **metrics,
            }
        )
        for label in sorted(np.unique(labels[holdout]).tolist()):
            label_mask = holdout & (labels == int(label))
            label_metrics = score_metrics(labels[label_mask], holdout_scores[labels[holdout] == int(label)], threshold)
            label_output.append(
                {
                    "layer": layer,
                    "readout": readout,
                    "held_out_source": source,
                    "held_out_source_label": source_display_name(source, label),
                    "held_out_label": int(label),
                    "label_name": "harmful" if int(label) == 1 else "benign",
                    "probe": "standardized_centroid",
                    "eligible": True,
                    "ineligible_reason": "",
                    "threshold_source": threshold_source,
                    "label_recall": (
                        label_metrics["tpr"] if int(label) == 1 else label_metrics["tnr"]
                    ),
                    **label_metrics,
                }
            )
    return output, label_output


def _summarize_loso(frame: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    for column in ("positive_n", "negative_n", "tpr", "tnr"):
        if column not in frame:
            frame[column] = np.nan
    output: list[dict[str, Any]] = []
    for group_index, ((readout, layer), layer_rows) in enumerate(frame.groupby(["readout", "layer"])):
        eligible = layer_rows[layer_rows["eligible"].astype(bool)].copy()
        harmful = eligible[
            (eligible["positive_n"].fillna(0).astype(float) > 0)
            & np.isfinite(eligible["tpr"].astype(float))
        ]
        benign = eligible[
            (eligible["negative_n"].fillna(0).astype(float) > 0)
            & np.isfinite(eligible["tnr"].astype(float))
        ]
        harmful_worst = harmful.loc[harmful["tpr"].astype(float).idxmin()] if not harmful.empty else None
        benign_worst = benign.loc[benign["tnr"].astype(float).idxmin()] if not benign.empty else None
        harmful_macro = float(harmful["tpr"].mean()) if not harmful.empty else float("nan")
        benign_macro = float(benign["tnr"].mean()) if not benign.empty else float("nan")
        row: dict[str, Any] = {
            "readout": readout,
            "layer": int(layer),
            "eligible_source_n": int(eligible["held_out_source"].nunique()),
            "ineligible_source_n": int(
                layer_rows.loc[~layer_rows["eligible"].astype(bool), "held_out_source"].nunique()
            ),
            "ineligible_reasons": canonical_json(
                layer_rows.loc[~layer_rows["eligible"].astype(bool), "ineligible_reason"]
                .fillna("unknown")
                .value_counts()
                .astype(int)
                .to_dict()
            ),
            "harmful_source_n": int(harmful["held_out_source"].nunique()),
            "benign_source_n": int(benign["held_out_source"].nunique()),
            "harmful_macro_tpr": harmful_macro,
            "harmful_worst_tpr": float(harmful_worst["tpr"]) if harmful_worst is not None else float("nan"),
            "harmful_worst_source": str(harmful_worst["held_out_source"]) if harmful_worst is not None else "",
            "benign_macro_tnr": benign_macro,
            "benign_worst_tnr": float(benign_worst["tnr"]) if benign_worst is not None else float("nan"),
            "benign_worst_source": str(benign_worst["held_out_source"]) if benign_worst is not None else "",
            "macro_balanced_source": (
                0.5 * (harmful_macro + benign_macro)
                if math.isfinite(harmful_macro) and math.isfinite(benign_macro)
                else float("nan")
            ),
            "worst_source_label_cell": min(
                value
                for value in (
                    float(harmful_worst["tpr"]) if harmful_worst is not None else float("nan"),
                    float(benign_worst["tnr"]) if benign_worst is not None else float("nan"),
                )
                if math.isfinite(value)
            ) if harmful_worst is not None or benign_worst is not None else float("nan"),
            "source_bootstrap_requested_b": int(iterations),
            "source_bootstrap_cluster_n": int(eligible["held_out_source"].nunique()),
        }
        bootstrap_values: dict[str, list[float]] = {
            "harmful_macro_tpr": [],
            "benign_macro_tnr": [],
            "macro_balanced_source": [],
            "worst_source_label_cell": [],
        }
        sources = eligible["held_out_source"].astype(str).unique()
        if iterations > 0 and len(sources) >= 2:
            rng = np.random.default_rng(seed + group_index * 104729)
            by_source = {source: eligible[eligible["held_out_source"].astype(str) == source] for source in sources}
            for _ in range(iterations):
                sampled = rng.choice(sources, size=len(sources), replace=True)
                sampled_rows = pd.concat([by_source[source] for source in sampled], ignore_index=True)
                h_values = sampled_rows.loc[
                    sampled_rows["positive_n"].fillna(0).astype(float) > 0, "tpr"
                ].astype(float)
                b_values = sampled_rows.loc[
                    sampled_rows["negative_n"].fillna(0).astype(float) > 0, "tnr"
                ].astype(float)
                h_values = h_values[np.isfinite(h_values)]
                b_values = b_values[np.isfinite(b_values)]
                if len(h_values):
                    bootstrap_values["harmful_macro_tpr"].append(float(h_values.mean()))
                if len(b_values):
                    bootstrap_values["benign_macro_tnr"].append(float(b_values.mean()))
                if len(h_values) and len(b_values):
                    bootstrap_values["macro_balanced_source"].append(
                        0.5 * (float(h_values.mean()) + float(b_values.mean()))
                    )
                    bootstrap_values["worst_source_label_cell"].append(
                        min(float(h_values.min()), float(b_values.min()))
                    )
        for metric, values in bootstrap_values.items():
            row[f"{metric}_bootstrap_valid_b"] = len(values)
            if values:
                row[f"{metric}_conditional_source_cluster_ci_low"] = float(np.quantile(values, 0.025))
                row[f"{metric}_conditional_source_cluster_ci_high"] = float(np.quantile(values, 0.975))
        output.append(row)
    return pd.DataFrame(output)


def _metadata_baseline(frame: pd.DataFrame, table: ActivationTable, seed: int) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=int)
    standard = ~frame["is_attack"].to_numpy(dtype=bool)
    train = standard & frame["split"].eq("train").to_numpy()
    validation = standard & frame["split"].eq("validation").to_numpy()
    test = standard & frame["split"].eq("test").to_numpy()
    features = np.column_stack(
        [
            frame["prompt_text"].astype(str).str.len().to_numpy(dtype=float),
            frame["prompt_text"].astype(str).str.split().str.len().to_numpy(dtype=float),
            (table.image_token_counts > 0).astype(float),
            table.sequence_lengths.astype(float),
            table.image_token_counts.astype(float),
            table.image_widths.astype(float),
            table.image_heights.astype(float),
        ]
    )
    model = _safe_fit(features[train], labels[train], seed)
    if (
        model is None
        or len(np.unique(labels[validation])) != 2
        or len(np.unique(labels[test])) != 2
    ):
        return {}
    validation_scores = probe_scores(model, features[validation])
    threshold = balanced_threshold(labels[validation], validation_scores)
    test_scores = probe_scores(model, features[test])
    image_presence = (table.image_token_counts[test] > 0).astype(float)
    return {
        "feature_names": [
            "prompt_chars",
            "prompt_words",
            "has_image_tokens",
            "sequence_length",
            "image_token_count",
            "image_width",
            "image_height",
        ],
        "combined_metadata_probe": json_clean(score_metrics(labels[test], test_scores, threshold)),
        "image_presence_only_auroc": roc_auc(labels[test], image_presence),
    }


def _write_report(
    path: Path,
    table: ActivationTable,
    frame: pd.DataFrame,
    selected: dict[str, Any],
    probe: pd.DataFrame,
    attack: pd.DataFrame,
    metadata_baseline: dict[str, Any],
    modality_panel: str,
) -> None:
    modality_counts = frame.groupby("modality").size().astype(int).to_dict()
    standard = frame.loc[~frame["is_attack"].astype(bool)]
    standard_source_label = _cross_counts(standard, "source", "label")
    standard_modality_label = _cross_counts(standard, "modality", "label")
    lines = [
        "# Layer-wise OOD Intent Study",
        "",
        f"- Model: `{table.metadata.get('model_name') or table.metadata.get('model')}`",
        f"- Extracted rows: `{len(frame)}`",
        f"- Modality panel: `{modality_panel}`",
        f"- Rows by modality: `{canonical_json(modality_counts)}`",
        f"- Layers: `{int(table.layers.min())}..{int(table.layers.max())}`",
        f"- Readouts: `{', '.join(table.readouts.tolist())}`",
        "- Layer selection: non-attack validation AUROC only; external attacks are frozen holdouts.",
        "",
        "## Standard Panel Composition",
        "",
        "| Source | Benign (0) | Harmful (1) | Total |",
        "|---|---:|---:|---:|",
    ]
    for source, label_counts in standard_source_label.items():
        benign = int(label_counts.get("0", 0))
        harmful = int(label_counts.get("1", 0))
        lines.append(f"| {source} | {benign} | {harmful} | {benign + harmful} |")
    lines.extend(
        [
            "",
            "| Modality | Benign (0) | Harmful (1) | Total |",
            "|---|---:|---:|---:|",
        ]
    )
    for modality, label_counts in standard_modality_label.items():
        benign = int(label_counts.get("0", 0))
        harmful = int(label_counts.get("1", 0))
        lines.append(f"| {modality} | {benign} | {harmful} | {benign + harmful} |")
    lines.extend(
        [
            "",
            "## Selected Layers",
            "",
            "| Readout | Layer | Validation AUROC | Standard test AUROC |",
            "|---|---:|---:|---:|",
        ]
    )
    for readout, row in selected.items():
        lines.append(
            f"| {readout} | {row['layer']} | {row.get('validation_auroc', float('nan')):.4f} | "
            f"{row.get('test_auroc', float('nan')):.4f} |"
        )
    primary = "last" if "last" in selected else next(iter(selected), None)
    if primary and not attack.empty:
        layer = int(selected[primary]["layer"])
        subset = attack[(attack["readout"] == primary) & (attack["layer"] == layer)]
        lines.extend(
            [
                "",
                f"## Frozen Attack Holdout at {primary} Layer {layer}",
                "",
                "| Attack | TPR | AUROC vs benign panel | Group-CV domain AUROC | Standardized score shift |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in subset.iterrows():
            lines.append(
                f"| {row['attack']} | {row['tpr']:.4f} | {row['auroc']:.4f} | "
                f"{row['group_cv_harmful_domain_auroc']:.4f} | {row['standardized_score_shift']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Constraints",
            "",
            "- High pooled AUROC does not establish an intent-invariant boundary; inspect leave-one-source-out and source-domain results.",
            "- `DAN-Prompts` is a jailbreak-template label, not a guaranteed harmful-intent label.",
            "- `OpenAssistant` rows in this asset are assistant replies, so role/style leakage is a known confound.",
            "- `MM-Vet`, `VizWiz-VQA`, Alpaca, and OpenAssistant benign labels are dataset assumptions; HADES/MM-SafetyBench harmful labels are dataset-derived.",
            "- The local `benchmark/VQAv2` payload is identified by its record IDs as VizWiz-VQA, not official COCO VQA v2.",
            "- JailBreakV-28K FigStep, LLM-transfer, and query-related carriers are separate frozen external holdouts; none participates in fitting, layer selection, or threshold calibration.",
            "- External attack curves across all layers are exploratory. Only the validation-selected layer is confirmatory.",
            "",
            "## Metadata-only Baseline",
            "",
            f"```json\n{json.dumps(metadata_baseline, ensure_ascii=False, indent=2)}\n```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap < 0:
        raise ValueError("--bootstrap cannot be negative")
    activation_dir = args.activations.expanduser().resolve()
    run_path = activation_dir / "run.json"
    run_summary = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    if not run_summary and not args.allow_incomplete:
        raise FileNotFoundError(
            "Extraction run.json is required for a formal analysis; pass --allow-incomplete only for diagnostics."
        )
    if run_summary and not args.allow_incomplete:
        if run_summary.get("status") != "complete":
            raise RuntimeError(
                "Extraction status is not complete. Finish/resume extraction or pass --allow-incomplete for diagnostics."
            )
        if int(run_summary.get("completed_rows", -1)) != int(run_summary.get("selected_rows", -2)):
            raise RuntimeError(
                "Extraction is incomplete. Finish/resume extraction or pass --allow-incomplete for a diagnostic-only analysis."
            )
        if int(run_summary.get("failed_rows", 0)) > 0:
            raise RuntimeError(
                "Extraction reports failed rows. Retry/fix them or pass --allow-incomplete for a diagnostic-only analysis."
            )
    table = load_activation_table(activation_dir)
    available_readouts = set(table.readouts.astype(str).tolist())
    for name, value in (
        ("--primary-readout", args.primary_readout),
        ("--domain-probe-readout", args.domain_probe_readout),
    ):
        if value not in available_readouts:
            raise ValueError(f"{name}={value!r} is absent from activation readouts {sorted(available_readouts)}")
    frame = _manifest_frame(args.manifest.expanduser().resolve(), table)
    if bool((frame["is_attack"] & ~frame["split"].eq("external")).any()):
        raise ValueError("Attack rows appeared outside the external split")
    table, frame = _filter_analysis_rows(
        table,
        frame,
        args.modality_panel,
        bool(args.strong_label_sensitivity),
    )
    standard_split_label_counts = _validate_standard_split_labels(
        frame,
        args.modality_panel,
        bool(args.strong_label_sensitivity),
    )

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    source_label_rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []
    loso_rows: list[dict[str, Any]] = []
    loso_label_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    maximum_layer = int(table.layers.max())

    for readout_index, readout in enumerate(table.readouts.tolist()):
        valid = table.readout_valid[:, readout_index]
        for layer_index, layer_value in enumerate(table.layers.tolist()):
            X = table.activations[:, readout_index, layer_index, :]
            probe_row, per_source, per_source_label, per_attack, _ = _analyze_layer(
                X,
                frame,
                valid,
                int(layer_value),
                str(readout),
                args,
            )
            if probe_row is None:
                continue
            probe_row["normalized_depth"] = int(layer_value) / maximum_layer
            probe_rows.append(probe_row)
            source_rows.extend(per_source)
            source_label_rows.extend(per_source_label)
            attack_rows.extend(per_attack)
            if not args.skip_loso and readout == args.primary_readout:
                per_loso, per_loso_label = _loso_rows(
                    X,
                    frame,
                    valid,
                    int(layer_value),
                    str(readout),
                    args.seed + int(layer_value),
                )
                loso_rows.extend(per_loso)
                loso_label_rows.extend(per_loso_label)
            if readout == args.domain_probe_readout:
                standard = valid & ~frame["is_attack"].to_numpy(dtype=bool)
                domain_row: dict[str, Any] = {
                    "layer": int(layer_value),
                    "readout": str(readout),
                }
                for label, name in ((0, "benign"), (1, "harmful")):
                    subset = standard & frame["label"].eq(label).to_numpy()
                    domain_row[f"{name}_source_macro_f1"] = multiclass_centroid_macro_f1(
                        X[subset].astype(np.float32, copy=False),
                        frame.loc[subset, "source"].astype(str).to_numpy(),
                        frame.loc[subset, "split_group_id"].astype(str).to_numpy(),
                        args.seed + int(layer_value) + label,
                    )
                domain_rows.append(domain_row)

    probe_frame = pd.DataFrame(probe_rows)
    if probe_frame.empty:
        raise RuntimeError("No layer had enough valid train/validation/test samples for analysis")
    selected: dict[str, dict[str, Any]] = {}
    for readout, subset in probe_frame.groupby("readout"):
        candidates = subset[np.isfinite(subset["validation_auroc"].astype(float))]
        if candidates.empty:
            continue
        row = candidates.sort_values(["validation_auroc", "layer"], ascending=[False, True]).iloc[0]
        selected[str(readout)] = json_clean(row.to_dict())
    probe_frame["validation_selected"] = False
    for readout, row in selected.items():
        probe_frame.loc[
            (probe_frame["readout"] == readout) & (probe_frame["layer"] == int(row["layer"])),
            "validation_selected",
        ] = True

    common_readouts = [
        value
        for value in ("last", "non_image_mean", "image_mean")
        if value in table.readouts.tolist()
    ]
    common_probe_rows: list[dict[str, Any]] = []
    common_attack_rows: list[dict[str, Any]] = []
    common_selected: dict[str, dict[str, Any]] = {}
    common_panel_sha = ""
    common_panel_n = 0
    if "image_mean" in common_readouts and len(common_readouts) >= 2:
        readout_indices = [table.readouts.tolist().index(value) for value in common_readouts]
        common_valid = np.logical_and.reduce(table.readout_valid[:, readout_indices], axis=1)
        common_valid &= table.image_token_counts > 0
        common_panel_ids = table.sample_ids[common_valid].tolist()
        common_panel_sha = sha256_text("\n".join(sorted(common_panel_ids)))
        common_panel_n = len(common_panel_ids)
        labels = frame["label"].to_numpy(dtype=int)
        common_masks = _standard_masks(frame, common_valid)
        for readout in common_readouts:
            readout_index = table.readouts.tolist().index(readout)
            for layer_index, layer_value in enumerate(table.layers.tolist()):
                X = table.activations[:, readout_index, layer_index, :]
                probe_row, _, _, per_attack, _ = _analyze_layer(
                    X,
                    frame,
                    common_valid,
                    int(layer_value),
                    str(readout),
                    args,
                    include_source_diagnostics=False,
                )
                if probe_row is None:
                    continue
                probe_row.update(
                    {
                        "panel": "common_multimodal",
                        "panel_sample_ids_sha256": common_panel_sha,
                        "normalized_depth": int(layer_value) / maximum_layer,
                    }
                )
                for split, split_mask in common_masks.items():
                    probe_row[f"{split}_positive_n"] = int(np.sum(split_mask & (labels == 1)))
                    probe_row[f"{split}_negative_n"] = int(np.sum(split_mask & (labels == 0)))
                common_probe_rows.append(probe_row)
                for attack_row in per_attack:
                    attack_row.update(
                        {
                            "panel": "common_multimodal",
                            "panel_sample_ids_sha256": common_panel_sha,
                        }
                    )
                    common_attack_rows.append(attack_row)
        common_probe_frame = pd.DataFrame(common_probe_rows)
        if not common_probe_frame.empty:
            for readout, subset in common_probe_frame.groupby("readout"):
                candidates = subset[np.isfinite(subset["validation_auroc"].astype(float))]
                if not candidates.empty:
                    best = candidates.sort_values(
                        ["validation_auroc", "layer"], ascending=[False, True]
                    ).iloc[0]
                    common_selected[str(readout)] = json_clean(best.to_dict())
            common_probe_frame["validation_selected"] = False
            for readout, row in common_selected.items():
                common_probe_frame.loc[
                    (common_probe_frame["readout"] == readout)
                    & (common_probe_frame["layer"] == int(row["layer"])),
                    "validation_selected",
                ] = True
    else:
        common_probe_frame = pd.DataFrame()
    common_attack_frame = pd.DataFrame(common_attack_rows)

    source_frame = pd.DataFrame(source_rows)
    source_label_frame = pd.DataFrame(source_label_rows)
    attack_frame = pd.DataFrame(attack_rows)
    loso_frame = pd.DataFrame(loso_rows)
    loso_label_frame = pd.DataFrame(loso_label_rows)
    loso_summary = _summarize_loso(loso_frame, args.bootstrap, args.seed)
    if not loso_summary.empty:
        loso_summary["validation_selected"] = False
        for readout, row in selected.items():
            loso_summary.loc[
                (loso_summary["readout"] == readout)
                & (loso_summary["layer"] == int(row["layer"])),
                "validation_selected",
            ] = True
    domain_frame = pd.DataFrame(domain_rows)
    probe_frame.to_csv(out_dir / "layer_probe_metrics.csv", index=False)
    source_frame.to_csv(out_dir / "source_metrics.csv", index=False)
    source_label_frame.to_csv(out_dir / "source_label_metrics.csv", index=False)
    attack_frame.to_csv(out_dir / "attack_shift_metrics.csv", index=False)
    loso_frame.to_csv(out_dir / "leave_one_source_out.csv", index=False)
    loso_label_frame.to_csv(out_dir / "leave_one_source_out_label_metrics.csv", index=False)
    loso_summary.to_csv(out_dir / "leave_one_source_out_summary.csv", index=False)
    domain_frame.to_csv(out_dir / "source_domain_metrics.csv", index=False)
    common_probe_frame.to_csv(out_dir / "common_panel_layer_metrics.csv", index=False)
    common_attack_frame.to_csv(out_dir / "common_panel_attack_metrics.csv", index=False)

    metadata_baseline = _metadata_baseline(frame, table, args.seed)
    standard_frame = frame.loc[~frame["is_attack"].astype(bool)]
    coverage = {
        "selected_manifest_rows": len(frame),
        "modality_panel": args.modality_panel,
        "by_source": frame.groupby("source").size().astype(int).to_dict(),
        "by_label": frame.groupby("label").size().astype(int).to_dict(),
        "by_modality": frame.groupby("modality").size().astype(int).to_dict(),
        "by_split": frame.groupby("split").size().astype(int).to_dict(),
        "standard_by_source_label": _cross_counts(
            standard_frame, "source", "label"
        ),
        "standard_by_modality_label": _cross_counts(
            standard_frame, "modality", "label"
        ),
        "standard_split_label_counts": standard_split_label_counts,
        "valid_by_readout": {
            readout: int(table.readout_valid[:, index].sum())
            for index, readout in enumerate(table.readouts.tolist())
        },
        "analysis_sample_ids_sha256": sha256_text("\n".join(sorted(table.sample_ids.tolist()))),
        "strong_label_sensitivity": bool(args.strong_label_sensitivity),
        "common_multimodal_panel_rows": common_panel_n,
    }
    result = {
        "schema_version": RESULT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": table.metadata.get("model_name") or table.metadata.get("model"),
        "manifest_sha256": table.metadata.get("manifest_sha256"),
        "run_fingerprint": table.metadata.get("run_fingerprint"),
        "modality_panel": args.modality_panel,
        "primary_readout": args.primary_readout,
        "domain_probe_readout": args.domain_probe_readout,
        "bootstrap_iterations": args.bootstrap,
        "external_attack_training_rows": 0,
        "selected_layers": selected,
        "common_multimodal_panel": {
            "sample_ids_sha256": common_panel_sha,
            "rows": common_panel_n,
            "readouts": common_readouts,
            "selected_layers": common_selected,
        },
        "coverage": coverage,
        "extraction_run": run_summary,
        "metadata_baseline": metadata_baseline,
        "protocol_notes": [
            "Layers and balanced thresholds are selected using non-attack validation only.",
            "Attack AUROC compares attack positives with the frozen standard benign test panel.",
            "Group-CV harmful-domain AUROC is a linear domain-separability diagnostic; source, modality, and unmatched-goal composition remain possible confounds.",
            "Matched-goal domain AUROC is reported only when exact semantic groups exist in both standard harmful and attack panels.",
            "LOSO uses a standardized centroid probe to keep the all-layer source audit computationally tractable.",
            "Bootstrap intervals condition on the fitted probe, selected layer, and frozen threshold; they resample evaluation clusters only.",
            "Readout comparisons intended as mechanistic evidence must use the common multimodal panel outputs.",
            "The modality panel is applied jointly with strong-label filtering before any probe fitting, threshold calibration, or layer selection.",
            "Source-label diagnostic tables split XSTest-safe and XSTest-unsafe without changing source-level LOSO holdouts or source-macro weighting.",
            "All-layer source-label tables report point estimates; clustered bootstrap intervals remain source-level to avoid millions of redundant resampling iterations.",
        ],
    }
    write_json_atomic(out_dir / "analysis.json", result)
    _write_report(
        out_dir / "report.md",
        table,
        frame,
        selected,
        probe_frame,
        attack_frame,
        metadata_baseline,
        args.modality_panel,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
