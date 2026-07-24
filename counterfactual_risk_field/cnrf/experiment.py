from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .activations import ActivationData, align_rows
from .baselines import MeanArrow, TwoSidedKNN
from .calibration import (
    robust_location_scale,
    robust_standardize,
    support_radius,
    worst_group_threshold,
)
from .io import stable_fraction
from .metrics import binary_metrics, grouped_metrics
from .risk_field import CounterfactualRiskField, DegenerateRiskFieldError
from .schema import (
    PairRecord,
    build_pair_records,
    evaluation_group,
    modality,
    pack_id,
    protection_group,
    protocol_split,
    row_label,
    validate_no_pack_leakage,
    validate_unique_ids,
)


@dataclass
class ViewFit:
    readout: str
    layer: int
    field: CounterfactualRiskField
    center: float
    scale: float
    support_radius: float
    validation_threshold: float
    validation_metrics: dict[str, Any]
    validation_groups: dict[str, Any]
    diagnostics: dict[str, Any]


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_finite_or_none(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def json_safe(value: Any) -> Any:
    return _finite_or_none(value)


def _rowwise_nanmax(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    available = np.any(finite, axis=1)
    safe = np.where(finite, values, -math.inf)
    output = np.max(safe, axis=1)
    output[~available] = np.nan
    winners = np.full(len(values), -1, dtype=np.int64)
    winners[available] = np.argmax(safe[available], axis=1)
    return output, winners


def fuse_view_scores(
    view_scores: list[np.ndarray],
    view_distances: list[np.ndarray],
    support_radii: list[float],
    *,
    policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse views and keep support tied to the view that supplies the score."""

    if policy not in {"legacy_max", "supported_max"}:
        raise ValueError(f"Unknown fusion policy {policy!r}")
    score_matrix = np.stack(view_scores, axis=1)
    distance_matrix = np.stack(view_distances, axis=1)
    radii = np.asarray(support_radii, dtype=np.float64)[None, :]
    view_supported = np.isfinite(distance_matrix) & (distance_matrix <= radii)
    eligible = score_matrix if policy == "legacy_max" else np.where(
        view_supported, score_matrix, np.nan
    )
    fused, winners = _rowwise_nanmax(eligible)
    supported = np.any(view_supported, axis=1)
    winner_supported = np.zeros(len(fused), dtype=bool)
    rows = np.flatnonzero(winners >= 0)
    winner_supported[rows] = view_supported[rows, winners[rows]]
    return fused, supported, view_supported, winners, winner_supported


class Experiment:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        activations: ActivationData,
        *,
        k: int = 5,
        alpha: float = 0.05,
        support_quantile: float = 0.99,
        min_arrow_norm: float = 1e-3,
        score_clip: float = 20.0,
        min_field_pairs: int = 2,
        min_field_packs: int = 2,
        min_field_retention_fraction: float = 0.0,
        fusion_policy: str = "legacy_max",
        seed: int = 20260721,
        max_reference_packs: int | None = None,
    ) -> None:
        validate_unique_ids(rows)
        self.pairs = build_pair_records(rows)
        validate_no_pack_leakage(self.pairs)
        order = align_rows(rows, activations)
        self.rows = rows
        self.activations = activations.activations[order]
        self.readout_valid = activations.readout_valid[order]
        self.layers = activations.layers
        self.readouts = activations.readouts
        self.metadata = activations.metadata
        self.k = int(k)
        self.alpha = float(alpha)
        self.support_quantile_value = float(support_quantile)
        self.min_arrow_norm = float(min_arrow_norm)
        self.score_clip = float(score_clip)
        self.min_field_pairs = int(min_field_pairs)
        self.min_field_packs = int(min_field_packs)
        self.min_field_retention_fraction = float(min_field_retention_fraction)
        if fusion_policy not in {"legacy_max", "supported_max"}:
            raise ValueError(f"Unknown fusion policy {fusion_policy!r}")
        self.fusion_policy = fusion_policy
        self.seed = int(seed)
        self.max_reference_packs = max_reference_packs
        self.labels = np.asarray([row_label(row) for row in rows], dtype=int)
        self.splits = np.asarray([protocol_split(row) for row in rows]).astype(str)
        self.modalities = np.asarray([modality(row) for row in rows]).astype(str)
        self.evaluation_groups = np.asarray([evaluation_group(row) for row in rows]).astype(str)
        self.protection_groups = np.asarray([protection_group(row) for row in rows]).astype(str)
        self.pack_ids = np.asarray([pack_id(row) for row in rows]).astype(str)

    def _reference_pairs(self, branch: str) -> list[PairRecord]:
        pairs = [
            pair
            for pair in self.pairs
            if pair.split == "reference" and pair.modality == branch
        ]
        if self.max_reference_packs and self.max_reference_packs > 0:
            packs = sorted(
                set(pair.pack_id for pair in pairs),
                key=lambda value: stable_fraction(value, self.seed),
            )[: self.max_reference_packs]
            selected = set(packs)
            pairs = [pair for pair in pairs if pair.pack_id in selected]
        if len(set(pair.pack_id for pair in pairs)) <= self.k:
            raise ValueError(
                f"Branch {branch!r} needs more than k={self.k} distinct reference packs"
            )
        return pairs

    def _layer_positions(self, candidates: list[int] | None) -> list[int]:
        if not candidates:
            return list(range(len(self.layers)))
        positions: list[int] = []
        for layer in candidates:
            matches = np.flatnonzero(self.layers == int(layer))
            if len(matches) != 1:
                raise ValueError(f"Requested layer {layer} is not present exactly once")
            positions.append(int(matches[0]))
        return positions

    def _view_index(self, name: str) -> int:
        matches = np.flatnonzero(self.readouts.astype(str) == name)
        if len(matches) != 1:
            raise ValueError(f"Readout {name!r} is not present exactly once")
        return int(matches[0])

    def _field_for(
        self,
        pairs: list[PairRecord],
        readout_index: int,
        layer_position: int,
    ) -> CounterfactualRiskField:
        valid_pairs = [
            pair
            for pair in pairs
            if self.readout_valid[pair.benign_index, readout_index]
            and self.readout_valid[pair.harmful_index, readout_index]
        ]
        benign = self.activations[
            [pair.benign_index for pair in valid_pairs], readout_index, layer_position, :
        ]
        harmful = self.activations[
            [pair.harmful_index for pair in valid_pairs], readout_index, layer_position, :
        ]
        return CounterfactualRiskField(
            benign,
            harmful,
            [pair.pair_id for pair in valid_pairs],
            [pair.pack_id for pair in valid_pairs],
            k=self.k,
            min_arrow_norm=self.min_arrow_norm,
            min_valid_pairs=self.min_field_pairs,
            min_unique_packs=max(self.k + 1, self.min_field_packs),
            min_retention_fraction=self.min_field_retention_fraction,
            score_clip=self.score_clip,
        )

    def _reference_scale(
        self,
        field: CounterfactualRiskField,
        pairs: list[PairRecord],
        readout_index: int,
        layer_position: int,
    ) -> tuple[float, float]:
        valid_pairs = [
            pair
            for pair in pairs
            if self.readout_valid[pair.benign_index, readout_index]
            and pair.pack_id in set(field.pack_ids.tolist())
        ]
        vectors = self.activations[
            [pair.benign_index for pair in valid_pairs], readout_index, layer_position, :
        ]
        scores = field.score(
            vectors, exclude_pack_ids=[pair.pack_id for pair in valid_pairs]
        ).scores
        return robust_location_scale(scores)

    def _score_view(
        self,
        field: CounterfactualRiskField,
        readout_index: int,
        layer_position: int,
        center: float,
        scale: float,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
        scores = np.full(len(indices), np.nan, dtype=np.float64)
        distances = np.full(len(indices), np.nan, dtype=np.float64)
        neighbors: list[list[str]] = [[] for _ in indices]
        locally_valid = self.readout_valid[indices, readout_index]
        if np.any(locally_valid):
            selected = indices[locally_valid]
            result = field.score(self.activations[selected, readout_index, layer_position, :])
            scores[locally_valid] = robust_standardize(result.scores, center, scale)
            distances[locally_valid] = result.nearest_midpoint_distance
            iterator = iter(result.neighbor_pack_ids)
            for local_index, is_valid in enumerate(locally_valid.tolist()):
                if is_valid:
                    neighbors[local_index] = next(iterator)
        return scores, distances, neighbors

    def select_view(
        self,
        branch: str,
        readout: str,
        layer_candidates: list[int] | None,
    ) -> tuple[ViewFit | None, list[dict[str, Any]]]:
        pairs = self._reference_pairs(branch)
        readout_index = self._view_index(readout)
        validation_indices = np.flatnonzero(
            (self.modalities == branch) & (self.splits == "validation")
        )
        if not len(validation_indices) or len(np.unique(self.labels[validation_indices])) != 2:
            raise ValueError(f"Branch {branch!r} validation split requires both labels")
        candidates: list[tuple[tuple[float, ...], ViewFit, dict[str, Any]]] = []
        reports: list[dict[str, Any]] = []
        for layer_position in self._layer_positions(layer_candidates):
            layer = int(self.layers[layer_position])
            try:
                field = self._field_for(pairs, readout_index, layer_position)
            except DegenerateRiskFieldError as exc:
                reports.append(
                    {
                        "readout": readout,
                        "layer": layer,
                        "status": "skipped_degenerate_field",
                        "error": str(exc),
                        "diagnostics": exc.diagnostics,
                    }
                )
                continue
            center, scale = self._reference_scale(field, pairs, readout_index, layer_position)
            scores, _, _ = self._score_view(
                field,
                readout_index,
                layer_position,
                center,
                scale,
                validation_indices,
            )
            finite = np.isfinite(scores)
            labels = self.labels[validation_indices][finite]
            score_values = scores[finite]
            groups = self.protection_groups[validation_indices][finite]
            benign = labels == 0
            threshold, calibration = worst_group_threshold(
                score_values[benign], groups[benign], alpha=self.alpha
            )
            metrics = binary_metrics(labels, score_values, threshold)
            group_report = grouped_metrics(
                labels,
                score_values,
                threshold,
                self.evaluation_groups[validation_indices][finite],
            )
            worst_tpr = group_report["worst_tpr"]
            key = (
                float(worst_tpr) if worst_tpr is not None else -math.inf,
                float(metrics["tpr"]),
                -float(metrics["fpr"]),
                float(metrics["auroc"]),
                -float(self.layers[layer_position]),
            )
            diagnostics = {
                **field.diagnostics(),
                "calibration": calibration,
                "valid_validation_rows": int(finite.sum()),
            }
            fit = ViewFit(
                readout=readout,
                layer=int(self.layers[layer_position]),
                field=field,
                center=center,
                scale=scale,
                # Fit after layer selection, using only independent benign
                # calibration rows.  Validation must not define deployment
                # support and select the layer at the same time.
                support_radius=math.nan,
                validation_threshold=threshold,
                validation_metrics=metrics,
                validation_groups=group_report,
                diagnostics=diagnostics,
            )
            report = {
                "readout": readout,
                "layer": layer,
                "status": "eligible",
                "validation_metrics": metrics,
                "validation_groups": group_report,
                "diagnostics": diagnostics,
            }
            candidates.append((key, fit, report))
            reports.append(report)
        if not candidates:
            return None, reports
        selected = max(candidates, key=lambda item: item[0])
        return selected[1], reports

    def _baseline_scores(
        self,
        branch: str,
        selected_views: list[ViewFit],
        factory: Callable[[np.ndarray, np.ndarray, list[str]], Any],
    ) -> np.ndarray:
        branch_indices = np.flatnonzero(self.modalities == branch)
        fused: list[np.ndarray] = []
        reference_pairs = self._reference_pairs(branch)
        for view in selected_views:
            readout_index = self._view_index(view.readout)
            layer_position = int(np.flatnonzero(self.layers == view.layer)[0])
            valid_pairs = [
                pair
                for pair in reference_pairs
                if self.readout_valid[pair.benign_index, readout_index]
                and self.readout_valid[pair.harmful_index, readout_index]
            ]
            benign = self.activations[
                [pair.benign_index for pair in valid_pairs], readout_index, layer_position, :
            ]
            harmful = self.activations[
                [pair.harmful_index for pair in valid_pairs], readout_index, layer_position, :
            ]
            baseline = factory(benign, harmful, [pair.pack_id for pair in valid_pairs])
            ref_safe = baseline.score(
                benign,
                exclude_pack_ids=[pair.pack_id for pair in valid_pairs],
            )
            center, scale = robust_location_scale(ref_safe)
            values = np.full(len(branch_indices), np.nan)
            valid = self.readout_valid[branch_indices, readout_index]
            values[valid] = robust_standardize(
                baseline.score(self.activations[branch_indices[valid], readout_index, layer_position, :]),
                center,
                scale,
            )
            fused.append(values)
        branch_scores, _ = _rowwise_nanmax(np.stack(fused, axis=1))
        output = np.full(len(self.rows), np.nan)
        output[branch_indices] = branch_scores
        return output

    def run_branch(
        self,
        branch: str,
        *,
        readouts: list[str],
        layer_candidates: list[int] | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        branch_indices = np.flatnonzero(self.modalities == branch)
        selected_views: list[ViewFit] = []
        layer_reports: dict[str, Any] = {}
        view_scores: list[np.ndarray] = []
        view_distances: list[np.ndarray] = []
        view_neighbors: list[list[list[str]]] = []
        for readout in readouts:
            readout_index = self._view_index(readout)
            if not np.any(self.readout_valid[branch_indices, readout_index]):
                continue
            fit, candidates = self.select_view(branch, readout, layer_candidates)
            layer_reports[readout] = candidates
            if fit is None:
                continue
            selected_views.append(fit)
            layer_position = int(np.flatnonzero(self.layers == fit.layer)[0])
            scores, distances, neighbors = self._score_view(
                fit.field,
                readout_index,
                layer_position,
                fit.center,
                fit.scale,
                branch_indices,
            )
            view_scores.append(scores)
            view_distances.append(distances)
            view_neighbors.append(neighbors)
        if not selected_views:
            raise ValueError(f"Branch {branch!r} has no valid configured readouts")
        calibration_local = self.splits[branch_indices] == "calibration"
        calibration_benign_base = calibration_local & (self.labels[branch_indices] == 0)
        for fit, distances in zip(selected_views, view_distances):
            radius_rows = calibration_benign_base & np.isfinite(distances)
            if not np.any(radius_rows):
                raise ValueError(
                    f"Branch {branch!r}/{fit.readout} has no benign calibration rows "
                    "with a valid support distance"
                )
            fit.support_radius = support_radius(
                distances[radius_rows], self.support_quantile_value
            )
        fused, supported, view_supported, winning_views, winning_supported = fuse_view_scores(
            view_scores,
            view_distances,
            [fit.support_radius for fit in selected_views],
            policy=self.fusion_policy,
        )

        calibration_benign = calibration_local & (self.labels[branch_indices] == 0) & np.isfinite(fused)
        if not np.any(calibration_benign):
            raise ValueError(f"Branch {branch!r} has no independent benign calibration rows")
        threshold, calibration = worst_group_threshold(
            fused[calibration_benign],
            self.protection_groups[branch_indices][calibration_benign],
            alpha=self.alpha,
        )
        predictions = np.isfinite(fused) & (fused >= threshold)
        triage = np.full(len(branch_indices), "abstain", dtype=object)
        triage[supported & ~predictions] = "safe"
        triage[predictions] = "risk"

        split_reports: dict[str, Any] = {}
        for split in ("validation", "calibration", "test", "external"):
            split_local = self.splits[branch_indices] == split
            if not np.any(split_local):
                continue
            scored = split_local & np.isfinite(fused)
            split_reports[split] = {
                "overall": (
                    binary_metrics(
                        self.labels[branch_indices][scored], fused[scored], threshold
                    )
                    if np.any(scored)
                    else None
                ),
                "by_group": (
                    grouped_metrics(
                        self.labels[branch_indices][scored],
                        fused[scored],
                        threshold,
                        self.evaluation_groups[branch_indices][scored],
                    )
                    if np.any(scored)
                    else None
                ),
                "rows": int(np.sum(split_local)),
                "scored_rows": int(np.sum(scored)),
                "score_coverage": float(np.mean(np.isfinite(fused[split_local]))),
                "support_coverage": float(np.mean(supported[split_local])),
                "winning_view_support_rate": float(
                    np.mean(winning_supported[split_local])
                ),
                "abstention_rate": float(np.mean(triage[split_local] == "abstain")),
                "safe_route_rate": float(np.mean(triage[split_local] == "safe")),
                "risk_route_rate": float(np.mean(triage[split_local] == "risk")),
            }

        full_scores = np.full(len(self.rows), np.nan)
        full_scores[branch_indices] = fused
        baseline_reports: dict[str, Any] = {}
        factories: dict[str, Callable[[np.ndarray, np.ndarray, list[str]], Any]] = {
            "two_sided_knn": lambda benign, harmful, packs: TwoSidedKNN(
                benign, harmful, packs, k=self.k
            ),
            "mean_arrow": lambda benign, harmful, packs: MeanArrow(benign, harmful),
        }
        for name, factory in factories.items():
            baseline = self._baseline_scores(branch, selected_views, factory)
            baseline_local = baseline[branch_indices]
            baseline_calibration = calibration_benign_base & np.isfinite(baseline_local)
            threshold_b, calibration_b = worst_group_threshold(
                baseline_local[baseline_calibration],
                self.protection_groups[branch_indices][baseline_calibration],
                alpha=self.alpha,
            )
            evaluations: dict[str, Any] = {}
            for split in ("test", "external"):
                local = (self.splits[branch_indices] == split) & np.isfinite(baseline_local)
                if np.any(local):
                    evaluations[split] = binary_metrics(
                        self.labels[branch_indices][local], baseline_local[local], threshold_b
                    )
            baseline_reports[name] = {
                "threshold": threshold_b,
                "calibration": calibration_b,
                "evaluations": evaluations,
            }

        score_rows: list[dict[str, Any]] = []
        for local_index, global_index in enumerate(branch_indices.tolist()):
            per_view = {
                fit.readout: {
                    "layer": fit.layer,
                    "score": float(view_scores[index][local_index]) if np.isfinite(view_scores[index][local_index]) else None,
                    "support_distance": float(view_distances[index][local_index]) if np.isfinite(view_distances[index][local_index]) else None,
                    "support_radius": fit.support_radius,
                    "supported": bool(view_supported[local_index, index]),
                    "neighbor_pack_ids": view_neighbors[index][local_index],
                }
                for index, fit in enumerate(selected_views)
            }
            score_rows.append(
                {
                    "sample_id": self.rows[global_index].get("sample_id", self.rows[global_index].get("id")),
                    "label": int(self.labels[global_index]),
                    "protocol_split": str(self.splits[global_index]),
                    "modality": branch,
                    "evaluation_group": str(self.evaluation_groups[global_index]),
                    "score": float(fused[local_index]) if np.isfinite(fused[local_index]) else None,
                    "threshold": threshold,
                    "detected": bool(predictions[local_index]) if np.isfinite(fused[local_index]) else None,
                    "supported": bool(supported[local_index]),
                    "winning_view": (
                        selected_views[int(winning_views[local_index])].readout
                        if winning_views[local_index] >= 0
                        else None
                    ),
                    "winning_view_supported": bool(winning_supported[local_index]),
                    "triage": str(triage[local_index]),
                    "views": per_view,
                }
            )
        report = {
            "branch": branch,
            "rows": int(len(branch_indices)),
            "fusion_policy": self.fusion_policy,
            "selected_views": [
                {
                    "readout": fit.readout,
                    "layer": fit.layer,
                    "robust_center": fit.center,
                    "robust_scale": fit.scale,
                    "support_radius": fit.support_radius,
                    "validation_threshold": fit.validation_threshold,
                    "validation_metrics": fit.validation_metrics,
                    "validation_groups": fit.validation_groups,
                    "field": fit.diagnostics,
                }
                for fit in selected_views
            ],
            "layer_candidates": layer_reports,
            "threshold": threshold,
            "calibration": calibration,
            "splits": split_reports,
            "matched_data_baselines": baseline_reports,
        }
        return json_safe(report), json_safe(score_rows)

    def run(
        self,
        *,
        readouts: list[str],
        layer_candidates: list[int] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        branches = sorted(
            branch
            for branch in set(self.modalities.tolist())
            if any(pair.modality == branch and pair.split == "reference" for pair in self.pairs)
        )
        reports: dict[str, Any] = {}
        scores: list[dict[str, Any]] = []
        for branch in branches:
            report, branch_scores = self.run_branch(
                branch, readouts=readouts, layer_candidates=layer_candidates
            )
            reports[branch] = report
            scores.extend(branch_scores)
        return (
            json_safe(
                {
                    "format_version": "cnrf_experiment_v1",
                    "model": self.metadata.get("model"),
                    "k": self.k,
                    "alpha": self.alpha,
                    "support_quantile": self.support_quantile_value,
                    "min_field_pairs": self.min_field_pairs,
                    "min_field_packs": self.min_field_packs,
                    "min_field_retention_fraction": self.min_field_retention_fraction,
                    "fusion_policy": self.fusion_policy,
                    "max_reference_packs": self.max_reference_packs,
                    "branches": reports,
                }
            ),
            scores,
        )
