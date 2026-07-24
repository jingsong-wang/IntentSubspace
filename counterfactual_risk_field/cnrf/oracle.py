from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from .activations import ActivationData, align_rows
from .calibration import (
    robust_location_scale,
    robust_standardize,
    support_radius,
    worst_group_threshold,
)
from .experiment import fuse_view_scores, json_safe
from .metrics import binary_metrics, grouped_metrics
from .risk_field import EPS, l2_normalize
from .schema import (
    PairRecord,
    build_pair_records,
    evaluation_group,
    modality,
    pack_id,
    protection_group,
    protocol_split,
    row_id,
    row_label,
    row_metadata,
    validate_no_pack_leakage,
    validate_unique_ids,
)


ORACLE_WARNING = (
    "ORACLE_ONLY: test/external labels participate in subset or threshold selection. "
    "These numbers are diagnostic ceilings and are not generalization estimates."
)


@dataclass(frozen=True)
class FixedView:
    readout: str
    layer: int


@dataclass
class ViewCandidateScores:
    scores: np.ndarray
    raw_scores: np.ndarray
    nearest_distances: np.ndarray
    center: float
    scale: float
    pair_count: int
    pack_count: int


@dataclass
class PackViewData:
    cache: "ViewPairCache"
    axes: tuple[str, ...]
    pack_ids: np.ndarray
    distances: np.ndarray
    coordinates: np.ndarray
    pair_choices: np.ndarray
    scale_benign_local: np.ndarray
    scale_pack_ids: np.ndarray
    pair_count: int

    def _selected_positions(self, selected_packs: set[str] | None) -> np.ndarray:
        if selected_packs is None:
            return np.arange(len(self.pack_ids), dtype=np.int64)
        return np.flatnonzero(
            np.asarray([str(value) in selected_packs for value in self.pack_ids], dtype=bool)
        )

    @staticmethod
    def _topk(
        distances: np.ndarray,
        coordinates: np.ndarray,
        *,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if distances.shape != coordinates.shape:
            raise ValueError("Distance and coordinate pack matrices must align")
        if distances.shape[1] < k:
            raise ValueError(
                f"Risk field needs at least k={k} eligible packs; got {distances.shape[1]}"
            )
        positions = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        chosen_distances = np.take_along_axis(distances, positions, axis=1)
        chosen_coordinates = np.take_along_axis(coordinates, positions, axis=1)
        finite = np.all(np.isfinite(chosen_distances), axis=1) & np.all(
            np.isfinite(chosen_coordinates), axis=1
        )
        scores = np.full(len(distances), np.nan, dtype=np.float64)
        nearest = np.full(len(distances), np.nan, dtype=np.float64)
        if np.any(finite):
            scores[finite] = np.median(chosen_coordinates[finite], axis=1)
            nearest[finite] = np.min(chosen_distances[finite], axis=1)
        return scores, nearest

    def score(
        self,
        *,
        selected_packs: set[str] | None,
        k: int,
    ) -> ViewCandidateScores:
        positions = self._selected_positions(selected_packs)
        if len(positions) <= k:
            raise ValueError(
                f"View {self.cache.branch}/{self.cache.view.readout}/layer{self.cache.view.layer} "
                f"needs more than k={k} packs for leave-pack-out scaling; got {len(positions)}"
            )
        selected_pack_values = self.pack_ids[positions].astype(str)
        distances = self.distances[:, positions]
        coordinates = self.coordinates[:, positions]
        raw_scores, nearest = self._topk(distances, coordinates, k=k)

        scale_mask = np.asarray(
            [str(value) in set(selected_pack_values.tolist()) for value in self.scale_pack_ids],
            dtype=bool,
        )
        scale_rows = self.scale_benign_local[scale_mask]
        scale_packs = self.scale_pack_ids[scale_mask].astype(str)
        if len(scale_rows) < 2:
            raise ValueError("Too few reference benign endpoints for robust scaling")
        scale_distances = distances[scale_rows].copy()
        scale_coordinates = coordinates[scale_rows]
        pack_position = {
            str(value): index for index, value in enumerate(selected_pack_values.tolist())
        }
        for row_index, excluded in enumerate(scale_packs.tolist()):
            position = pack_position.get(str(excluded))
            if position is not None:
                scale_distances[row_index, position] = math.inf
        scale_scores, _ = self._topk(scale_distances, scale_coordinates, k=k)
        center, scale = robust_location_scale(scale_scores)
        standardized = robust_standardize(raw_scores, center, scale)
        selected_pack_set = set(selected_pack_values.tolist())
        return ViewCandidateScores(
            scores=standardized,
            raw_scores=raw_scores,
            nearest_distances=nearest,
            center=center,
            scale=scale,
            pair_count=int(
                np.sum(
                    np.asarray(
                        [str(value) in selected_pack_set for value in self.scale_pack_ids],
                        dtype=bool,
                    )
                )
            ),
            pack_count=int(len(positions)),
        )


@dataclass
class ViewPairCache:
    branch: str
    view: FixedView
    branch_indices: np.ndarray
    local_valid: np.ndarray
    pair_ids: np.ndarray
    pack_ids: np.ndarray
    axes: np.ndarray
    benign_local: np.ndarray
    pair_distances: np.ndarray
    pair_coordinates: np.ndarray
    score_clip: float | None

    def prepare(self, axes: Iterable[str]) -> PackViewData:
        axis_values = tuple(sorted(set(str(value) for value in axes)))
        pair_mask = np.isin(self.axes.astype(str), np.asarray(axis_values).astype(str))
        pair_positions = np.flatnonzero(pair_mask)
        if not len(pair_positions):
            raise ValueError(
                f"No reference pairs remain for {self.branch}/{self.view.readout} "
                f"under axes={axis_values}"
            )
        packs = np.asarray(sorted(set(self.pack_ids[pair_positions].astype(str).tolist())))
        distances = np.full((len(self.branch_indices), len(packs)), math.inf, dtype=np.float64)
        coordinates = np.full((len(self.branch_indices), len(packs)), np.nan, dtype=np.float64)
        pair_choices = np.full((len(self.branch_indices), len(packs)), -1, dtype=np.int32)
        valid_pair_distances = self.pair_distances[:, pair_positions]
        for pack_position, current_pack in enumerate(packs.tolist()):
            within = np.flatnonzero(
                self.pack_ids[pair_positions].astype(str) == str(current_pack)
            )
            current_pair_positions = pair_positions[within]
            current_distances = valid_pair_distances[:, within]
            nearest = np.argmin(current_distances, axis=1)
            chosen_pair_positions = current_pair_positions[nearest]
            row_positions = np.arange(len(self.local_valid), dtype=np.int64)
            distances[self.local_valid, pack_position] = current_distances[
                row_positions, nearest
            ]
            coordinates[self.local_valid, pack_position] = self.pair_coordinates[
                row_positions, chosen_pair_positions
            ]
            pair_choices[self.local_valid, pack_position] = chosen_pair_positions.astype(
                np.int32
            )
        return PackViewData(
            cache=self,
            axes=axis_values,
            pack_ids=packs,
            distances=distances,
            coordinates=coordinates,
            pair_choices=pair_choices,
            scale_benign_local=self.benign_local[pair_positions],
            scale_pack_ids=self.pack_ids[pair_positions],
            pair_count=int(len(pair_positions)),
        )


@dataclass
class PreparedBranch:
    branch: str
    axes: tuple[str, ...]
    views: list[PackViewData]
    pack_universe: tuple[str, ...]


def enumerate_axis_subsets(axes: Sequence[str]) -> Iterator[tuple[str, ...]]:
    values = tuple(sorted(dict.fromkeys(str(value) for value in axes)))
    for size in range(1, len(values) + 1):
        yield from itertools.combinations(values, size)


def _policy_scores(values: np.ndarray, *, policy: str) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(scores)
    if policy == "scored_only":
        return scores
    if policy not in {"abstain_safe", "abstain_risk"}:
        raise ValueError(f"Unknown abstention policy {policy!r}")
    finite_values = scores[finite]
    if len(finite_values):
        span = max(1.0, float(np.ptp(finite_values)) + 1.0)
        replacement = (
            float(np.min(finite_values) - span)
            if policy == "abstain_safe"
            else float(np.max(finite_values) + span)
        )
    else:
        replacement = -1.0 if policy == "abstain_safe" else 1.0
    scores[~finite] = replacement
    return scores


def _target_tprs(
    report: dict[str, Any],
    target_groups: Sequence[str],
) -> list[float]:
    output: list[float] = []
    for group in target_groups:
        value = report["groups"].get(str(group))
        if value is None or int(value["positive_n"]) == 0:
            output.append(0.0)
        else:
            output.append(float(value["tpr"]))
    return output


def oracle_operating_point(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: Sequence[str],
    *,
    target_groups: Sequence[str],
    max_fpr: float,
    objective_mode: str,
) -> dict[str, Any]:
    """Select a label-leaking diagnostic threshold under a worst-group FPR cap."""

    return oracle_operating_points(
        labels,
        scores,
        groups,
        targets={"target": tuple(target_groups)},
        max_fprs=[max_fpr],
        objective_modes={"target": objective_mode},
    )[f"{float(max_fpr):.6g}"]["target"]


def oracle_operating_points(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: Sequence[str],
    *,
    targets: dict[str, tuple[str, ...]],
    max_fprs: Sequence[float],
    objective_modes: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Compute many oracle targets from one vectorized threshold sweep."""

    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=np.float64)
    group_values = np.asarray(groups).astype(str)
    if not (y.shape == values.shape == group_values.shape):
        raise ValueError("Oracle labels, scores, and groups must align")
    if not np.isfinite(values).all():
        raise ValueError("Oracle operating-point scores must be finite")
    for target_name, target_values in targets.items():
        if not target_values:
            raise ValueError(f"Oracle target {target_name!r} contains no groups")
        if objective_modes.get(target_name) not in {"macro", "worst"}:
            raise ValueError(
                f"Oracle target {target_name!r} needs objective mode 'macro' or 'worst'"
            )

    unique = np.unique(values)
    thresholds = np.concatenate(
        [
            np.asarray([np.nextafter(float(np.max(unique)), math.inf)]),
            unique[::-1],
        ]
    )
    group_names = tuple(sorted(set(group_values.tolist())))
    tpr_curves: dict[str, np.ndarray] = {}
    fpr_curves: dict[str, np.ndarray] = {}
    for group in group_names:
        group_mask = group_values == group
        positive = np.sort(values[group_mask & (y == 1)])
        negative = np.sort(values[group_mask & (y == 0)])
        tpr_curves[group] = (
            (len(positive) - np.searchsorted(positive, thresholds, side="left"))
            / len(positive)
            if len(positive)
            else np.full(len(thresholds), np.nan)
        )
        fpr_curves[group] = (
            (len(negative) - np.searchsorted(negative, thresholds, side="left"))
            / len(negative)
            if len(negative)
            else np.full(len(thresholds), np.nan)
        )
    benign_curves = [curve for curve in fpr_curves.values() if np.isfinite(curve).any()]
    worst_fpr_curve = (
        np.nanmax(np.stack(benign_curves, axis=0), axis=0)
        if benign_curves
        else np.zeros(len(thresholds), dtype=np.float64)
    )
    report_cache: dict[float, dict[str, Any]] = {}
    output: dict[str, dict[str, Any]] = {}
    for max_fpr in max_fprs:
        feasible = np.flatnonzero(worst_fpr_curve <= float(max_fpr) + 1e-12)
        if not len(feasible):
            raise ValueError("No threshold satisfies the requested worst-group FPR cap")
        target_output: dict[str, Any] = {}
        for target_name, raw_target_values in targets.items():
            target_values = tuple(str(value) for value in raw_target_values)
            target_matrix = np.stack(
                [
                    (
                        tpr_curves[group]
                        if group in tpr_curves and np.isfinite(tpr_curves[group]).any()
                        else np.zeros(len(thresholds), dtype=np.float64)
                    )
                    for group in target_values
                ],
                axis=0,
            )
            macro_curve = np.mean(target_matrix, axis=0)
            worst_curve = np.min(target_matrix, axis=0)
            mode = objective_modes[target_name]
            objective_curve = macro_curve if mode == "macro" else worst_curve
            best_index = max(
                feasible.tolist(),
                key=lambda index: (
                    float(objective_curve[index]),
                    float(worst_curve[index]),
                    float(macro_curve[index]),
                    -float(worst_fpr_curve[index]),
                    -float(thresholds[index]),
                ),
            )
            threshold = float(thresholds[best_index])
            if threshold not in report_cache:
                report_cache[threshold] = grouped_metrics(
                    y, values, threshold, group_values
                )
            report = report_cache[threshold]
            benign_upper_bounds = [
                float(value["fpr_ci95"][1])
                for value in report["groups"].values()
                if int(value["negative_n"]) > 0
                and value.get("fpr_ci95") is not None
            ]
            target_output[target_name] = json_safe(
                {
                    "oracle_only": True,
                    "warning": ORACLE_WARNING,
                    "max_empirical_fpr": float(max_fpr),
                    "objective_mode": mode,
                    "target_groups": list(target_values),
                    "threshold": threshold,
                    "objective_tpr": float(objective_curve[best_index]),
                    "macro_target_tpr": float(macro_curve[best_index]),
                    "worst_target_tpr": float(worst_curve[best_index]),
                    "worst_empirical_fpr": float(worst_fpr_curve[best_index]),
                    "worst_fpr_ci95_upper": (
                        max(benign_upper_bounds) if benign_upper_bounds else None
                    ),
                    "by_group": report,
                }
            )
        output[f"{float(max_fpr):.6g}"] = target_output
    return output


def oracle_targets(labels: np.ndarray, groups: np.ndarray) -> dict[str, tuple[str, ...]]:
    y = np.asarray(labels, dtype=int)
    values = np.asarray(groups).astype(str)
    harmful = tuple(sorted(set(values[y == 1].tolist())))
    targets: dict[str, tuple[str, ...]] = {
        "macro_harmful": harmful,
        "worst_harmful": harmful,
    }
    targets.update({f"group:{group}": (group,) for group in harmful})
    return targets


class OracleAnalyzer:
    """Fixed-view, label-leaking diagnostic analysis over a frozen CNRF bank."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        activations: ActivationData,
        fixed_views: dict[str, list[FixedView]],
        *,
        k: int,
        alpha: float,
        support_quantile_value: float,
        min_arrow_norm: float,
        score_clip: float | None,
        fusion_policy: str,
    ) -> None:
        validate_unique_ids(rows)
        pairs = build_pair_records(rows)
        validate_no_pack_leakage(pairs)
        order = align_rows(rows, activations)
        self.rows = rows
        self.activations = activations.activations[order]
        self.readout_valid = activations.readout_valid[order]
        self.layers = activations.layers
        self.readouts = activations.readouts.astype(str)
        self.metadata = activations.metadata
        self.pairs = pairs
        self.k = int(k)
        self.alpha = float(alpha)
        self.support_quantile_value = float(support_quantile_value)
        self.min_arrow_norm = float(min_arrow_norm)
        self.score_clip = score_clip
        self.fusion_policy = str(fusion_policy)
        self.labels = np.asarray([row_label(row) for row in rows], dtype=int)
        self.splits = np.asarray([protocol_split(row) for row in rows]).astype(str)
        self.modalities = np.asarray([modality(row) for row in rows]).astype(str)
        self.groups = np.asarray([evaluation_group(row) for row in rows]).astype(str)
        self.protection_groups = np.asarray(
            [protection_group(row) for row in rows]
        ).astype(str)
        self.pack_ids = np.asarray([pack_id(row) for row in rows]).astype(str)
        self.sample_ids = np.asarray([row_id(row) for row in rows]).astype(str)
        self.fixed_views = fixed_views
        self.view_caches: dict[str, list[ViewPairCache]] = {
            branch: [self._build_view_cache(branch, view) for view in views]
            for branch, views in fixed_views.items()
        }

    def _readout_position(self, readout: str) -> int:
        matches = np.flatnonzero(self.readouts == str(readout))
        if len(matches) != 1:
            raise ValueError(f"Readout {readout!r} is not present exactly once")
        return int(matches[0])

    def _layer_position(self, layer: int) -> int:
        matches = np.flatnonzero(self.layers == int(layer))
        if len(matches) != 1:
            raise ValueError(f"Layer {layer} is not present exactly once")
        return int(matches[0])

    def _pair_axis(self, pair: PairRecord) -> str:
        metadata = row_metadata(self.rows[pair.harmful_index])
        return str(metadata.get("counterfactual_axis") or "legacy_or_unknown")

    def _build_view_cache(self, branch: str, view: FixedView) -> ViewPairCache:
        readout_position = self._readout_position(view.readout)
        layer_position = self._layer_position(view.layer)
        branch_indices = np.flatnonzero(self.modalities == branch)
        local_by_global = {value: index for index, value in enumerate(branch_indices.tolist())}
        local_valid_mask = self.readout_valid[branch_indices, readout_position]
        local_valid = np.flatnonzero(local_valid_mask)
        query = l2_normalize(
            self.activations[
                branch_indices[local_valid], readout_position, layer_position, :
            ]
        )
        reference_pairs = [
            pair
            for pair in self.pairs
            if pair.split == "reference"
            and pair.modality == branch
            and self.readout_valid[pair.benign_index, readout_position]
            and self.readout_valid[pair.harmful_index, readout_position]
        ]
        benign = l2_normalize(
            self.activations[
                [pair.benign_index for pair in reference_pairs],
                readout_position,
                layer_position,
                :,
            ]
        )
        harmful = l2_normalize(
            self.activations[
                [pair.harmful_index for pair in reference_pairs],
                readout_position,
                layer_position,
                :,
            ]
        )
        arrows = harmful - benign
        arrow_norm_squared = np.sum(arrows * arrows, axis=1)
        midpoints = 0.5 * (benign + harmful)
        midpoint_norms = np.linalg.norm(midpoints, axis=1)
        keep = (
            (np.sqrt(arrow_norm_squared) >= self.min_arrow_norm)
            & (midpoint_norms > EPS)
        )
        if int(np.sum(keep)) <= self.k:
            raise ValueError(
                f"Fixed view {branch}/{view.readout}/layer{view.layer} has too few arrows"
            )
        reference_pairs = [pair for pair, retained in zip(reference_pairs, keep) if retained]
        arrows = arrows[keep]
        arrow_norm_squared = arrow_norm_squared[keep]
        midpoints = midpoints[keep]
        midpoint_unit = l2_normalize(midpoints)
        distances = 1.0 - query @ midpoint_unit.T
        coordinates = 2.0 * (query @ arrows.T)
        offsets = 2.0 * np.sum(midpoints * arrows, axis=1)
        coordinates = (coordinates - offsets[None, :]) / np.maximum(
            arrow_norm_squared[None, :], EPS
        )
        if self.score_clip is not None:
            coordinates = np.clip(coordinates, -self.score_clip, self.score_clip)
        return ViewPairCache(
            branch=branch,
            view=view,
            branch_indices=branch_indices,
            local_valid=local_valid,
            pair_ids=np.asarray([pair.pair_id for pair in reference_pairs]).astype(str),
            pack_ids=np.asarray([pair.pack_id for pair in reference_pairs]).astype(str),
            axes=np.asarray([self._pair_axis(pair) for pair in reference_pairs]).astype(str),
            benign_local=np.asarray(
                [local_by_global[pair.benign_index] for pair in reference_pairs],
                dtype=np.int64,
            ),
            pair_distances=np.asarray(distances, dtype=np.float64),
            pair_coordinates=np.asarray(coordinates, dtype=np.float64),
            score_clip=self.score_clip,
        )

    def branches(self) -> tuple[str, ...]:
        return tuple(sorted(self.view_caches))

    def axes(self, branch: str | None = None) -> tuple[str, ...]:
        caches = (
            self.view_caches[str(branch)]
            if branch is not None
            else [cache for values in self.view_caches.values() for cache in values]
        )
        return tuple(
            sorted(
                set(
                    axis
                    for cache in caches
                    for axis in cache.axes.astype(str).tolist()
                )
            )
        )

    def prepare_branch(self, branch: str, axes: Iterable[str]) -> PreparedBranch:
        prepared = [cache.prepare(axes) for cache in self.view_caches[branch]]
        universe = tuple(
            sorted(set(pack for view in prepared for pack in view.pack_ids.astype(str)))
        )
        if len(universe) <= self.k:
            raise ValueError(
                f"Branch {branch!r} has too few packs under axes={tuple(axes)}"
            )
        return PreparedBranch(
            branch=branch,
            axes=tuple(sorted(set(str(value) for value in axes))),
            views=prepared,
            pack_universe=universe,
        )

    def _branch_local_arrays(self, branch: str) -> tuple[np.ndarray, ...]:
        branch_indices = self.view_caches[branch][0].branch_indices
        return (
            branch_indices,
            self.labels[branch_indices],
            self.splits[branch_indices],
            self.groups[branch_indices],
            self.protection_groups[branch_indices],
        )

    def evaluate_prepared(
        self,
        prepared: PreparedBranch,
        *,
        selected_packs: Iterable[str] | None = None,
        max_fprs: Sequence[float] = (0.01, 0.05),
        oracle_policies: Sequence[str] = ("abstain_safe", "abstain_risk"),
    ) -> dict[str, Any]:
        selected = (
            set(str(value) for value in selected_packs)
            if selected_packs is not None
            else None
        )
        branch_indices, labels, splits, groups, protection = self._branch_local_arrays(
            prepared.branch
        )
        view_scores: list[np.ndarray] = []
        view_distances: list[np.ndarray] = []
        view_reports: list[dict[str, Any]] = []
        for view in prepared.views:
            values = view.score(selected_packs=selected, k=self.k)
            calibration_benign = (
                (splits == "calibration")
                & (labels == 0)
                & np.isfinite(values.nearest_distances)
            )
            if not np.any(calibration_benign):
                raise ValueError("No benign calibration distances remain")
            radius = support_radius(
                values.nearest_distances[calibration_benign],
                self.support_quantile_value,
            )
            view_scores.append(values.scores)
            view_distances.append(values.nearest_distances)
            view_reports.append(
                {
                    "readout": view.cache.view.readout,
                    "layer": view.cache.view.layer,
                    "center": values.center,
                    "scale": values.scale,
                    "support_radius": radius,
                    "pairs": values.pair_count,
                    "packs": values.pack_count,
                }
            )
        fused, supported, _, winners, winner_supported = fuse_view_scores(
            view_scores,
            view_distances,
            [float(value["support_radius"]) for value in view_reports],
            policy=self.fusion_policy,
        )
        calibration_benign = (
            (splits == "calibration") & (labels == 0) & np.isfinite(fused)
        )
        if not np.any(calibration_benign):
            raise ValueError("No finite benign calibration scores remain")
        threshold, calibration = worst_group_threshold(
            fused[calibration_benign],
            protection[calibration_benign],
            alpha=self.alpha,
        )

        split_reports: dict[str, Any] = {}
        for split in ("validation", "calibration", "test", "external"):
            mask = splits == split
            if not np.any(mask):
                continue
            policies: dict[str, Any] = {}
            for policy in ("scored_only", "abstain_safe", "abstain_risk"):
                policy_values = _policy_scores(fused[mask], policy=policy)
                finite = np.isfinite(policy_values)
                policies[policy] = {
                    "overall": binary_metrics(
                        labels[mask][finite], policy_values[finite], threshold
                    ),
                    "by_group": grouped_metrics(
                        labels[mask][finite],
                        policy_values[finite],
                        threshold,
                        groups[mask][finite],
                    ),
                }
            split_reports[split] = {
                "rows": int(np.sum(mask)),
                "scored_rows": int(np.sum(mask & np.isfinite(fused))),
                "score_coverage": float(np.mean(np.isfinite(fused[mask]))),
                "support_coverage": float(np.mean(supported[mask])),
                "winning_view_support_rate": float(np.mean(winner_supported[mask])),
                "policies": policies,
            }

        evaluation_mask = np.isin(splits, np.asarray(["test", "external"]))
        targets = oracle_targets(labels[evaluation_mask], groups[evaluation_mask])
        oracle: dict[str, Any] = {}
        for policy in oracle_policies:
            values = _policy_scores(fused[evaluation_mask], policy=policy)
            oracle[policy] = oracle_operating_points(
                labels[evaluation_mask],
                values,
                groups[evaluation_mask],
                targets=targets,
                max_fprs=max_fprs,
                objective_modes={
                    target_name: (
                        "worst" if target_name == "worst_harmful" else "macro"
                    )
                    for target_name in targets
                },
            )
        selected_values = sorted(selected) if selected is not None else list(prepared.pack_universe)
        return json_safe(
            {
                "format_version": "cnrf_oracle_candidate_v1",
                "oracle_only": True,
                "warning": ORACLE_WARNING,
                "branch": prepared.branch,
                "axes": list(prepared.axes),
                "selected_packs": selected_values,
                "selected_pack_count": len(selected_values),
                "fusion_policy": self.fusion_policy,
                "threshold": threshold,
                "calibration": calibration,
                "views": view_reports,
                "splits": split_reports,
                "oracle": oracle,
            }
        )

    @staticmethod
    def oracle_value(
        result: dict[str, Any],
        *,
        policy: str,
        max_fpr: float,
        target: str,
    ) -> float:
        return float(
            result["oracle"][policy][f"{float(max_fpr):.6g}"][target][
                "objective_tpr"
            ]
        )

    def sample_pack_influence(
        self,
        prepared: PreparedBranch,
        result: dict[str, Any],
        *,
        selected_packs: Iterable[str] | None = None,
        splits: Sequence[str] = ("test", "external"),
    ) -> Iterator[dict[str, Any]]:
        selected = (
            set(str(value) for value in selected_packs)
            if selected_packs is not None
            else set(prepared.pack_universe)
        )
        branch_indices = prepared.views[0].cache.branch_indices
        local_splits = self.splits[branch_indices]
        selected_local_rows = np.flatnonzero(np.isin(local_splits, np.asarray(splits)))
        result_views = {
            (str(value["readout"]), int(value["layer"])): value
            for value in result["views"]
        }
        for view in prepared.views:
            positions = view._selected_positions(selected)
            if len(positions) <= self.k:
                continue
            distances = view.distances[:, positions]
            coordinates = view.coordinates[:, positions]
            choices = view.pair_choices[:, positions]
            packs = view.pack_ids[positions].astype(str)
            width = min(self.k + 1, len(positions))
            nearest = np.argpartition(distances, kth=width - 1, axis=1)[:, :width]
            nearest_distances = np.take_along_axis(distances, nearest, axis=1)
            order = np.argsort(nearest_distances, axis=1, kind="mergesort")
            nearest = np.take_along_axis(nearest, order, axis=1)
            scale = float(result_views[(view.cache.view.readout, view.cache.view.layer)]["scale"])
            for local_row in selected_local_rows.tolist():
                chosen = nearest[local_row]
                if not np.isfinite(distances[local_row, chosen[: self.k]]).all():
                    continue
                full_coordinate = float(
                    np.median(coordinates[local_row, chosen[: self.k]])
                )
                for rank, pack_position in enumerate(chosen[: self.k].tolist(), start=1):
                    alternatives = [
                        value for value in chosen.tolist() if value != pack_position
                    ][: self.k]
                    without = float(np.median(coordinates[local_row, alternatives]))
                    pair_position = int(choices[local_row, pack_position])
                    global_row = int(branch_indices[local_row])
                    yield json_safe(
                        {
                            "sample_id": str(self.sample_ids[global_row]),
                            "branch": prepared.branch,
                            "protocol_split": str(self.splits[global_row]),
                            "evaluation_group": str(self.groups[global_row]),
                            "label": int(self.labels[global_row]),
                            "readout": view.cache.view.readout,
                            "layer": view.cache.view.layer,
                            "neighbor_rank": rank,
                            "pack_id": str(packs[pack_position]),
                            "pair_id": (
                                str(view.cache.pair_ids[pair_position])
                                if pair_position >= 0
                                else None
                            ),
                            "distance": float(distances[local_row, pack_position]),
                            "coordinate": float(coordinates[local_row, pack_position]),
                            "leave_pack_out_delta_raw": full_coordinate - without,
                            "leave_pack_out_delta_standardized": (
                                full_coordinate - without
                            )
                            / max(scale, EPS),
                        }
                    )


def fixed_views_from_summary(summary: dict[str, Any]) -> dict[str, list[FixedView]]:
    output: dict[str, list[FixedView]] = {}
    for branch, report in summary.get("branches", {}).items():
        values = [
            FixedView(readout=str(view["readout"]), layer=int(view["layer"]))
            for view in report.get("selected_views", [])
        ]
        if values:
            output[str(branch)] = values
    if not output:
        raise ValueError("Experiment summary contains no selected views")
    return output
