from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


EPS = 1e-12


class DegenerateRiskFieldError(ValueError):
    """Raised when a layer/readout has too few non-degenerate arrows."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, EPS)


@dataclass(frozen=True)
class RiskFieldScores:
    scores: np.ndarray
    nearest_midpoint_distance: np.ndarray
    neighbor_pair_ids: list[list[str]]
    neighbor_pack_ids: list[list[str]]
    neighbor_coordinates: list[list[float]]


class CounterfactualRiskField:
    """Non-parametric local field built from benign -> harmful counterfactual arrows."""

    def __init__(
        self,
        benign: np.ndarray,
        harmful: np.ndarray,
        pair_ids: Iterable[str],
        pack_ids: Iterable[str],
        *,
        k: int = 5,
        min_arrow_norm: float = 1e-3,
        min_valid_pairs: int = 2,
        min_unique_packs: int = 2,
        min_retention_fraction: float = 0.0,
        score_clip: float | None = 20.0,
    ) -> None:
        benign_raw = np.asarray(benign, dtype=np.float64)
        harmful_raw = np.asarray(harmful, dtype=np.float64)
        if benign_raw.ndim != 2 or harmful_raw.shape != benign_raw.shape:
            raise ValueError("benign and harmful endpoints must be equal [pair, hidden] matrices")
        if len(benign_raw) < 2:
            raise ValueError("A risk field requires at least two counterfactual pairs")
        if not np.isfinite(benign_raw).all() or not np.isfinite(harmful_raw).all():
            raise ValueError("Counterfactual endpoints contain non-finite values")
        pair_array = np.asarray(list(pair_ids)).astype(str)
        pack_array = np.asarray(list(pack_ids)).astype(str)
        if pair_array.shape != (len(benign_raw),) or pack_array.shape != pair_array.shape:
            raise ValueError("pair_ids and pack_ids must align with endpoint matrices")
        if len(set(pair_array.tolist())) != len(pair_array):
            raise ValueError("pair_ids must be unique within one risk-field view")
        if k < 1:
            raise ValueError("k must be positive")
        if min_valid_pairs < 2 or min_unique_packs < 2:
            raise ValueError("risk-field minimum pair/pack counts must be at least two")
        if not 0.0 <= float(min_retention_fraction) <= 1.0:
            raise ValueError("min_retention_fraction must lie in [0, 1]")

        benign_unit = l2_normalize(benign_raw)
        harmful_unit = l2_normalize(harmful_raw)
        arrows = harmful_unit - benign_unit
        arrow_norms = np.linalg.norm(arrows, axis=1)
        midpoints = 0.5 * (benign_unit + harmful_unit)
        midpoint_norms = np.linalg.norm(midpoints, axis=1)
        keep = (arrow_norms >= float(min_arrow_norm)) & (midpoint_norms > EPS)
        pairs_kept = int(keep.sum())
        packs_kept = len(set(pack_array[keep].tolist()))
        retention_fraction = pairs_kept / len(keep)
        required_pairs = max(2, int(min_valid_pairs))
        required_packs = max(2, int(min_unique_packs))
        if (
            pairs_kept < required_pairs
            or packs_kept < required_packs
            or retention_fraction < float(min_retention_fraction)
        ):
            diagnostics = {
                "status": "degenerate_risk_field",
                "pairs_total": int(len(keep)),
                "pairs_kept": pairs_kept,
                "unique_packs_kept": packs_kept,
                "retention_fraction": retention_fraction,
                "min_arrow_norm": float(min_arrow_norm),
                "required_pairs": required_pairs,
                "required_unique_packs": required_packs,
                "required_retention_fraction": float(min_retention_fraction),
                "arrow_norm_min": float(np.min(arrow_norms)),
                "arrow_norm_median": float(np.median(arrow_norms)),
                "arrow_norm_max": float(np.max(arrow_norms)),
                "nonzero_midpoints": int(np.sum(midpoint_norms > EPS)),
            }
            raise DegenerateRiskFieldError(
                "Risk field failed the retained-arrow quality gate: "
                f"kept={pairs_kept}, packs={packs_kept}, total={len(keep)}, "
                f"retention={retention_fraction:.6g}, "
                f"arrow_norm_max={float(np.max(arrow_norms)):.8g}",
                diagnostics,
            )
        self.benign = benign_unit[keep]
        self.harmful = harmful_unit[keep]
        self.arrows = arrows[keep]
        self.arrow_norm_squared = np.sum(self.arrows * self.arrows, axis=1)
        self.midpoints = midpoints[keep]
        self.midpoint_unit = l2_normalize(self.midpoints)
        self.pair_ids = pair_array[keep]
        self.pack_ids = pack_array[keep]
        self.k = int(k)
        self.min_arrow_norm = float(min_arrow_norm)
        self.pairs_total = int(len(keep))
        self.retention_fraction = retention_fraction
        self.score_clip = score_clip

    @property
    def hidden_dim(self) -> int:
        return int(self.benign.shape[1])

    def _neighbors(
        self,
        distances: np.ndarray,
        excluded_pack: str | None,
    ) -> np.ndarray:
        order = np.argsort(distances, kind="mergesort")
        selected: list[int] = []
        seen: set[str] = set()
        for raw_index in order.tolist():
            current_pack = str(self.pack_ids[raw_index])
            if excluded_pack and current_pack == excluded_pack:
                continue
            if current_pack in seen:
                continue
            seen.add(current_pack)
            selected.append(raw_index)
            if len(selected) >= self.k:
                break
        if not selected:
            raise ValueError("No eligible counterfactual neighbors remain after exclusion")
        return np.asarray(selected, dtype=np.int64)

    def score(
        self,
        vectors: np.ndarray,
        *,
        exclude_pack_ids: Iterable[str | None] | None = None,
    ) -> RiskFieldScores:
        values = np.asarray(vectors, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.hidden_dim:
            raise ValueError(f"Expected [sample, {self.hidden_dim}] vectors, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("Query vectors contain non-finite values")
        normalized = l2_normalize(values)
        exclusions = (
            list(exclude_pack_ids)
            if exclude_pack_ids is not None
            else [None] * len(normalized)
        )
        if len(exclusions) != len(normalized):
            raise ValueError("exclude_pack_ids must align with query vectors")

        cosine_distances = 1.0 - normalized @ self.midpoint_unit.T
        output_scores = np.empty(len(normalized), dtype=np.float64)
        support = np.empty(len(normalized), dtype=np.float64)
        neighbor_pair_ids: list[list[str]] = []
        neighbor_pack_ids: list[list[str]] = []
        neighbor_coordinates: list[list[float]] = []
        for row_index, value in enumerate(normalized):
            neighbors = self._neighbors(cosine_distances[row_index], exclusions[row_index])
            # Equivalent to (||z-b||^2 - ||z-q||^2) / ||q-b||^2.
            coordinates = (
                2.0
                * np.sum((value[None, :] - self.midpoints[neighbors]) * self.arrows[neighbors], axis=1)
                / np.maximum(self.arrow_norm_squared[neighbors], EPS)
            )
            if self.score_clip is not None:
                coordinates = np.clip(coordinates, -self.score_clip, self.score_clip)
            output_scores[row_index] = float(np.median(coordinates))
            support[row_index] = float(cosine_distances[row_index, neighbors[0]])
            neighbor_pair_ids.append(self.pair_ids[neighbors].astype(str).tolist())
            neighbor_pack_ids.append(self.pack_ids[neighbors].astype(str).tolist())
            neighbor_coordinates.append(coordinates.astype(float).tolist())
        return RiskFieldScores(
            scores=output_scores,
            nearest_midpoint_distance=support,
            neighbor_pair_ids=neighbor_pair_ids,
            neighbor_pack_ids=neighbor_pack_ids,
            neighbor_coordinates=neighbor_coordinates,
        )

    def diagnostics(self) -> dict[str, Any]:
        norms = np.sqrt(self.arrow_norm_squared)
        return {
            "pairs_total": self.pairs_total,
            "pairs": int(len(self.pair_ids)),
            "unique_packs": int(len(set(self.pack_ids.tolist()))),
            "retention_fraction": self.retention_fraction,
            "hidden_dim": self.hidden_dim,
            "k": self.k,
            "arrow_norm_min": float(np.min(norms)),
            "arrow_norm_median": float(np.median(norms)),
            "arrow_norm_max": float(np.max(norms)),
        }
