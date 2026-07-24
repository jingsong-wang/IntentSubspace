from __future__ import annotations

from typing import Iterable

import numpy as np

from .risk_field import EPS, l2_normalize


class TwoSidedKNN:
    def __init__(
        self,
        benign: np.ndarray,
        harmful: np.ndarray,
        pack_ids: Iterable[str],
        *,
        k: int = 5,
    ) -> None:
        self.benign = l2_normalize(np.asarray(benign, dtype=np.float64))
        self.harmful = l2_normalize(np.asarray(harmful, dtype=np.float64))
        if self.benign.shape != self.harmful.shape or self.benign.ndim != 2:
            raise ValueError("TwoSidedKNN endpoints must be aligned matrices")
        self.pack_ids = np.asarray(list(pack_ids)).astype(str)
        if self.pack_ids.shape != (len(self.benign),):
            raise ValueError("pack_ids do not align with reference endpoints")
        self.k = int(k)

    def score(
        self,
        vectors: np.ndarray,
        *,
        exclude_pack_ids: Iterable[str | None] | None = None,
    ) -> np.ndarray:
        values = l2_normalize(np.atleast_2d(np.asarray(vectors, dtype=np.float64)))
        exclusions = list(exclude_pack_ids) if exclude_pack_ids is not None else [None] * len(values)
        output = np.empty(len(values), dtype=np.float64)
        for row_index, value in enumerate(values):
            eligible = self.pack_ids != exclusions[row_index] if exclusions[row_index] else np.ones(len(self.pack_ids), dtype=bool)
            benign_distance = np.linalg.norm(self.benign[eligible] - value[None, :], axis=1)
            harmful_distance = np.linalg.norm(self.harmful[eligible] - value[None, :], axis=1)
            k = min(max(self.k, 1), len(benign_distance), len(harmful_distance))
            output[row_index] = float(
                np.partition(benign_distance, k - 1)[k - 1]
                - np.partition(harmful_distance, k - 1)[k - 1]
            )
        return output


class MeanArrow:
    def __init__(self, benign: np.ndarray, harmful: np.ndarray) -> None:
        safe = l2_normalize(np.asarray(benign, dtype=np.float64))
        unsafe = l2_normalize(np.asarray(harmful, dtype=np.float64))
        if safe.shape != unsafe.shape or safe.ndim != 2:
            raise ValueError("MeanArrow endpoints must be aligned matrices")
        self.center = 0.5 * (safe + unsafe).mean(axis=0)
        direction = (unsafe - safe).mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if norm <= EPS:
            raise ValueError("Mean counterfactual arrow is zero")
        self.direction = direction / norm

    def score(self, vectors: np.ndarray, **_: object) -> np.ndarray:
        values = l2_normalize(np.atleast_2d(np.asarray(vectors, dtype=np.float64)))
        return (values - self.center[None, :]) @ self.direction
