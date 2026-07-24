from __future__ import annotations

import math
from collections import defaultdict
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np


EPS = 1e-12


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        raise ValueError("Cannot fit robust standardization without finite scores")
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    # 1.4826 makes MAD consistent with sigma for a normal reference distribution.
    scale = max(1.4826 * mad, 1e-6)
    return center, scale


def robust_standardize(values: np.ndarray, center: float, scale: float) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - float(center)) / max(float(scale), EPS)


def split_conformal_upper_threshold(values: np.ndarray, alpha: float) -> float:
    """Smallest strict decision threshold with finite-sample upper-tail rank control."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    array = np.sort(np.asarray(values, dtype=np.float64))
    array = array[np.isfinite(array)]
    if len(array) == 0:
        raise ValueError("Calibration group contains no finite scores")
    rank = min(len(array), int(math.ceil((len(array) + 1) * (1.0 - alpha))))
    quantile = float(array[rank - 1])
    # The detector uses score >= threshold. Move above ties at the selected order statistic.
    return float(np.nextafter(quantile, math.inf))


def worst_group_threshold(
    benign_scores: np.ndarray,
    groups: Iterable[str],
    *,
    alpha: float,
) -> tuple[float, dict[str, Any]]:
    scores = np.asarray(benign_scores, dtype=np.float64)
    group_values = np.asarray(list(groups)).astype(str)
    if group_values.shape != scores.shape:
        raise ValueError("Calibration groups must align with benign scores")
    reports: dict[str, Any] = {}
    thresholds: list[float] = []
    for group in sorted(set(group_values.tolist())):
        values = scores[group_values == group]
        threshold = split_conformal_upper_threshold(values, alpha)
        thresholds.append(threshold)
        reports[group] = {
            "n": int(len(values)),
            "threshold": threshold,
            "score_median": float(np.median(values)),
            "score_max": float(np.max(values)),
        }
    if not thresholds:
        raise ValueError("No benign calibration groups were provided")
    threshold = float(max(thresholds))
    for group, report in reports.items():
        mask = group_values == group
        false_positives = int(np.sum(scores[mask] >= threshold))
        report["false_positives_at_joint_threshold"] = false_positives
        report["empirical_fpr_at_joint_threshold"] = false_positives / int(mask.sum())
    return threshold, {"alpha": alpha, "groups": reports, "joint_threshold": threshold}


def support_radius(distances: np.ndarray, quantile: float = 0.99) -> float:
    if not 0.0 < quantile <= 1.0:
        raise ValueError("support quantile must be in (0, 1]")
    values = np.asarray(distances, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Cannot calibrate support radius without finite distances")
    return float(np.quantile(values, quantile, method="higher"))


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float] | None:
    if total <= 0:
        return None
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)
