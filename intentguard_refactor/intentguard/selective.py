from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np


class SelectiveRoute(str, Enum):
    CONFIDENT_SAFE = "confident_safe"
    REVIEW = "review"
    CONFIDENT_DANGEROUS = "confident_dangerous"


@dataclass(frozen=True)
class SelectiveThresholds:
    safe_max: float
    danger_min: float
    safe_enabled: bool = True
    danger_enabled: bool = True

    def __post_init__(self) -> None:
        for name, value in (("safe_max", self.safe_max), ("danger_min", self.danger_min)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.safe_enabled and self.danger_enabled and self.safe_max >= self.danger_min:
            raise ValueError("safe_max must be smaller than danger_min")


@dataclass(frozen=True)
class SelectiveDecision:
    route: SelectiveRoute
    probability: float
    route_margin: float
    route_error_upper_bound: float | None

    @property
    def requires_intervention(self) -> bool:
        return self.route is not SelectiveRoute.CONFIDENT_SAFE

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["route"] = self.route.value
        result["requires_intervention"] = self.requires_intervention
        return result


def _validate_binary_inputs(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=int).reshape(-1)
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) != len(scores):
        raise ValueError("labels and probabilities must align")
    if len(y) == 0:
        raise ValueError("selective calibration requires at least one row")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("labels must be binary 0/1")
    if len(np.unique(y)) != 2:
        raise ValueError("selective calibration requires both harmful and benign labels")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    return y, scores


def wilson_upper_bound(successes: int, total: int, confidence: float = 0.95) -> float | None:
    if total <= 0:
        return None
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie in (0.5, 1)")
    z = NormalDist().inv_cdf(confidence)
    n = float(total)
    p = float(successes) / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return float(min(1.0, center + radius))


def assign_routes(
    probabilities: np.ndarray,
    thresholds: SelectiveThresholds,
) -> np.ndarray:
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    routes = np.full(len(scores), SelectiveRoute.REVIEW.value, dtype=object)
    if thresholds.safe_enabled:
        routes[scores <= thresholds.safe_max] = SelectiveRoute.CONFIDENT_SAFE.value
    if thresholds.danger_enabled:
        routes[scores >= thresholds.danger_min] = SelectiveRoute.CONFIDENT_DANGEROUS.value
    return routes.astype(str)


def decide_route(
    probability: float,
    thresholds: SelectiveThresholds,
    *,
    safe_error_upper_bound: float | None = None,
    danger_error_upper_bound: float | None = None,
) -> SelectiveDecision:
    score = float(probability)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if thresholds.safe_enabled and score <= thresholds.safe_max:
        scale = max(thresholds.safe_max, 1e-12)
        margin = (thresholds.safe_max - score) / scale
        return SelectiveDecision(
            route=SelectiveRoute.CONFIDENT_SAFE,
            probability=score,
            route_margin=float(max(0.0, min(1.0, margin))),
            route_error_upper_bound=safe_error_upper_bound,
        )
    if thresholds.danger_enabled and score >= thresholds.danger_min:
        scale = max(1.0 - thresholds.danger_min, 1e-12)
        margin = (score - thresholds.danger_min) / scale
        return SelectiveDecision(
            route=SelectiveRoute.CONFIDENT_DANGEROUS,
            probability=score,
            route_margin=float(max(0.0, min(1.0, margin))),
            route_error_upper_bound=danger_error_upper_bound,
        )
    return SelectiveDecision(
        route=SelectiveRoute.REVIEW,
        probability=score,
        route_margin=0.0,
        route_error_upper_bound=None,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def selective_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: SelectiveThresholds,
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    y, scores = _validate_binary_inputs(labels, probabilities)
    routes = assign_routes(scores, thresholds)
    safe = routes == SelectiveRoute.CONFIDENT_SAFE.value
    review = routes == SelectiveRoute.REVIEW.value
    danger = routes == SelectiveRoute.CONFIDENT_DANGEROUS.value
    harmful = y == 1
    benign = y == 0

    safe_n = int(safe.sum())
    review_n = int(review.sum())
    danger_n = int(danger.sum())
    harmful_n = int(harmful.sum())
    benign_n = int(benign.sum())
    safe_errors = int(np.sum(safe & harmful))
    danger_errors = int(np.sum(danger & benign))
    harmful_review = int(np.sum(review & harmful))
    benign_review = int(np.sum(review & benign))

    return {
        "n": int(len(y)),
        "positive_n": harmful_n,
        "negative_n": benign_n,
        "thresholds": {
            "safe_max": float(thresholds.safe_max),
            "danger_min": float(thresholds.danger_min),
            "safe_enabled": bool(thresholds.safe_enabled),
            "danger_enabled": bool(thresholds.danger_enabled),
        },
        "route_counts": {
            SelectiveRoute.CONFIDENT_SAFE.value: safe_n,
            SelectiveRoute.REVIEW.value: review_n,
            SelectiveRoute.CONFIDENT_DANGEROUS.value: danger_n,
        },
        "confident_safe_count": safe_n,
        "confident_safe_rate": _rate(safe_n, len(y)),
        "confident_safe_error_count": safe_errors,
        "confident_safe_error_rate": _rate(safe_errors, safe_n),
        "confident_safe_error_upper_bound": wilson_upper_bound(
            safe_errors, safe_n, confidence
        ),
        "harmful_unsafe_escape_count": safe_errors,
        "harmful_unsafe_escape_rate": _rate(safe_errors, harmful_n),
        "harmful_unsafe_escape_upper_bound": wilson_upper_bound(
            safe_errors, harmful_n, confidence
        ),
        "confident_dangerous_count": danger_n,
        "confident_dangerous_rate": _rate(danger_n, len(y)),
        "confident_dangerous_error_count": danger_errors,
        "confident_dangerous_error_rate": _rate(danger_errors, danger_n),
        "confident_dangerous_error_upper_bound": wilson_upper_bound(
            danger_errors, danger_n, confidence
        ),
        "benign_hard_refusal_count": danger_errors,
        "benign_hard_refusal_rate": _rate(danger_errors, benign_n),
        "benign_hard_refusal_upper_bound": wilson_upper_bound(
            danger_errors, benign_n, confidence
        ),
        "review_count": review_n,
        "review_rate": _rate(review_n, len(y)),
        "harmful_review_count": harmful_review,
        "harmful_review_rate": _rate(harmful_review, harmful_n),
        "benign_review_count": benign_review,
        "benign_review_rate": _rate(benign_review, benign_n),
        "harmful_routing_recall": _rate(harmful_review + int(np.sum(danger & harmful)), harmful_n),
        "direct_decision_coverage": _rate(safe_n + danger_n, len(y)),
    }


def _group_error_ok(
    labels: np.ndarray,
    selected: np.ndarray,
    groups: np.ndarray | None,
    *,
    error_label: int,
    maximum_error: float,
    minimum_selected: int,
) -> tuple[bool, dict[str, Any]]:
    if groups is None:
        return True, {}
    report: dict[str, Any] = {}
    passed = True
    for group in sorted(set(groups.astype(str).tolist())):
        mask = selected & (groups.astype(str) == group)
        total = int(mask.sum())
        if total < minimum_selected:
            continue
        errors = int(np.sum(mask & (labels == error_label)))
        rate = errors / total
        report[group] = {"n": total, "errors": errors, "error_rate": rate}
        passed = passed and rate <= maximum_error
    return passed, report


def choose_selective_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    maximum_safe_error: float = 0.01,
    maximum_danger_error: float = 0.01,
    maximum_harmful_escape: float = 0.05,
    maximum_benign_hard_refusal: float = 0.05,
    confidence: float = 0.95,
    groups: Iterable[str] | np.ndarray | None = None,
    minimum_group_selected: int = 10,
) -> tuple[SelectiveThresholds, dict[str, Any]]:
    y, scores = _validate_binary_inputs(labels, probabilities)
    for name, value in (
        ("maximum_safe_error", maximum_safe_error),
        ("maximum_danger_error", maximum_danger_error),
        ("maximum_harmful_escape", maximum_harmful_escape),
        ("maximum_benign_hard_refusal", maximum_benign_hard_refusal),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    group_values = None if groups is None else np.asarray(list(groups)).astype(str)
    if group_values is not None and len(group_values) != len(y):
        raise ValueError("groups must align with labels")

    harmful_n = int(np.sum(y == 1))
    benign_n = int(np.sum(y == 0))
    unique_scores = np.unique(scores)

    safe_candidates: list[dict[str, Any]] = [
        {
            "enabled": False,
            "threshold": 0.0,
            "n": 0,
            "errors": 0,
            "error_rate": None,
            "error_upper_bound": None,
            "class_error_rate": 0.0,
            "class_error_upper_bound": wilson_upper_bound(0, harmful_n, confidence),
            "groups": {},
            "constraints_met": True,
        }
    ]
    danger_candidates: list[dict[str, Any]] = [
        {
            "enabled": False,
            "threshold": 1.0,
            "n": 0,
            "errors": 0,
            "error_rate": None,
            "error_upper_bound": None,
            "class_error_rate": 0.0,
            "class_error_upper_bound": wilson_upper_bound(0, benign_n, confidence),
            "groups": {},
            "constraints_met": True,
        }
    ]

    for threshold in unique_scores.tolist():
        selected = scores <= float(threshold)
        total = int(selected.sum())
        errors = int(np.sum(selected & (y == 1)))
        error_upper = wilson_upper_bound(errors, total, confidence)
        escape_upper = wilson_upper_bound(errors, harmful_n, confidence)
        group_ok, group_report = _group_error_ok(
            y,
            selected,
            group_values,
            error_label=1,
            maximum_error=maximum_safe_error,
            minimum_selected=minimum_group_selected,
        )
        constraints_met = bool(
            error_upper is not None
            and escape_upper is not None
            and error_upper <= maximum_safe_error
            and escape_upper <= maximum_harmful_escape
            and group_ok
        )
        safe_candidates.append(
            {
                "enabled": True,
                "threshold": float(threshold),
                "n": total,
                "errors": errors,
                "error_rate": errors / total,
                "error_upper_bound": error_upper,
                "class_error_rate": errors / harmful_n,
                "class_error_upper_bound": escape_upper,
                "groups": group_report,
                "constraints_met": constraints_met,
            }
        )

    for threshold in unique_scores.tolist():
        selected = scores >= float(threshold)
        total = int(selected.sum())
        errors = int(np.sum(selected & (y == 0)))
        error_upper = wilson_upper_bound(errors, total, confidence)
        hard_refusal_upper = wilson_upper_bound(errors, benign_n, confidence)
        group_ok, group_report = _group_error_ok(
            y,
            selected,
            group_values,
            error_label=0,
            maximum_error=maximum_danger_error,
            minimum_selected=minimum_group_selected,
        )
        constraints_met = bool(
            error_upper is not None
            and hard_refusal_upper is not None
            and error_upper <= maximum_danger_error
            and hard_refusal_upper <= maximum_benign_hard_refusal
            and group_ok
        )
        danger_candidates.append(
            {
                "enabled": True,
                "threshold": float(threshold),
                "n": total,
                "errors": errors,
                "error_rate": errors / total,
                "error_upper_bound": error_upper,
                "class_error_rate": errors / benign_n,
                "class_error_upper_bound": hard_refusal_upper,
                "groups": group_report,
                "constraints_met": constraints_met,
            }
        )

    feasible_safe = [item for item in safe_candidates if item["constraints_met"]]
    feasible_danger = [item for item in danger_candidates if item["constraints_met"]]
    best: tuple[tuple[float, ...], dict[str, Any], dict[str, Any]] | None = None
    for safe in feasible_safe:
        for danger in feasible_danger:
            if safe["enabled"] and danger["enabled"] and safe["threshold"] >= danger["threshold"]:
                continue
            key = (
                float(bool(danger["enabled"])),
                -float(danger["error_rate"] if danger["error_rate"] is not None else 1.0),
                float(danger["n"]),
                float(bool(safe["enabled"])),
                -float(safe["error_rate"] if safe["error_rate"] is not None else 1.0),
                float(safe["n"]),
                float(safe["n"] + danger["n"]),
            )
            if best is None or key > best[0]:
                best = (key, safe, danger)
    if best is None:
        raise RuntimeError("selective threshold search produced no valid fallback")

    safe = best[1]
    danger = best[2]
    thresholds = SelectiveThresholds(
        safe_max=float(safe["threshold"]),
        danger_min=float(danger["threshold"]),
        safe_enabled=bool(safe["enabled"]),
        danger_enabled=bool(danger["enabled"]),
    )
    metrics = selective_metrics(y, scores, thresholds, confidence=confidence)
    deployable = bool(safe["enabled"] and danger["enabled"])
    return thresholds, {
        "format_version": "CISR_v4_selective_calibration_v1",
        "coverage_confidence": confidence,
        "targets": {
            "maximum_confident_safe_error": maximum_safe_error,
            "maximum_confident_dangerous_error": maximum_danger_error,
            "maximum_harmful_unsafe_escape": maximum_harmful_escape,
            "maximum_benign_hard_refusal": maximum_benign_hard_refusal,
        },
        "thresholds": metrics["thresholds"],
        "confident_safe_selection": safe,
        "confident_dangerous_selection": danger,
        "selective_metrics": metrics,
        "deployment_constraints_met": deployable,
    }
