from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class InterventionRole(str, Enum):
    """Mutually exclusive roles used by CSRL training and evaluation."""

    ROUTE_POSITIVE = "route_positive"
    SAFE_REFUSAL_TEACHER = "safe_refusal_teacher"
    SAFE_TARGET_CONTROL = "safe_target_control"
    RETAIN_BENIGN = "retain_benign"
    OVER_REFUSAL_CONTROL = "over_refusal_control"
    BENIGN_JUDGE_FAILURE = "benign_judge_failure"
    UNJUDGED = "unjudged"


@dataclass(frozen=True)
class OutcomeSignals:
    harmful: bool | None
    refused: bool | None

    @property
    def safe_refusal(self) -> bool:
        return self.harmful is False and self.refused is True

    @property
    def leaky_refusal(self) -> bool:
        return self.harmful is True and self.refused is True


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def outcome_signals(judge_row: dict[str, Any] | None) -> OutcomeSignals:
    if not judge_row:
        return OutcomeSignals(harmful=None, refused=None)
    harmful = _optional_bool(judge_row.get("judge_harmful"))
    refused = _optional_bool(judge_row.get("refusal_heuristic"))
    if refused is None:
        state = str(judge_row.get("refusal_state", "")).strip().lower()
        if state == "refused":
            refused = True
        elif state == "not_refused":
            refused = False
    return OutcomeSignals(harmful=harmful, refused=refused)


def intervention_role(label: int, signals: OutcomeSignals) -> InterventionRole:
    is_risk = int(label) == 1
    if signals.harmful is None:
        return InterventionRole.UNJUDGED
    if is_risk:
        if signals.harmful:
            return InterventionRole.ROUTE_POSITIVE
        if signals.safe_refusal:
            return InterventionRole.SAFE_REFUSAL_TEACHER
        return InterventionRole.SAFE_TARGET_CONTROL
    if signals.harmful:
        return InterventionRole.BENIGN_JUDGE_FAILURE
    if signals.safe_refusal:
        return InterventionRole.OVER_REFUSAL_CONTROL
    return InterventionRole.RETAIN_BENIGN


def build_manifest_row(
    detection_row: dict[str, Any],
    judge_row: dict[str, Any] | None,
) -> dict[str, Any]:
    signals = outcome_signals(judge_row)
    label = int(detection_row.get("label", 0))
    role = intervention_role(label, signals)
    detected = _optional_bool(
        detection_row.get("cisr_detected", detection_row.get("intent_detected"))
    )
    risk_score = detection_row.get(
        "cisr_probability", detection_row.get("intent_score")
    )
    risk_threshold = detection_row.get(
        "cisr_threshold", detection_row.get("intent_threshold")
    )
    split = str(detection_row.get("evaluation_split", ""))
    ground_truth_risk = label == 1
    safe_refusal = signals.safe_refusal
    live_candidate = bool(detected) and not safe_refusal

    return {
        "id": str(detection_row.get("id", "")),
        "pair_key": detection_row.get("pair_key"),
        "evaluation_split": split,
        "label": label,
        "label_name": detection_row.get("label_name"),
        "condition": detection_row.get("condition"),
        "intent_family": detection_row.get("intent_family"),
        "image_role": detection_row.get("image_role"),
        "risk_score": risk_score,
        "risk_threshold": risk_threshold,
        "risk_detected": detected,
        "judge_harmful": signals.harmful,
        "refusal_heuristic": signals.refused,
        "safe_refusal": safe_refusal,
        "leaky_refusal": signals.leaky_refusal,
        "intervention_role": role.value,
        "live_route_candidate_observed": live_candidate,
        "oracle_route": ground_truth_risk,
        "oracle_effectiveness_eval": ground_truth_risk and signals.harmful is True,
        "oracle_non_regression_eval": ground_truth_risk and signals.harmful is False,
        "fit_eligible": split == "train"
        and role
        in {
            InterventionRole.ROUTE_POSITIVE,
            InterventionRole.SAFE_REFUSAL_TEACHER,
            InterventionRole.SAFE_TARGET_CONTROL,
            InterventionRole.RETAIN_BENIGN,
            InterventionRole.OVER_REFUSAL_CONTROL,
        },
        "prompt": (
            (judge_row or {}).get("prompt")
            or detection_row.get("prompt")
            or detection_row.get("prompt_text")
        ),
        "base_response": (judge_row or {}).get("response"),
    }


def build_manifest(
    detection_rows: Iterable[dict[str, Any]],
    judge_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    judge_by_id = {str(row.get("id", "")): row for row in judge_rows}
    result = []
    for row in detection_rows:
        sample_id = str(row.get("id", ""))
        result.append(build_manifest_row(row, judge_by_id.get(sample_id)))
    return result


def summarize_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "role_counts": dict(sorted(Counter(str(row["intervention_role"]) for row in rows).items())),
        "split_counts": dict(sorted(Counter(str(row["evaluation_split"]) for row in rows).items())),
        "live_route_candidate_observed": sum(
            bool(row["live_route_candidate_observed"]) for row in rows
        ),
        "oracle_route": sum(bool(row["oracle_route"]) for row in rows),
        "oracle_effectiveness_eval": sum(
            bool(row["oracle_effectiveness_eval"]) for row in rows
        ),
        "oracle_non_regression_eval": sum(
            bool(row["oracle_non_regression_eval"]) for row in rows
        ),
        "fit_eligible": sum(bool(row["fit_eligible"]) for row in rows),
    }

