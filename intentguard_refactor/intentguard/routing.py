from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal


class RiskRefusalRegion(str, Enum):
    """Operational regions induced by risk and safe-refusal scores."""

    SAFE_ANSWER = "safe_answer"
    OVER_REFUSAL = "over_refusal"
    JAILBREAK = "jailbreak"
    ALREADY_SAFE = "already_safe"
    RISK_UNCERTAIN = "risk_uncertain"
    REFUSAL_UNCERTAIN = "refusal_uncertain"
    REFUSAL_MISSING = "refusal_missing"
    ORACLE_RISK = "oracle_risk"
    ORACLE_BENIGN = "oracle_benign"
    FORCED = "forced"


RoutingAction = Literal["normal", "route", "monitor"]
MissingRefusalAction = Literal["monitor", "route_extreme_risk", "route"]


@dataclass(frozen=True)
class RoutingThresholds:
    """Two-sided thresholds leave an explicit abstention band for each score."""

    risk_low: float
    risk_high: float
    refusal_low: float
    refusal_high: float
    extreme_risk: float = 1.0
    missing_refusal_action: MissingRefusalAction = "monitor"

    def __post_init__(self) -> None:
        values = {
            "risk_low": self.risk_low,
            "risk_high": self.risk_high,
            "refusal_low": self.refusal_low,
            "refusal_high": self.refusal_high,
            "extreme_risk": self.extreme_risk,
        }
        for name, value in values.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.risk_low > self.risk_high:
            raise ValueError("risk_low must be <= risk_high")
        if self.refusal_low > self.refusal_high:
            raise ValueError("refusal_low must be <= refusal_high")
        if self.extreme_risk < self.risk_high:
            raise ValueError("extreme_risk must be >= risk_high")
        if self.missing_refusal_action not in {"monitor", "route_extreme_risk", "route"}:
            raise ValueError(
                f"Unsupported missing_refusal_action={self.missing_refusal_action!r}"
            )


@dataclass(frozen=True)
class RoutingDecision:
    region: RiskRefusalRegion
    action: RoutingAction
    triggered: bool
    should_verify: bool
    risk_score: float | None
    refusal_score: float | None
    source: Literal["live", "oracle", "forced"]
    reason: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["region"] = self.region.value
        return result


def _validate_score(name: str, score: float | None) -> float | None:
    if score is None:
        return None
    value = float(score)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {score}")
    return value


class SynchronizedRoutingPolicy:
    """Decide whether the learned safety transform should run.

    ``refusal_score`` means probability of a *safe* refusal or safe pivot. A
    lexical refusal that still leaks harmful instructions must be labelled 0.
    Oracle mode deliberately bypasses both detectors so intervention efficacy
    can be measured on every ground-truth risk sample.
    """

    def __init__(self, thresholds: RoutingThresholds):
        self.thresholds = thresholds

    def decide(
        self,
        risk_score: float | None,
        refusal_score: float | None,
        *,
        oracle_risk: bool | None = None,
        force_route: bool = False,
    ) -> RoutingDecision:
        risk = _validate_score("risk_score", risk_score)
        refusal = _validate_score("refusal_score", refusal_score)

        if force_route:
            return RoutingDecision(
                region=RiskRefusalRegion.FORCED,
                action="route",
                triggered=True,
                should_verify=True,
                risk_score=risk,
                refusal_score=refusal,
                source="forced",
                reason="The caller explicitly forced the safety route.",
            )

        if oracle_risk is not None:
            if oracle_risk:
                return RoutingDecision(
                    region=RiskRefusalRegion.ORACLE_RISK,
                    action="route",
                    triggered=True,
                    should_verify=True,
                    risk_score=risk,
                    refusal_score=refusal,
                    source="oracle",
                    reason="Ground-truth risk forced routing; detector and refusal gate were bypassed.",
                )
            return RoutingDecision(
                region=RiskRefusalRegion.ORACLE_BENIGN,
                action="normal",
                triggered=False,
                should_verify=False,
                risk_score=risk,
                refusal_score=refusal,
                source="oracle",
                reason="Ground-truth benign sample remains on the base path in oracle mode.",
            )

        if risk is None:
            raise ValueError("Live routing requires risk_score.")

        thresholds = self.thresholds
        if risk < thresholds.risk_low:
            if refusal is not None and refusal >= thresholds.refusal_high:
                return RoutingDecision(
                    region=RiskRefusalRegion.OVER_REFUSAL,
                    action="normal",
                    triggered=False,
                    should_verify=True,
                    risk_score=risk,
                    refusal_score=refusal,
                    source="live",
                    reason="Risk is low but safe-refusal propensity is high; record a utility error, do not safety-route.",
                )
            return RoutingDecision(
                region=RiskRefusalRegion.SAFE_ANSWER,
                action="normal",
                triggered=False,
                should_verify=False,
                risk_score=risk,
                refusal_score=refusal,
                source="live",
                reason="Risk is below the low threshold.",
            )

        if risk < thresholds.risk_high:
            return RoutingDecision(
                region=RiskRefusalRegion.RISK_UNCERTAIN,
                action="monitor",
                triggered=False,
                should_verify=True,
                risk_score=risk,
                refusal_score=refusal,
                source="live",
                reason="Risk lies in the calibration abstention band.",
            )

        if refusal is None:
            action: RoutingAction = "monitor"
            if thresholds.missing_refusal_action == "route":
                action = "route"
            elif (
                thresholds.missing_refusal_action == "route_extreme_risk"
                and risk >= thresholds.extreme_risk
            ):
                action = "route"
            return RoutingDecision(
                region=RiskRefusalRegion.REFUSAL_MISSING,
                action=action,
                triggered=action == "route",
                should_verify=True,
                risk_score=risk,
                refusal_score=None,
                source="live",
                reason="Risk is high but no calibrated safe-refusal score is available.",
            )

        if refusal < thresholds.refusal_low:
            return RoutingDecision(
                region=RiskRefusalRegion.JAILBREAK,
                action="route",
                triggered=True,
                should_verify=True,
                risk_score=risk,
                refusal_score=refusal,
                source="live",
                reason="Risk is high and safe-refusal propensity is low.",
            )

        if refusal >= thresholds.refusal_high:
            return RoutingDecision(
                region=RiskRefusalRegion.ALREADY_SAFE,
                action="monitor",
                triggered=False,
                should_verify=True,
                risk_score=risk,
                refusal_score=refusal,
                source="live",
                reason="Risk is high but the base model is already on a calibrated safe-refusal path.",
            )

        return RoutingDecision(
            region=RiskRefusalRegion.REFUSAL_UNCERTAIN,
            action="monitor",
            triggered=False,
            should_verify=True,
            risk_score=risk,
            refusal_score=refusal,
            source="live",
            reason="Safe-refusal propensity lies in the calibration abstention band.",
        )
