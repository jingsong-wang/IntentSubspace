from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fit_subspaces import infer_refusal_label  # noqa: E402
from intentguard.intervention_data import (  # noqa: E402
    InterventionRole,
    build_manifest_row,
    intervention_role,
    outcome_signals,
)
from intentguard.intervention_eval import evaluate_oracle_bypass  # noqa: E402
from intentguard.routing import (  # noqa: E402
    RiskRefusalRegion,
    RoutingThresholds,
    SynchronizedRoutingPolicy,
)
from intentguard.outcomes import derive_response_outcome  # noqa: E402

try:  # noqa: E402
    import torch
    from torch import nn

    from intentguard.intervention import (  # noqa: E402
        LowRankSafetyAdapter,
        SafeLayerRouter,
        compute_intervention_loss,
    )
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


class RoutingPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = SynchronizedRoutingPolicy(
            RoutingThresholds(
                risk_low=0.70,
                risk_high=0.80,
                refusal_low=0.30,
                refusal_high=0.70,
                extreme_risk=0.95,
                missing_refusal_action="route_extreme_risk",
            )
        )

    def test_jailbreak_window_routes(self) -> None:
        decision = self.policy.decide(0.90, 0.10)
        self.assertEqual(decision.region, RiskRefusalRegion.JAILBREAK)
        self.assertTrue(decision.triggered)

    def test_already_safe_is_monitored_without_routing(self) -> None:
        decision = self.policy.decide(0.90, 0.90)
        self.assertEqual(decision.region, RiskRefusalRegion.ALREADY_SAFE)
        self.assertFalse(decision.triggered)
        self.assertTrue(decision.should_verify)

    def test_over_refusal_is_recorded_but_not_reverse_steered(self) -> None:
        decision = self.policy.decide(0.10, 0.90)
        self.assertEqual(decision.region, RiskRefusalRegion.OVER_REFUSAL)
        self.assertEqual(decision.action, "normal")

    def test_abstention_band_does_not_activate_adapter(self) -> None:
        decision = self.policy.decide(0.75, 0.10)
        self.assertEqual(decision.region, RiskRefusalRegion.RISK_UNCERTAIN)
        self.assertFalse(decision.triggered)

    def test_missing_refusal_only_routes_extreme_risk(self) -> None:
        self.assertFalse(self.policy.decide(0.90, None).triggered)
        self.assertTrue(self.policy.decide(0.97, None).triggered)

    def test_oracle_bypass_routes_every_ground_truth_risk(self) -> None:
        decision = self.policy.decide(0.01, 0.99, oracle_risk=True)
        self.assertEqual(decision.region, RiskRefusalRegion.ORACLE_RISK)
        self.assertTrue(decision.triggered)


class InterventionDataTest(unittest.TestCase):
    def test_leaky_refusal_is_a_route_positive(self) -> None:
        signals = outcome_signals(
            {"judge_harmful": True, "refusal_heuristic": True}
        )
        self.assertTrue(signals.leaky_refusal)
        self.assertEqual(
            intervention_role(1, signals), InterventionRole.ROUTE_POSITIVE
        )

    def test_benign_safe_refusal_is_over_refusal_control(self) -> None:
        signals = outcome_signals(
            {"judge_harmful": False, "refusal_heuristic": True}
        )
        self.assertEqual(
            intervention_role(0, signals), InterventionRole.OVER_REFUSAL_CONTROL
        )

    def test_manifest_never_marks_test_rows_fit_eligible(self) -> None:
        row = build_manifest_row(
            {
                "id": "risk-test",
                "label": 1,
                "evaluation_split": "test",
                "cisr_detected": False,
                "cisr_probability": 0.1,
            },
            {"judge_harmful": True, "refusal_heuristic": False},
        )
        self.assertTrue(row["oracle_route"])
        self.assertTrue(row["oracle_effectiveness_eval"])
        self.assertFalse(row["fit_eligible"])

    def test_safe_refusal_label_excludes_harmful_leakage(self) -> None:
        self.assertEqual(
            infer_refusal_label(
                {
                    "judge_harmful": True,
                    "judge_harmfulness_score": 4,
                    "refusal_heuristic": True,
                },
                harmful_score_threshold=3,
            ),
            0,
        )

    def test_unknown_refusal_state_is_not_a_positive_label(self) -> None:
        self.assertIsNone(
            infer_refusal_label(
                {"response_outcome": "judge_safe_unclear_refusal"},
                harmful_score_threshold=3,
            )
        )

    def test_harmless_nonrefusal_is_not_relabelled_as_refusal(self) -> None:
        harmful, refusal_state, outcome = derive_response_outcome(
            {"parse_error": None, "harmfulness_score": 1},
            {"refusal_heuristic": False},
            harmful_score_threshold=3,
        )
        self.assertFalse(harmful)
        self.assertEqual(refusal_state, "not_refused")
        self.assertEqual(outcome, "safe_nonrefusal")


class OracleEvaluationTest(unittest.TestCase):
    def test_reports_conversion_and_non_regression_separately(self) -> None:
        manifest = [
            {"id": "failed", "oracle_route": True, "condition": "ocr", "intent_family": "cyber"},
            {"id": "safe", "oracle_route": True, "condition": "text", "intent_family": "cyber"},
            {"id": "benign", "oracle_route": False, "condition": "text", "intent_family": "none"},
        ]
        base = [
            {"id": "failed", "judge_harmful": True, "judge_harmfulness_score": 4},
            {"id": "safe", "judge_harmful": False, "judge_harmfulness_score": 1},
        ]
        post = [
            {"id": "failed", "judge_harmful": False, "judge_harmfulness_score": 1},
            {"id": "safe", "judge_harmful": False, "judge_harmfulness_score": 1},
        ]
        result = evaluate_oracle_bypass(manifest, base, post)
        self.assertTrue(result["all_ground_truth_risk_routed"])
        self.assertEqual(result["overall"]["harmful_to_safe_rate"], 1.0)
        self.assertEqual(result["overall"]["safe_to_harmful_rate"], 0.0)

    def test_requires_post_result_for_every_oracle_risk(self) -> None:
        with self.assertRaisesRegex(ValueError, "every ground-truth risk sample"):
            evaluate_oracle_bypass(
                [{"id": "missing", "oracle_route": True}],
                [{"id": "missing", "judge_harmful": True}],
                [],
            )


@unittest.skipIf(torch is None, "torch is not installed")
class SafeLayerRouterTest(unittest.TestCase):
    def make_adapter(self) -> LowRankSafetyAdapter:
        adapter = LowRankSafetyAdapter(hidden_size=4, rank=2, alpha=2.0)
        with torch.no_grad():
            adapter.down.weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            )
            adapter.up.weight.fill_(0.5)
        return adapter

    def test_hook_changes_only_active_rows_and_last_token(self) -> None:
        adapter = self.make_adapter()
        router = SafeLayerRouter(adapter)
        layer = nn.Identity()
        router.attach(layer)
        hidden = torch.tensor(
            [
                [[1.0, 2.0, 0.0, -1.0], [2.0, 0.0, -1.0, 1.0]],
                [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]],
            ]
        )
        try:
            with router.routing([True, False], token_scope="last", max_route_calls=1):
                routed = layer(hidden)
                exhausted = layer(hidden)
        finally:
            router.detach()

        self.assertTrue(torch.equal(routed[0, 0], hidden[0, 0]))
        self.assertFalse(torch.equal(routed[0, 1], hidden[0, 1]))
        self.assertTrue(torch.equal(routed[1], hidden[1]))
        self.assertTrue(torch.equal(exhausted, hidden))

    def test_representation_loss_backpropagates(self) -> None:
        adapter = self.make_adapter()
        base = torch.randn(4, 4)
        routed = adapter(base)
        teacher = base.clone()
        teacher[0] += 1.0
        teacher[1] += 1.0
        losses = compute_intervention_loss(
            base_hidden=base,
            routed_hidden=routed,
            route_mask=torch.tensor([True, True, False, False]),
            retain_mask=torch.tensor([False, False, True, True]),
            teacher_hidden=teacher,
        )
        losses["total"].backward()
        self.assertIsNotNone(adapter.up.weight.grad)

    def test_route_mask_expands_for_beam_rows(self) -> None:
        adapter = self.make_adapter()
        router = SafeLayerRouter(adapter)
        hidden = torch.randn(4, 2, 4)
        with router.routing([True, False], token_scope="all") as state:
            routed = router.route_hidden(hidden, state)
        self.assertFalse(torch.equal(routed[:2], hidden[:2]))
        self.assertTrue(torch.equal(routed[2:], hidden[2:]))


if __name__ == "__main__":
    unittest.main()
