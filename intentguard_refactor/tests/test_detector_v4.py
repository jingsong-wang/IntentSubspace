from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intentguard.detector import CISRDetector, TinyMLP  # noqa: E402
from intentguard.selective import (  # noqa: E402
    SelectiveRoute,
    SelectiveThresholds,
    choose_selective_thresholds,
    selective_metrics,
)


def v4_detector() -> CISRDetector:
    rank = 3
    return CISRDetector(
        path=Path("detector_v4.npz"),
        model_id="test-model",
        model_alias="test-v4",
        layer=2,
        rank=rank,
        pooling="last",
        basis=np.eye(rank, 4),
        center=np.zeros(4),
        residual_center=np.zeros(4),
        feature_mean=np.zeros(rank),
        feature_std=np.ones(rank),
        role_categories=[],
        network=TinyMLP(
            np.array([[2.0], [0.0], [0.0]]),
            np.zeros(1),
            np.ones((1, 1)) * 8.0,
            np.zeros(1),
        ),
        threshold=0.8,
        anchor_prompt="",
        uses_anchor=False,
        format_version="CISR_v4_detector_v1",
        feature_mode="raw_rank3",
        deployment_constraints_met=True,
        safe_threshold=0.2,
        danger_threshold=0.8,
        safe_route_enabled=True,
        danger_route_enabled=True,
        confident_safe_error_upper_bound=0.02,
        confident_dangerous_error_upper_bound=0.02,
    )


class CISRV4SelectiveTest(unittest.TestCase):
    def test_selective_calibration_finds_two_certified_regions(self) -> None:
        labels = np.array([0] * 300 + [1] * 300)
        probabilities = np.concatenate(
            [np.linspace(0.01, 0.20, 300), np.linspace(0.80, 0.99, 300)]
        )
        groups = np.array(["text"] * 150 + ["image"] * 150 + ["text"] * 150 + ["image"] * 150)

        thresholds, selection = choose_selective_thresholds(
            labels,
            probabilities,
            maximum_safe_error=0.02,
            maximum_danger_error=0.02,
            maximum_harmful_escape=0.05,
            maximum_benign_hard_refusal=0.05,
            groups=groups,
            minimum_group_selected=20,
        )

        self.assertTrue(selection["deployment_constraints_met"])
        self.assertTrue(thresholds.safe_enabled)
        self.assertTrue(thresholds.danger_enabled)
        self.assertLess(thresholds.safe_max, thresholds.danger_min)
        metrics = selection["selective_metrics"]
        self.assertEqual(metrics["confident_safe_error_count"], 0)
        self.assertEqual(metrics["confident_dangerous_error_count"], 0)
        self.assertEqual(metrics["review_count"], 0)

    def test_metrics_separate_route_conditional_and_class_conditional_errors(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.9, 0.1, 0.5, 0.9])
        metrics = selective_metrics(
            labels,
            probabilities,
            SelectiveThresholds(safe_max=0.2, danger_min=0.8),
        )

        self.assertEqual(metrics["confident_safe_count"], 3)
        self.assertAlmostEqual(metrics["confident_safe_error_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["harmful_unsafe_escape_rate"], 1 / 3)
        self.assertEqual(metrics["confident_dangerous_count"], 2)
        self.assertAlmostEqual(metrics["confident_dangerous_error_rate"], 1 / 2)
        self.assertAlmostEqual(metrics["benign_hard_refusal_rate"], 1 / 3)
        self.assertEqual(metrics["review_count"], 1)

    def test_detector_emits_all_three_routes(self) -> None:
        detector = v4_detector()
        safe = detector.score_hidden(np.array([-2.0, 0.0, 0.0, 0.0]), image_role="none")
        review = detector.score_hidden(np.zeros(4), image_role="none")
        danger = detector.score_hidden(np.array([2.0, 0.0, 0.0, 0.0]), image_role="none")

        self.assertEqual(safe["route"], SelectiveRoute.CONFIDENT_SAFE.value)
        self.assertFalse(safe["requires_intervention"])
        self.assertEqual(review["route"], SelectiveRoute.REVIEW.value)
        self.assertTrue(review["requires_intervention"])
        self.assertEqual(danger["route"], SelectiveRoute.CONFIDENT_DANGEROUS.value)
        self.assertTrue(danger["detected"])


if __name__ == "__main__":
    unittest.main()
