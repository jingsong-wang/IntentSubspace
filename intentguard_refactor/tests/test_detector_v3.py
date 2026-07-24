from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intentguard.detector import CISRDetector, TinyMLP, train_tiny_mlp
from train_detector_v3 import choose_joint_threshold


class CISRV3DetectorTest(unittest.TestCase):
    def test_joint_threshold_certifies_separable_data(self) -> None:
        labels = np.array([0] * 200 + [1] * 200)
        probabilities = np.concatenate(
            [np.linspace(0.01, 0.35, 200), np.linspace(0.65, 0.99, 200)]
        )
        hard_benign = labels == 0

        threshold, selection = choose_joint_threshold(
            labels,
            probabilities,
            target_tpr=0.90,
            target_fpr=0.10,
            target_hard_benign_fpr=0.10,
            confidence=0.95,
            hard_benign_mask=hard_benign,
        )

        self.assertGreater(threshold, 0.35)
        self.assertLess(threshold, 0.65)
        self.assertTrue(selection["joint_confidence_constraints_met"])
        self.assertLessEqual(selection["fpr_upper_confidence_bound"], 0.10)
        self.assertLessEqual(
            selection["hard_benign"]["fpr_upper_confidence_bound"], 0.10
        )

    def test_joint_threshold_reports_infeasible_operating_point(self) -> None:
        labels = np.array([0] * 100 + [1] * 100)
        probabilities = np.concatenate(
            [np.linspace(0.20, 0.90, 100), np.linspace(0.25, 0.95, 100)]
        )
        _, selection = choose_joint_threshold(
            labels,
            probabilities,
            target_tpr=0.95,
            target_fpr=0.05,
            target_hard_benign_fpr=0.05,
            confidence=0.95,
            hard_benign_mask=labels == 0,
        )
        self.assertFalse(selection["joint_confidence_constraints_met"])
        self.assertGreater(selection["normalized_constraint_violation"], 0.0)

    def test_consistency_regularizer_accepts_same_label_view_groups(self) -> None:
        features = np.array(
            [
                [-2.0, 0.0, 0.0],
                [-1.7, 0.2, 0.0],
                [-1.8, -0.2, 0.1],
                [-1.6, 0.1, -0.1],
                [1.6, 0.0, 0.0],
                [1.9, 0.2, 0.0],
                [1.7, -0.2, 0.1],
                [2.0, 0.1, -0.1],
            ]
        )
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        groups = np.array(["n0", "n0", "n1", "n1", "p0", "p0", "p1", "p1"])
        model, report = train_tiny_mlp(
            features,
            labels,
            consistency_groups=groups,
            consistency_weight=0.25,
            hidden_dim=3,
            epochs=30,
            seed=4,
        )
        self.assertEqual(report["consistency_group_count"], 4)
        self.assertEqual(model.predict_proba(features).shape, (8,))

    def test_raw_rank3_artifact_ignores_anchor_and_role_features(self) -> None:
        rank = 3
        detector = CISRDetector(
            path=Path("detector.npz"),
            model_id="test-model",
            model_alias="test",
            layer=4,
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
                np.ones((1, 1)) * 5.0,
                np.zeros(1),
            ),
            threshold=0.5,
            anchor_prompt="",
            uses_anchor=False,
            format_version="CISR_v3_detector_v1",
            feature_mode="raw_rank3",
            deployment_constraints_met=True,
        )
        result = detector.score_hidden(
            np.array([2.0, 0.0, 0.0, 0.0]), image_role="previously_unseen_role"
        )
        self.assertTrue(result["detected"])
        self.assertEqual(result["feature_mode"], "raw_rank3")
        self.assertEqual(result["residual_coordinates"], [])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "detector.npz"
            np.savez_compressed(
                path,
                format_version=np.array([detector.format_version]),
                feature_mode=np.array([detector.feature_mode]),
                model_id=np.array([detector.model_id]),
                model_alias=np.array([detector.model_alias]),
                layer=np.array([detector.layer]),
                rank=np.array([detector.rank]),
                pooling=np.array([detector.pooling]),
                basis=detector.basis,
                center=detector.center,
                residual_center=detector.residual_center,
                feature_mean=detector.feature_mean,
                feature_std=detector.feature_std,
                role_categories=np.array([], dtype=str),
                weight_1=detector.network.weight_1,
                bias_1=detector.network.bias_1,
                weight_2=detector.network.weight_2,
                bias_2=detector.network.bias_2,
                threshold=np.array([detector.threshold]),
                anchor_prompt=np.array([""]),
                uses_anchor=np.array([False]),
                deployment_constraints_met=np.array([True]),
            )
            loaded = CISRDetector.load(path)
        self.assertEqual(loaded.feature_mode, "raw_rank3")
        self.assertEqual(loaded.network.weight_1.shape[0], rank)


if __name__ == "__main__":
    unittest.main()
