from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from jailbreak_repro.defenses import run_defense
from jailbreak_repro.models import BaseModelRunner, Generation, HiddenRepresentation
from jailbreak_repro.representation_detectors import (
    REPRESENTATION_ARTIFACT_FORMAT,
    RepresentationDetector,
    save_representation_artifact,
)
from jailbreak_repro.train_representation_detector import calibrate_rcs_threshold


def base_metadata(method: str, threshold: float = 0.0) -> dict:
    return {
        "method": method,
        "model_id": "victim",
        "model_revision": None,
        "backend": "fake",
        "layer": 2,
        "hidden_dim": 2,
        "pooling": "last",
        "threshold": threshold,
        "protocol": "matched-test",
        "paper_training_protocol": False,
        "core_algorithm_compatible": True,
        "batch_norm_epsilon": 1e-5,
    }


def identity_projection_arrays() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for index in range(3):
        arrays[f"proj_linear_{index}_weight"] = np.eye(2, dtype=np.float32)
        arrays[f"proj_linear_{index}_bias"] = np.zeros(2, dtype=np.float32)
        arrays[f"proj_bn_{index}_weight"] = np.ones(2, dtype=np.float32)
        arrays[f"proj_bn_{index}_bias"] = np.zeros(2, dtype=np.float32)
        arrays[f"proj_bn_{index}_running_mean"] = np.zeros(2, dtype=np.float32)
        arrays[f"proj_bn_{index}_running_var"] = np.ones(2, dtype=np.float32)
    return arrays


def vlmguard_classifier_arrays() -> dict[str, np.ndarray]:
    return {
        "classifier_linear_0_weight": np.eye(2, dtype=np.float32),
        "classifier_linear_0_bias": np.zeros(2, dtype=np.float32),
        "classifier_linear_1_weight": np.eye(2, dtype=np.float32),
        "classifier_linear_1_bias": np.zeros(2, dtype=np.float32),
        "classifier_linear_2_weight": np.array([[1.0, 0.0]], dtype=np.float32),
        "classifier_linear_2_bias": np.zeros(1, dtype=np.float32),
        "svd_center": np.zeros(2, dtype=np.float32),
        "svd_components": np.array([[1.0, 0.0]], dtype=np.float32),
        "svd_singular_values": np.array([2.0], dtype=np.float32),
    }


class FakeRepresentationRunner(BaseModelRunner):
    backend = "fake"
    model_name = "victim"

    def __init__(self) -> None:
        self.generate_calls = 0

    def generate(
        self,
        prompt,
        image_path=None,
        system_prompt=None,
        max_new_tokens=256,
        temperature=0.0,
        top_p=0.9,
    ):
        self.generate_calls += 1
        return Generation("model response", rendered_prompt=prompt, backend=self.backend)

    def extract_hidden(
        self,
        prompt,
        layer,
        image_path=None,
        system_prompt=None,
        pooling="last",
    ):
        vector = np.array([2.0, 0.0]) if "unsafe" in prompt else np.array([-2.0, 0.0])
        return HiddenRepresentation(
            vector=vector,
            rendered_prompt=prompt,
            backend=self.backend,
            layer=layer,
            metadata={"pooling": pooling},
        )


class RepresentationDetectorTest(unittest.TestCase):
    def test_nearside_artifact_round_trip_and_projection_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = save_representation_artifact(
                Path(tmp) / "nearside.npz",
                base_metadata("nearside", threshold=0.5),
                {"direction": np.array([2.0, 0.0], dtype=np.float32)},
            )
            loaded = RepresentationDetector.load(artifact.path)

        self.assertEqual(loaded.metadata["format_version"], REPRESENTATION_ARTIFACT_FORMAT)
        self.assertEqual(loaded.method, "nearside")
        self.assertAlmostEqual(loaded.score_vector([2.0, 1.0])["score"], 2.0)
        self.assertTrue(loaded.score_vector([2.0, 1.0])["detected"])

    def test_kcd_contrastive_score_has_expected_sign(self) -> None:
        arrays = identity_projection_arrays()
        arrays.update(
            {
                "benign_reference": np.array([[1.0, 0.0]], dtype=np.float32),
                "malicious_reference": np.array([[0.0, 1.0]], dtype=np.float32),
            }
        )
        metadata = base_metadata("rcs-kcd")
        metadata["k"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            detector = save_representation_artifact(Path(tmp) / "kcd.npz", metadata, arrays)

        self.assertLess(detector.score_vector([1.0, 0.0])["score"], 0.0)
        self.assertGreater(detector.score_vector([0.0, 1.0])["score"], 0.0)
        batch = detector.score_vectors(np.array([[1.0, 0.0], [0.0, 1.0]]))
        self.assertLess(batch[0], 0.0)
        self.assertGreater(batch[1], 0.0)

    def test_mcd_uses_relative_mahalanobis_distance(self) -> None:
        arrays = identity_projection_arrays()
        arrays.update(
            {
                "benign_means": np.array([[0.0, 0.0]], dtype=np.float32),
                "benign_precisions": np.array([np.eye(2)], dtype=np.float32),
                "malicious_means": np.array([[3.0, 3.0]], dtype=np.float32),
                "malicious_precisions": np.array([np.eye(2)], dtype=np.float32),
            }
        )
        metadata = base_metadata("rcs-mcd")
        metadata["benign_cluster_names"] = ["safe"]
        metadata["malicious_cluster_names"] = ["unsafe"]
        with tempfile.TemporaryDirectory() as tmp:
            detector = save_representation_artifact(Path(tmp) / "mcd.npz", metadata, arrays)

        self.assertLess(detector.score_vector([0.0, 0.0])["score"], 0.0)
        self.assertGreater(detector.score_vector([3.0, 3.0])["score"], 0.0)

    def test_vlmguard_classifier_returns_probability_and_direct_score(self) -> None:
        metadata = base_metadata("vlmguard", threshold=0.75)
        with tempfile.TemporaryDirectory() as tmp:
            detector = save_representation_artifact(
                Path(tmp) / "vlmguard.npz",
                metadata,
                vlmguard_classifier_arrays(),
            )

        result = detector.score_vector([2.0, 0.0])
        batch = detector.score_vectors(np.array([[2.0, 0.0], [-2.0, 0.0]]))
        direct = detector.vlmguard_direct_scores(
            np.array([[2.0, 0.0], [-2.0, 0.0]])
        )
        self.assertAlmostEqual(result["score"], 1.0 / (1.0 + np.exp(-2.0)))
        self.assertAlmostEqual(result["details"]["direct_projection_score"], 8.0)
        self.assertTrue(result["detected"])
        self.assertGreater(batch[0], batch[1])
        np.testing.assert_allclose(direct, np.array([8.0, 8.0]))

    def test_monitor_preserves_response_and_block_skips_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            detector = save_representation_artifact(
                Path(tmp) / "nearside.npz",
                base_metadata("nearside"),
                {"direction": np.array([1.0, 0.0], dtype=np.float32)},
            )
            monitor_runner = FakeRepresentationRunner()
            monitored = run_defense(
                "nearside",
                {"prompt_text": "unsafe query", "image_path": ""},
                monitor_runner,
                max_new_tokens=8,
                temperature=0.0,
                top_p=0.9,
                representation_detector=detector,
                representation_action="monitor",
            )
            block_runner = FakeRepresentationRunner()
            blocked = run_defense(
                "nearside",
                {"prompt_text": "unsafe query", "image_path": ""},
                block_runner,
                max_new_tokens=8,
                temperature=0.0,
                top_p=0.9,
                representation_detector=detector,
                representation_action="block",
            )

        self.assertTrue(monitored["representation_detected"])
        self.assertEqual(monitored["response"], "model response")
        self.assertEqual(monitor_runner.generate_calls, 1)
        self.assertFalse(monitored["paper_claim_compatible"])
        self.assertTrue(blocked["representation_detected"])
        self.assertEqual(block_runner.generate_calls, 0)
        self.assertEqual(blocked["response_source"], "nearside_block_policy")

    def test_rcs_threshold_uses_only_supplied_calibration_scores(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        scores = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float64)
        threshold, metrics = calibrate_rcs_threshold(labels, scores, "rcs-kcd")
        self.assertGreaterEqual(threshold, -1.0)
        self.assertLess(threshold, 1.0)
        self.assertEqual(metrics["tpr"], 1.0)
        self.assertEqual(metrics["fpr"], 0.0)


if __name__ == "__main__":
    unittest.main()
