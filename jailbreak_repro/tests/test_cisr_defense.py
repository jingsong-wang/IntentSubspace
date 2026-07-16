from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from intentguard_refactor.intentguard.detector import CISRDetector, TinyMLP
from jailbreak_repro.defenses import run_cisr_defense
from jailbreak_repro.models import BaseModelRunner, Generation, HiddenRepresentation


class HiddenRunner(BaseModelRunner):
    backend = "test"
    model_name = "test-model"

    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector
        self.generate_calls = 0
        self.hidden_calls = 0

    def generate(
        self,
        prompt: str,
        image_path: str | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = 0.9,
    ) -> Generation:
        self.generate_calls += 1
        return Generation("ordinary model response", rendered_prompt=prompt, backend=self.backend)

    def extract_hidden(
        self,
        prompt: str,
        layer: int,
        image_path: str | None = None,
        system_prompt: str | None = None,
        pooling: str = "last",
    ) -> HiddenRepresentation:
        self.hidden_calls += 1
        return HiddenRepresentation(self.vector, prompt, self.backend, layer)


def detector(first_weight: float, threshold: float = 0.5) -> CISRDetector:
    rank = 3
    roles = ["none", "__other__"]
    feature_dim = rank * 2 + 1 + len(roles)
    weight_1 = np.zeros((feature_dim, 1), dtype=float)
    weight_1[0, 0] = first_weight
    return CISRDetector(
        path=Path("detector.npz"),
        model_id="test-model",
        model_alias="test",
        layer=2,
        rank=rank,
        pooling="last",
        basis=np.eye(rank, 4),
        center=np.zeros(4),
        residual_center=np.zeros(4),
        feature_mean=np.zeros(feature_dim),
        feature_std=np.ones(feature_dim),
        role_categories=roles,
        network=TinyMLP(weight_1, np.zeros(1), np.ones((1, 1)) * 8.0, np.zeros(1)),
        threshold=threshold,
        anchor_prompt="describe image safely",
        uses_anchor=False,
    )


class CISRDefenseTest(unittest.TestCase):
    def test_triggered_detection_skips_victim_generation(self) -> None:
        runner = HiddenRunner(np.array([2.0, 0.0, 0.0, 0.0]))
        result = run_cisr_defense(
            sample={"prompt_text": "test", "image_role": "none"},
            runner=runner,
            detector=detector(first_weight=2.0),
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.9,
        )

        self.assertTrue(result["cisr_detected"])
        self.assertTrue(result["defense_triggered"])
        self.assertEqual(runner.generate_calls, 0)
        self.assertEqual(len(result["cisr_coordinates"]), 3)

    def test_safe_detection_runs_victim_generation(self) -> None:
        runner = HiddenRunner(np.array([-2.0, 0.0, 0.0, 0.0]))
        result = run_cisr_defense(
            sample={"prompt_text": "test", "image_role": "none"},
            runner=runner,
            detector=detector(first_weight=2.0),
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.9,
        )

        self.assertFalse(result["cisr_detected"])
        self.assertFalse(result["defense_triggered"])
        self.assertEqual(result["response"], "ordinary model response")
        self.assertEqual(runner.generate_calls, 1)

    def test_model_mismatch_is_marked_incompatible(self) -> None:
        runner = HiddenRunner(np.array([-2.0, 0.0, 0.0, 0.0]))
        mismatched = detector(first_weight=2.0)
        mismatched.model_id = "different-model"

        result = run_cisr_defense(
            sample={"prompt_text": "test", "image_role": "none"},
            runner=runner,
            detector=mismatched,
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.9,
        )

        self.assertFalse(result["cisr_model_match"])
        self.assertFalse(result["paper_claim_compatible"])

    def test_detector_artifact_round_trip(self) -> None:
        source = detector(first_weight=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "detector.npz"
            np.savez_compressed(
                path,
                model_id=np.array([source.model_id]),
                model_alias=np.array([source.model_alias]),
                layer=np.array([source.layer]),
                rank=np.array([source.rank]),
                pooling=np.array([source.pooling]),
                basis=source.basis,
                center=source.center,
                residual_center=source.residual_center,
                feature_mean=source.feature_mean,
                feature_std=source.feature_std,
                role_categories=np.array(source.role_categories),
                weight_1=source.network.weight_1,
                bias_1=source.network.bias_1,
                weight_2=source.network.weight_2,
                bias_2=source.network.bias_2,
                threshold=np.array([source.threshold]),
                anchor_prompt=np.array([source.anchor_prompt]),
                uses_anchor=np.array([source.uses_anchor]),
            )
            loaded = CISRDetector.load(path)

        self.assertEqual(loaded.rank, 3)
        self.assertEqual(loaded.layer, 2)
        self.assertEqual(loaded.network.weight_1.shape[0], 9)


if __name__ == "__main__":
    unittest.main()
