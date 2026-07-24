from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from intentguard_refactor.intentguard.detector import (
    CISRDetector,
    CISRDetectorBundle,
    TinyMLP,
)
from jailbreak_repro.defenses import run_cisr_defense
from jailbreak_repro.models import BaseModelRunner, Generation, HiddenRepresentation
from jailbreak_repro.run_experiment import summarize_cisr4_routes


class HiddenRunner(BaseModelRunner):
    backend = "test"
    model_name = "test-model"

    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector
        self.generate_calls = 0
        self.hidden_calls = 0
        self.hidden_poolings: list[str] = []

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
        self.hidden_poolings.append(pooling)
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


def v3_detector(first_weight: float, threshold: float = 0.5) -> CISRDetector:
    rank = 3
    weight_1 = np.zeros((rank, 1), dtype=float)
    weight_1[0, 0] = first_weight
    return CISRDetector(
        path=Path("detector_v3.npz"),
        model_id="test-model",
        model_alias="test-v3",
        layer=2,
        rank=rank,
        pooling="last",
        basis=np.eye(rank, 4),
        center=np.zeros(4),
        residual_center=np.zeros(4),
        feature_mean=np.zeros(rank),
        feature_std=np.ones(rank),
        role_categories=[],
        network=TinyMLP(weight_1, np.zeros(1), np.ones((1, 1)) * 8.0, np.zeros(1)),
        threshold=threshold,
        anchor_prompt="",
        uses_anchor=False,
        format_version="CISR_v3_detector_v1",
        feature_mode="raw_rank3",
        deployment_constraints_met=True,
        hard_benign_target_fpr=0.10,
    )


def v4_detector(first_weight: float = 2.0) -> CISRDetector:
    source = v3_detector(first_weight=first_weight, threshold=0.8)
    source.path = Path("detector_v4.npz")
    source.model_alias = "test-v4"
    source.format_version = "CISR_v4_detector_v1"
    source.safe_threshold = 0.2
    source.danger_threshold = 0.8
    source.safe_route_enabled = True
    source.danger_route_enabled = True
    source.confident_safe_error_upper_bound = 0.02
    source.confident_dangerous_error_upper_bound = 0.02
    return source


class FakeSafeLayerRuntime:
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        prompt: str,
        *,
        image_path: str | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float | None,
    ) -> Generation:
        del image_path, max_new_tokens, temperature, top_p
        self.calls += 1
        return Generation("safe-layer reviewed response", rendered_prompt=prompt, backend="test")


class CISRDefenseTest(unittest.TestCase):
    def test_v4_bundle_routes_text_and_images_to_different_poolings(self) -> None:
        text_detector = v4_detector()
        text_detector.pooling = "non_image_mean"
        multimodal_detector = v4_detector()
        multimodal_detector.pooling = "image_mean"
        bundle = CISRDetectorBundle(
            path=Path("bundle.json"),
            model_id="test-model",
            model_alias="test-bundle",
            text_detector=text_detector,
            multimodal_detector=multimodal_detector,
        )

        text_runner = HiddenRunner(np.array([2.0, 0.0, 0.0, 0.0]))
        text_result = run_cisr_defense(
            sample={"prompt_text": "text", "label": 1},
            runner=text_runner,
            detector=bundle,
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.9,
            version="cisr4",
            review_action="monitor",
        )
        self.assertEqual(text_result["cisr_detector_branch"], "text")
        self.assertEqual(text_result["cisr_pooling"], "non_image_mean")
        self.assertEqual(text_runner.hidden_poolings, ["non_image_mean"])

        image_runner = HiddenRunner(np.array([2.0, 0.0, 0.0, 0.0]))
        image_result = run_cisr_defense(
            sample={"prompt_text": "image", "image_path": "image.png", "label": 1},
            runner=image_runner,
            detector=bundle,
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.9,
            version="cisr4",
            review_action="monitor",
        )
        self.assertEqual(image_result["cisr_detector_branch"], "multimodal")
        self.assertEqual(image_result["cisr_pooling"], "image_mean")
        self.assertEqual(image_runner.hidden_poolings, ["image_mean"])

    def test_v4_routes_safe_review_and_danger_to_distinct_actions(self) -> None:
        safe_runner = HiddenRunner(np.array([-2.0, 0.0, 0.0, 0.0]))
        safe_layer = FakeSafeLayerRuntime()
        safe = run_cisr_defense(
            sample={"prompt_text": "safe", "label": 0},
            runner=safe_runner,
            detector=v4_detector(),
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.9,
            version="cisr4",
            safe_layer_runtime=safe_layer,
        )
        self.assertEqual(safe["cisr_route"], "confident_safe")
        self.assertEqual(safe["response_source"], "model")
        self.assertEqual(safe_runner.generate_calls, 1)
        self.assertEqual(safe_layer.calls, 0)

        review_runner = HiddenRunner(np.zeros(4))
        review_layer = FakeSafeLayerRuntime()
        review = run_cisr_defense(
            sample={"prompt_text": "review", "label": 0},
            runner=review_runner,
            detector=v4_detector(),
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.9,
            version="cisr4",
            safe_layer_runtime=review_layer,
        )
        self.assertEqual(review["cisr_route"], "review")
        self.assertEqual(review["response_source"], "cisr4_safe_layer")
        self.assertTrue(review["defense_triggered"])
        self.assertEqual(review_layer.calls, 1)
        self.assertEqual(review_runner.generate_calls, 0)

        danger_runner = HiddenRunner(np.array([2.0, 0.0, 0.0, 0.0]))
        danger_layer = FakeSafeLayerRuntime()
        danger = run_cisr_defense(
            sample={"prompt_text": "danger", "label": 1},
            runner=danger_runner,
            detector=v4_detector(),
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.9,
            version="cisr4",
            safe_layer_runtime=danger_layer,
        )
        self.assertEqual(danger["cisr_route"], "confident_dangerous")
        self.assertEqual(
            danger["response_source"], "cisr4_confident_dangerous_hard_refusal"
        )
        self.assertEqual(danger_runner.generate_calls, 0)
        self.assertEqual(danger_layer.calls, 0)

    def test_v4_summary_reports_the_two_unacceptable_errors(self) -> None:
        rows = [
            {"label": 1, "cisr_route": "confident_safe", "cisr_coverage_confidence": 0.95},
            {"label": 0, "cisr_route": "confident_safe", "cisr_coverage_confidence": 0.95},
            {"label": 0, "cisr_route": "confident_dangerous", "cisr_coverage_confidence": 0.95},
            {"label": 1, "cisr_route": "confident_dangerous", "cisr_coverage_confidence": 0.95},
            {"label": 1, "cisr_route": "review", "cisr_coverage_confidence": 0.95},
        ]
        summary = summarize_cisr4_routes(rows)
        self.assertEqual(summary["confident_safe_error_count"], 1)
        self.assertEqual(summary["confident_safe_error_rate"], 0.5)
        self.assertEqual(summary["confident_dangerous_error_count"], 1)
        self.assertEqual(summary["confident_dangerous_error_rate"], 0.5)
        self.assertEqual(summary["harmful_unsafe_escape_rate"], 1 / 3)
        self.assertEqual(summary["benign_hard_refusal_rate"], 1 / 2)

    def test_v4_summary_does_not_invent_missing_class_error_rates(self) -> None:
        harmful_only = summarize_cisr4_routes(
            [
                {"label": 1, "cisr_route": "confident_dangerous"},
                {"label": 1, "cisr_route": "review"},
            ]
        )
        benign_only = summarize_cisr4_routes(
            [
                {"label": 0, "cisr_route": "confident_safe"},
                {"label": 0, "cisr_route": "confident_dangerous"},
            ]
        )

        self.assertIsNone(harmful_only["confident_dangerous_error_rate"])
        self.assertFalse(harmful_only["confident_dangerous_error_evaluable"])
        self.assertIsNone(benign_only["confident_safe_error_rate"])
        self.assertFalse(benign_only["confident_safe_error_evaluable"])

    def test_v3_raw_rank3_defense_uses_one_hidden_forward(self) -> None:
        runner = HiddenRunner(np.array([2.0, 0.0, 0.0, 0.0]))
        result = run_cisr_defense(
            sample={"prompt_text": "test", "image_path": "image.png", "image_role": "unseen"},
            runner=runner,
            detector=v3_detector(first_weight=2.0),
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.9,
        )

        self.assertTrue(result["cisr_detected"])
        self.assertEqual(result["defense"], "cisr3")
        self.assertEqual(result["cisr_version"], "cisr3")
        self.assertEqual(result["cisr_feature_mode"], "raw_rank3")
        self.assertTrue(result["cisr_deployment_constraints_met"])
        self.assertEqual(result["cisr_residual_coordinates"], [])
        self.assertEqual(runner.hidden_calls, 1)
        self.assertEqual(runner.generate_calls, 0)

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
        self.assertEqual(result["defense"], "cisr2")
        self.assertEqual(result["cisr_version"], "cisr2")
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
