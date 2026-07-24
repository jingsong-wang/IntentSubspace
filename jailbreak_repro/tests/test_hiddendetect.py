from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from jailbreak_repro.defenses import run_defense
from jailbreak_repro.hiddendetect import (
    HIDDENDETECT_PROFILE_FORMAT,
    HIDDENDETECT_REFUSAL_TOKENS,
    HIDDENDETECT_SCORE_FORMAT,
    HiddenDetectProfile,
    binary_auprc,
    binary_auroc,
    build_hiddendetect_profile,
    select_safety_aware_layers,
    sample_safety_label,
    trapezoid_score,
)
from jailbreak_repro.io_utils import write_json
from jailbreak_repro.models import BaseModelRunner, Generation, HFModelRunner, RefusalLayerScores
from jailbreak_repro.models import _module_device


class FakeHiddenRunner(BaseModelRunner):
    backend = "fake"
    model_name = "victim"

    def __init__(self) -> None:
        self.generate_calls = 0

    def generate(self, prompt, image_path=None, system_prompt=None, max_new_tokens=256, temperature=0.0, top_p=0.9):
        self.generate_calls += 1
        return Generation("model response", rendered_prompt=prompt, backend=self.backend)

    def score_refusal_layers(self, prompt, refusal_tokens, image_path=None, system_prompt=None):
        unsafe = "unsafe" in prompt
        scores = [0.0, 0.5, 0.3, 0.1] if unsafe else [0.0, 0.1, 0.0, 0.0]
        return RefusalLayerScores(
            scores=scores,
            refusal_token_ids=list(range(len(refusal_tokens))),
            rendered_prompt=prompt,
            backend=self.backend,
        )


def profile(threshold: float = 0.3) -> HiddenDetectProfile:
    return HiddenDetectProfile(
        format_version=HIDDENDETECT_PROFILE_FORMAT,
        score_format=HIDDENDETECT_SCORE_FORMAT,
        model_id="victim",
        backend="fake",
        refusal_tokens=HIDDENDETECT_REFUSAL_TOKENS,
        refusal_token_ids=list(range(len(HIDDENDETECT_REFUSAL_TOKENS))),
        layer_count=4,
        safety_aware_layers=[1, 2],
        safe_mean=[0.0, 0.1, 0.0, 0.0],
        unsafe_mean=[0.0, 0.5, 0.3, 0.1],
        fdv=[0.0, 0.4, 0.3, 0.1],
        threshold=threshold,
        threshold_method="test",
        threshold_metrics={},
        calibration_safe_scores=[0.05],
        calibration_unsafe_scores=[0.4],
        fewshot_sha1="test",
        profile_fingerprint="fingerprint",
    )


class HiddenDetectTest(unittest.TestCase):
    def test_sample_safety_label_accepts_attack_label_field(self) -> None:
        self.assertEqual(sample_safety_label({"label": 1}), 1)
        self.assertEqual(sample_safety_label({"label": 0}), 0)
        self.assertEqual(
            sample_safety_label({"xstest_label": "safe", "label": 1}),
            0,
        )

    def test_component_resolution_supports_qwen_gemma_and_llava_layouts(self) -> None:
        tokenizer = object()
        head = object()
        norm = object()
        layouts = [
            SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(norm=norm))),
            SimpleNamespace(
                model=SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(norm=norm)))
            ),
            SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(norm=norm))),
        ]
        for model in layouts:
            model.get_output_embeddings = lambda: head
            runner = object.__new__(HFModelRunner)
            runner.processor_or_tokenizer = SimpleNamespace(tokenizer=tokenizer)
            runner.model = model
            self.assertEqual(runner._hidden_detect_components(), (tokenizer, norm, head))

    def test_offloaded_module_uses_accelerate_execution_device_instead_of_meta(self) -> None:
        module = SimpleNamespace(
            _hf_hook=SimpleNamespace(execution_device="cuda:0"),
            parameters=lambda: iter([SimpleNamespace(device="meta")]),
        )
        self.assertEqual(_module_device(module, "cpu"), "cuda:0")

    def test_meta_module_without_hook_uses_fallback_device(self) -> None:
        module = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="meta")]))
        self.assertEqual(_module_device(module, "cpu"), "cpu")

    def test_trapezoid_matches_released_numpy_trapz(self) -> None:
        self.assertAlmostEqual(trapezoid_score([1.0, 2.0, 4.0]), 4.5)
        self.assertAlmostEqual(trapezoid_score([2.0]), 0.0)

    def test_layer_selection_uses_final_fdv_as_baseline(self) -> None:
        self.assertEqual(select_safety_aware_layers([0.0, 0.4, 0.3, 0.1]), [1, 2])

    def test_profile_is_built_from_the_current_victim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fewshot = root / "fewshot.json"
            write_json(
                fewshot,
                [
                    {"txt": "safe one", "img": None, "toxicity": 0},
                    {"txt": "safe two", "img": None, "toxicity": 0},
                    {"txt": "unsafe one", "img": None, "toxicity": 1},
                    {"txt": "unsafe two", "img": None, "toxicity": 1},
                ],
            )
            runner = FakeHiddenRunner()
            built = build_hiddendetect_profile(
                runner,
                root / "profile.json",
                source_dir=root,
                fewshot_file=fewshot,
            )

        self.assertEqual(built.model_id, "victim")
        self.assertEqual(built.safety_aware_layers, [1, 2])
        self.assertAlmostEqual(built.threshold, 0.225)
        self.assertEqual(built.threshold_method, "balanced_accuracy_on_official_fewshot_platform_calibration")

    def test_monitor_scores_but_preserves_model_response(self) -> None:
        runner = FakeHiddenRunner()
        row = run_defense(
            "hiddendetect",
            {"prompt_text": "unsafe query", "image_path": ""},
            runner,
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.9,
            hiddendetect_profile=profile(),
            hiddendetect_action="monitor",
        )
        self.assertTrue(row["hiddendetect_detected"])
        self.assertEqual(row["response"], "model response")
        self.assertEqual(runner.generate_calls, 1)
        self.assertTrue(row["paper_claim_compatible"])

    def test_block_is_explicit_non_paper_intervention(self) -> None:
        runner = FakeHiddenRunner()
        row = run_defense(
            "hiddendetect",
            {"prompt_text": "unsafe query", "image_path": ""},
            runner,
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.9,
            hiddendetect_profile=profile(),
            hiddendetect_action="block",
        )
        self.assertTrue(row["hiddendetect_detected"])
        self.assertEqual(row["response_source"], "hiddendetect_block_policy")
        self.assertEqual(runner.generate_calls, 0)
        self.assertFalse(row["paper_claim_compatible"])

    def test_binary_auroc_handles_ties(self) -> None:
        self.assertEqual(binary_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(binary_auroc([0, 1], [0.5, 0.5]), 0.5)

    def test_binary_auprc_matches_trapezoidal_precision_recall_auc(self) -> None:
        self.assertEqual(binary_auprc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertAlmostEqual(binary_auprc([0, 1], [0.5, 0.5]), 0.75)
        self.assertIsNone(binary_auprc([0, 0], [0.1, 0.2]))
        self.assertIsNone(binary_auprc([1, 1], [0.1, 0.2]))


if __name__ == "__main__":
    unittest.main()
