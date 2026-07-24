from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jailbreak_repro.cider import (
    CIDER_HARD_REFUSAL,
    CIDER_OFFICIAL_THRESHOLD,
    CiderConfig,
    _DiffusionDenoiser,
    calibrate_cider_threshold,
    cider_detect,
    parse_cider_steps,
    prepare_cider_samples,
    resolve_llava_component,
    verify_paper_llava15_7b,
)
from jailbreak_repro.defenses import run_cider_defense
from jailbreak_repro.models import BaseModelRunner, Generation
from jailbreak_repro.run_experiment import parse_args


class CapturingRunner(BaseModelRunner):
    backend = "capture"
    model_name = "capture-model"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def generate(
        self,
        prompt: str,
        image_path: str | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = 0.9,
    ) -> Generation:
        self.calls.append((prompt, image_path))
        return Generation("model answer", rendered_prompt=prompt, backend=self.backend)


class CiderTest(unittest.TestCase):
    def test_llava_component_resolution_supports_old_and_new_transformers_layouts(self) -> None:
        direct_vision = object()
        nested_vision = object()
        old_layout = SimpleNamespace(vision_tower=direct_vision)
        new_layout = SimpleNamespace(model=SimpleNamespace(vision_tower=nested_vision))

        self.assertIs(resolve_llava_component(old_layout, "vision_tower"), direct_vision)
        self.assertIs(resolve_llava_component(new_layout, "vision_tower"), nested_vision)
        with self.assertRaises(AttributeError):
            resolve_llava_component(SimpleNamespace(), "vision_tower")

    def test_diffusion_denoising_batches_images_and_writes_all_checkpoints(self) -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed in the lightweight test runtime")
        from PIL import Image

        class FakeDiffusion:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def q_sample(self, x_start, t, noise):
                return x_start

            def p_sample(self, model, noisy, timestep, clip_denoised):
                self.batch_sizes.append(len(noisy))
                return {"pred_xstart": noisy}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            items = []
            for index in range(3):
                image_path = root / f"input_{index}.png"
                Image.new("RGB", (20, 30), color=(index * 30, 20, 10)).save(image_path)
                items.append((image_path, root / f"output_{index}"))

            denoiser = object.__new__(_DiffusionDenoiser)
            denoiser.model = object()
            denoiser.diffusion = FakeDiffusion()
            denoiser.device = "cpu"
            denoiser.seed = 0
            outputs = denoiser.checkpoints_many(items, (0, 50), batch_size=2)

            self.assertEqual(denoiser.diffusion.batch_sizes, [2, 1])
            self.assertEqual(len(outputs), 3)
            self.assertTrue(all(path.is_file() for paths in outputs.values() for path in paths))

    def test_preprocessing_orchestration_persists_detection_metadata(self) -> None:
        from PIL import Image

        class FakeDenoiser:
            def __init__(self, config: CiderConfig) -> None:
                pass

            def checkpoints(self, image_path: Path, output_dir: Path, steps: tuple[int, ...]) -> list[Path]:
                output_dir.mkdir(parents=True, exist_ok=True)
                paths = []
                for index, step in enumerate(steps):
                    path = output_dir / f"checkpoint_{step:03d}.jpg"
                    Image.new("RGB", (224, 224), color=(index * 40, 10, 20)).save(path)
                    paths.append(path)
                return paths

            def close(self) -> None:
                pass

        class FakeEncoder:
            def __init__(self, config: CiderConfig) -> None:
                self.paper_encoder_verified = True
                self.verification = {"paper_llava15_7b": True}

            def text_embedding(self, text: str) -> str:
                return text

            def image_embedding(self, image_path: Path) -> float:
                return 0.5 if "000" in image_path.stem else 0.2

            def cosine(self, text_embedding: str, image_embedding: float) -> float:
                return image_embedding

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "input.jpg"
            Image.new("RGB", (40, 30), color=(20, 30, 40)).save(image_path)
            config = CiderConfig(
                source_dir=root / "source",
                artifact_dir=root / "artifacts",
                threshold=-0.1,
                denoise_steps=(0, 50),
            )
            samples = [{"id": "image", "prompt_text": "request", "image_path": str(image_path)}]

            with patch("jailbreak_repro.cider._DiffusionDenoiser", FakeDenoiser), patch(
                "jailbreak_repro.cider._LlavaCrossModalEncoder", FakeEncoder
            ):
                first = prepare_cider_samples(samples, config)
                second = prepare_cider_samples(samples, config)

            self.assertTrue(first[0]["cider_detected"])
            self.assertEqual(first[0]["cider_cosine_similarities"], [0.5, 0.2])
            self.assertAlmostEqual(first[0]["cider_minimum_delta"], -0.3)
            self.assertEqual(first[0]["cider_selected_denoise_step"], 50)
            self.assertEqual(first[0]["cider_encoder_role"], "fixed_auxiliary_detector")
            self.assertFalse(first[0]["cider_uses_victim_encoder"])
            self.assertTrue(first[0]["cider_paper_encoder_verified"])
            self.assertEqual(second[0]["cider_cosine_similarities"], [0.5, 0.2])
            self.assertTrue((root / "artifacts" / "detections.jsonl").is_file())

    def test_official_similarity_decline_rule(self) -> None:
        detected = cider_detect([0.20, 0.21, 0.195], CIDER_OFFICIAL_THRESHOLD)
        passed = cider_detect([0.20, 0.21, 0.198], CIDER_OFFICIAL_THRESHOLD)

        self.assertTrue(detected["detected"])
        self.assertEqual(detected["selected_index"], 2)
        self.assertAlmostEqual(detected["minimum_delta"], -0.005)
        self.assertFalse(passed["detected"])
        self.assertEqual(passed["selected_index"], 0)

    def test_threshold_calibration_uses_clean_pass_percentile(self) -> None:
        threshold = calibrate_cider_threshold([-0.01, -0.005, 0.0, 0.01], pass_rate=0.75)
        self.assertAlmostEqual(threshold, -0.00625)

    def test_detected_sample_returns_official_hard_refusal_without_victim_call(self) -> None:
        runner = CapturingRunner()
        sample = {
            "id": "detected",
            "prompt_text": "request",
            "image_path": "original.jpg",
            "cider_preprocessed": True,
            "cider_detected": True,
            "cider_paper_configuration": True,
        }

        result = run_cider_defense(sample, runner, 32, 0.0, 1.0)

        self.assertEqual(result["response"], CIDER_HARD_REFUSAL)
        self.assertTrue(result["defense_triggered"])
        self.assertEqual(runner.calls, [])

    def test_passed_sample_uses_cider_processed_image(self) -> None:
        runner = CapturingRunner()
        sample = {
            "id": "passed",
            "prompt_text": "request",
            "image_path": "original.jpg",
            "cider_preprocessed": True,
            "cider_detected": False,
            "cider_processed_image_path": "checkpoint_000.jpg",
            "cider_paper_configuration": True,
        }

        result = run_cider_defense(sample, runner, 32, 0.0, 1.0)

        self.assertEqual(result["response"], "model answer")
        self.assertFalse(result["defense_triggered"])
        self.assertEqual(runner.calls, [("request", "checkpoint_000.jpg")])

    def test_no_image_preprocessing_needs_no_detector_weights_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = CiderConfig(source_dir=root / "source", artifact_dir=root / "artifacts")
            samples = [{"id": "text-only", "prompt_text": "hello", "image_path": ""}]

            first = prepare_cider_samples(samples, config)
            second = prepare_cider_samples(samples, config)
            changed = prepare_cider_samples(
                samples,
                CiderConfig(
                    source_dir=root / "source",
                    artifact_dir=root / "artifacts",
                    threshold=-0.25,
                ),
            )

            self.assertEqual(first[0]["cider_skip_reason"], "no_image")
            self.assertFalse(first[0]["cider_detected"])
            self.assertEqual(second[0]["cider_skip_reason"], "no_image")
            self.assertNotEqual(
                first[0]["cider_config_fingerprint"],
                changed[0]["cider_config_fingerprint"],
            )
            self.assertEqual(changed[0]["cider_threshold"], -0.25)

    def test_qwen_and_gemma_presets_accept_cider(self) -> None:
        for preset in ("qwen25vl7b", "gemma3_12b"):
            with self.subTest(preset=preset):
                args = parse_args(
                    [
                        "--model-preset",
                        preset,
                        "--attack",
                        "figstep",
                        "--defense",
                        "cider",
                        "--judge-mode",
                        "none",
                    ]
                )
                self.assertEqual(args.defense, "cider")
                self.assertEqual(args.cider_dtype, "float16")
                self.assertEqual(args.cider_encoder_mode, "paper_llava15")
                self.assertEqual(args.cider_encoder_model, "llava-hf/llava-1.5-7b-hf")

    def test_paper_encoder_signature_rejects_victim_specific_encoder(self) -> None:
        paper_model_class = type("LlavaForConditionalGeneration", (), {})
        paper_model = paper_model_class()
        paper_model.config = SimpleNamespace(
            model_type="llava",
            architectures=["LlavaForConditionalGeneration"],
            text_config=SimpleNamespace(hidden_size=4096, num_hidden_layers=32),
            vision_config=SimpleNamespace(hidden_size=1024, num_hidden_layers=24),
        )
        qwen_model_class = type("Qwen2_5_VLForConditionalGeneration", (), {})
        qwen_model = qwen_model_class()
        qwen_model.config = SimpleNamespace(
            model_type="qwen2_5_vl",
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            text_config=SimpleNamespace(hidden_size=3584, num_hidden_layers=28),
            vision_config=SimpleNamespace(hidden_size=1280, num_hidden_layers=32),
        )

        self.assertTrue(verify_paper_llava15_7b(paper_model)["paper_llava15_7b"])
        self.assertFalse(verify_paper_llava15_7b(qwen_model)["paper_llava15_7b"])

    def test_denoise_step_validation(self) -> None:
        self.assertEqual(parse_cider_steps("0,50,100", "diffusion"), (0, 50, 100))
        self.assertEqual(parse_cider_steps("0,50,100,150,200,250,300,350", "dncnn"), (0, 1))
        with self.assertRaises(ValueError):
            parse_cider_steps("50,100", "diffusion")


if __name__ == "__main__":
    unittest.main()
