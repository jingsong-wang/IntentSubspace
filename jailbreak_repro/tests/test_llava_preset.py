from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from jailbreak_repro.models import HFModelRunner, _configure_public_llava_download, normalize_backend
from jailbreak_repro.run_experiment import parse_args
from jailbreak_repro.run_selected import _model_args


class LlavaPresetTest(unittest.TestCase):
    def test_llava_victim_preset_uses_generic_hf_vlm(self) -> None:
        args = parse_args(
            [
                "--model-preset",
                "llava15_7b",
                "--benchmark",
                "XSTest",
                "--defense",
                "none",
                "--judge-mode",
                "none",
            ]
        )

        self.assertEqual(args.model, "llava-hf/llava-1.5-7b-hf")
        self.assertEqual(args.model_backend, "generic_vlm")
        self.assertEqual(args.model_source, "hf")

    def test_llava_is_available_as_judge_preset(self) -> None:
        args = parse_args(
            [
                "--model-preset",
                "mock",
                "--benchmark",
                "XSTest",
                "--defense",
                "none",
                "--judge-mode",
                "model",
                "--judge-preset",
                "llava15_7b",
            ]
        )

        self.assertEqual(args.judge_model, "llava-hf/llava-1.5-7b-hf")
        self.assertEqual(args.judge_backend, "generic_vlm")
        self.assertEqual(args.judge_model_source, "hf")

    def test_llava_backend_aliases_normalize_to_generic_vlm(self) -> None:
        self.assertEqual(normalize_backend("llava"), "generic_vlm")
        self.assertEqual(normalize_backend("llava15"), "generic_vlm")

    def test_selected_launcher_forwards_llava_as_a_preset(self) -> None:
        self.assertEqual(
            _model_args("model", "llava15_7b", "auto", "auto"),
            ["--model-preset", "llava15_7b"],
        )

    def test_public_llava_disables_xet_downloads(self) -> None:
        previous = os.environ.pop("HF_HUB_DISABLE_XET", None)
        try:
            _configure_public_llava_download("llava-hf/llava-1.5-7b-hf", "hf")
            self.assertEqual(os.environ["HF_HUB_DISABLE_XET"], "1")
        finally:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
            if previous is not None:
                os.environ["HF_HUB_DISABLE_XET"] = previous

    def test_generic_vlm_message_keeps_llava_image_before_text(self) -> None:
        captured = {}

        def process_messages(processor, messages, images):
            captured["messages"] = messages
            captured["images"] = images
            return {"input_ids": "fake"}, "rendered"

        runner = object.__new__(HFModelRunner)
        runner.processor_or_tokenizer = object()
        runner._helpers = {"generic_vlm_process_messages": process_messages}

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.png"
            Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_path)
            batch, rendered = runner._generic_vlm_batch("describe", str(image_path), None)

        content = captured["messages"][0]["content"]
        self.assertEqual(content, [{"type": "image"}, {"type": "text", "text": "describe"}])
        self.assertEqual(len(captured["images"]), 1)
        self.assertEqual(batch, {"input_ids": "fake"})
        self.assertEqual(rendered, "rendered")


if __name__ == "__main__":
    unittest.main()
