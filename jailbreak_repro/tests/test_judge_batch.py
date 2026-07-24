from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image

from jailbreak_repro.judges import ModelJudge, XSTestModelJudge
from jailbreak_repro.models import BaseModelRunner, Generation, HFModelRunner
from jailbreak_repro.run_experiment import judge_config, parse_args


class BatchedRunner(BaseModelRunner):
    backend = "batched"
    model_name = "batched-judge"

    def __init__(self, output_batches: list[list[str]]) -> None:
        self.output_batches = output_batches
        self.calls: list[list[str]] = []

    def generate(self, *args: Any, **kwargs: Any) -> Generation:
        raise AssertionError("judge should use generate_batch")

    def generate_batch(
        self,
        prompts: list[str],
        image_paths: list[str | None] | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = 0.9,
    ) -> list[Generation]:
        call_index = len(self.calls)
        self.calls.append(prompts)
        outputs = self.output_batches[call_index]
        if len(outputs) != len(prompts):
            raise AssertionError("test output batch does not match prompt batch")
        return [
            Generation(
                text,
                rendered_prompt=prompt,
                backend=self.backend,
                metadata={"batched": True, "batch_size": len(prompts), "batch_index": index},
            )
            for index, (prompt, text) in enumerate(zip(prompts, outputs))
        ]


class GroupedHFRunner(HFModelRunner):
    backend = "generic_vlm"
    model_name = "grouped-hf"

    def __init__(self) -> None:
        self.group_calls: list[tuple[list[str], list[str | None]]] = []

    def _generate_batch_group(
        self,
        prompts: list[str],
        image_paths: list[str | None],
        system_prompt: str | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float | None,
    ) -> list[Generation]:
        self.group_calls.append((list(prompts), list(image_paths)))
        return [
            Generation(
                text=f"output:{prompt}",
                rendered_prompt=prompt,
                backend=self.backend,
                metadata={"batched": True, "batch_size": len(prompts)},
            )
            for prompt in prompts
        ]


class FakeTokenizer:
    padding_side = "right"
    pad_token_id = 0
    eos_token = "</s>"


class Gemma3ProcessorFake:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.image_batch_shape: list[int] | None = None
        self.padding_side_during_call: str | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        return f"rendered:{messages[-1]['content'][-1]['text']}"

    def __call__(
        self,
        text: list[str],
        images: list[list[Image.Image]],
        padding: bool,
        return_tensors: str,
    ) -> dict[str, Any]:
        self.image_batch_shape = [len(sample) for sample in images]
        self.padding_side_during_call = self.tokenizer.padding_side
        return {"input_ids": [[1], [1]], "attention_mask": [[1], [1]]}


class QwenProcessorFake:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.call: dict[str, Any] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        return f"rendered:{messages[-1]['content'][-1]['text']}"

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.call = kwargs
        return {"input_ids": [[1], [1]], "attention_mask": [[1], [1]]}


class JudgeBatchTest(unittest.TestCase):
    @staticmethod
    def _judge_args(task: str, prompt_file: Path | None = None):
        argv = [
            "--model-preset",
            "mock",
            "--benchmark",
            "HADES" if task == "asr" else "XSTest",
            "--defense",
            "none",
            "--judge-mode",
            "model",
            "--judge-model",
            "mock",
            "--judge-backend",
            "mock",
        ]
        if prompt_file is not None:
            argv.extend(["--judge-prompt-file", str(prompt_file)])
        args = parse_args(argv)
        args.effective_judge_task = task
        return args

    def test_asr_prompt_content_changes_judge_config_hash_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = Path(tmp) / "judge_prompt.py"
            prompt_file.write_text('JUDGE_SYSTEM_PROMPT = "prompt one"\n', encoding="utf-8")
            first = judge_config(self._judge_args("asr", prompt_file))
            repeated = judge_config(self._judge_args("asr", prompt_file))
            prompt_file.write_text('JUDGE_SYSTEM_PROMPT = "prompt two"\n', encoding="utf-8")
            changed = judge_config(self._judge_args("asr", prompt_file))

        self.assertEqual(first, repeated)
        self.assertNotEqual(first["judge_system_prompt_sha1"], changed["judge_system_prompt_sha1"])

    def test_xstest_prompt_content_changes_judge_config_hash_input(self) -> None:
        args = self._judge_args("xstest")
        first = judge_config(args)
        repeated = judge_config(args)
        with patch(
            "jailbreak_repro.run_experiment.XSTEST_JUDGE_PROMPT_TEMPLATE",
            "changed XSTest prompt {question} {response}",
        ):
            changed = judge_config(args)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first["xstest_judge_prompt_sha1"], changed["xstest_judge_prompt_sha1"])

    def test_asr_judge_uses_one_generation_batch(self) -> None:
        outputs = [
            json.dumps({"reasoning": "safe", "is_mismatch": False, "harmfulness_score": 1}),
            json.dumps({"reasoning": "harmful", "is_mismatch": False, "harmfulness_score": 4}),
        ]
        runner = BatchedRunner([outputs])
        judge = ModelJudge(runner)
        rows = [
            {"id": "safe", "prompt": "request one", "response": "refusal"},
            {"id": "harmful", "prompt": "request two", "response": "answer"},
        ]

        results = judge.judge_batch(rows)

        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(len(runner.calls[0]), 2)
        self.assertEqual([row["judge_harmfulness_score"] for row in results], [1, 4])
        self.assertTrue(all(row["judge_generation_meta"]["batched"] for row in results))

    def test_xstest_retries_only_parse_errors_as_a_smaller_batch(self) -> None:
        runner = BatchedRunner(
            [
                ["unclear free-form classification", "2_full_refusal"],
                ["1_full_compliance"],
            ]
        )
        judge = XSTestModelJudge(runner)
        rows = [
            {"id": "retry", "prompt": "question one", "response": "answer", "xstest_label": "safe"},
            {"id": "valid", "prompt": "question two", "response": "refusal", "xstest_label": "unsafe"},
        ]

        results = judge.judge_batch(rows)

        self.assertEqual([len(call) for call in runner.calls], [2, 1])
        self.assertEqual(results[0]["xstest_class"], "1_full_compliance")
        self.assertEqual(results[0]["judge_retry_count"], 1)
        self.assertEqual(results[1]["xstest_class"], "2_full_refusal")
        self.assertEqual(results[1]["judge_retry_count"], 0)

    def test_hf_runner_batches_image_and_text_groups_and_restores_order(self) -> None:
        runner = GroupedHFRunner()
        prompts = ["image-zero", "text-one", "image-two", "text-three"]
        image_paths = ["zero.png", None, "two.png", None]

        results = runner.generate_batch(prompts, image_paths=image_paths)

        self.assertEqual(
            runner.group_calls,
            [
                (["text-one", "text-three"], [None, None]),
                (["image-zero", "image-two"], ["zero.png", "two.png"]),
            ],
        )
        self.assertEqual([result.text for result in results], [f"output:{prompt}" for prompt in prompts])
        self.assertEqual([result.metadata["batch_index"] for result in results], [0, 1, 2, 3])
        self.assertTrue(all(result.metadata["requested_batch_size"] == 4 for result in results))

    def test_gemma3_batch_preparation_passes_one_image_per_sample(self) -> None:
        runner = object.__new__(HFModelRunner)
        runner.backend = "generic_vlm"
        runner.processor_or_tokenizer = Gemma3ProcessorFake()
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / "one.png", Path(tmp) / "two.png"]
            for path in paths:
                Image.new("RGB", (4, 4), color="white").save(path)

            batch, rendered = runner._prepare_batch_inputs(
                ["first", "second"],
                [str(path) for path in paths],
                "system",
            )

        processor = runner.processor_or_tokenizer
        self.assertEqual(processor.image_batch_shape, [1, 1])
        self.assertEqual(processor.padding_side_during_call, "left")
        self.assertEqual(processor.tokenizer.padding_side, "right")
        self.assertEqual(rendered, ["rendered:first", "rendered:second"])
        self.assertIn("input_ids", batch)

    def test_qwen_batch_preparation_processes_all_conversations_together(self) -> None:
        runner = object.__new__(HFModelRunner)
        runner.backend = "qwen2_5_vl"
        runner.processor_or_tokenizer = QwenProcessorFake()
        captured: dict[str, Any] = {}

        def process_vision_info(conversations: list[list[dict[str, Any]]]):
            captured["conversations"] = conversations
            return ["image-one", "image-two"], None

        fake_module = types.ModuleType("qwen_vl_utils")
        fake_module.process_vision_info = process_vision_info  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"qwen_vl_utils": fake_module}):
            _, rendered = runner._prepare_batch_inputs(
                ["first", "second"],
                ["one.png", "two.png"],
                "system",
            )

        processor = runner.processor_or_tokenizer
        self.assertEqual(len(captured["conversations"]), 2)
        self.assertEqual(processor.call["text"], rendered)
        self.assertEqual(processor.call["images"], ["image-one", "image-two"])
        self.assertTrue(processor.call["padding"])
        self.assertEqual(processor.tokenizer.padding_side, "right")


if __name__ == "__main__":
    unittest.main()
