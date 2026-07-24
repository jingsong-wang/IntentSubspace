from __future__ import annotations

import hashlib
import unittest

from jailbreak_repro.judges import (
    DEFAULT_JUDGE_SYSTEM_PROMPT,
    XSTEST_JUDGE_PROMPT_TEMPLATE,
    XSTEST_JUDGE_SYSTEM_PROMPT,
    XSTEST_LABEL_ONLY_INSTRUCTION,
)
from jailbreak_repro.run_experiment import judge_config, parse_args


class CISRJudgePromptTest(unittest.TestCase):
    def test_defensive_actionability_is_not_harmfulness(self) -> None:
        prompt = " ".join(DEFAULT_JUDGE_SYSTEM_PROMPT.lower().split())
        self.assertIn("defensive advice may be concrete and actionable", prompt)
        self.assertIn("actionability alone does not make defensive content harmful", prompt)
        self.assertIn("prevents", prompt)
        self.assertIn("translates", prompt)

    def test_gemma_model_judge_config_hashes_standard_asr_prompt(self) -> None:
        args = parse_args(
            [
                "--model-preset",
                "mock",
                "--benchmark",
                "HADES",
                "--defense",
                "none",
                "--judge-mode",
                "model",
                "--judge-preset",
                "gemma3_12b",
                "--judge-model",
                "google/gemma-3-12b-it",
                "--judge-backend",
                "generic_vlm",
                "--judge-model-source",
                "modelscope",
            ]
        )
        args.effective_judge_task = "asr"
        config = judge_config(args)

        self.assertEqual(config["judge_mode"], "model")
        self.assertEqual(config["judge_model"], "google/gemma-3-12b-it")
        self.assertEqual(config["judge_backend"], "generic_vlm")
        self.assertEqual(config["judge_model_source"], "modelscope")
        self.assertEqual(
            config["judge_system_prompt_sha1"],
            hashlib.sha1(DEFAULT_JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        )

    def test_xstest_config_hashes_repository_standard_protocol(self) -> None:
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
                "gemma3_12b",
            ]
        )
        args.effective_judge_task = "xstest"
        config = judge_config(args)
        payload = "\n\n".join(
            [
                XSTEST_JUDGE_SYSTEM_PROMPT,
                XSTEST_JUDGE_PROMPT_TEMPLATE,
                XSTEST_LABEL_ONLY_INSTRUCTION,
            ]
        )
        self.assertEqual(
            config["xstest_judge_prompt_sha1"],
            hashlib.sha1(payload.encode("utf-8")).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
