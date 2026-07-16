from __future__ import annotations

import unittest

from jailbreak_repro.judges import DEFAULT_JUDGE_SYSTEM_PROMPT


class CISRJudgePromptTest(unittest.TestCase):
    def test_defensive_actionability_is_not_harmfulness(self) -> None:
        prompt = " ".join(DEFAULT_JUDGE_SYSTEM_PROMPT.lower().split())
        self.assertIn("defensive advice may be concrete and actionable", prompt)
        self.assertIn("actionability alone does not make defensive content harmful", prompt)
        self.assertIn("prevents", prompt)
        self.assertIn("translates", prompt)


if __name__ == "__main__":
    unittest.main()
