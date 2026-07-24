from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np

from jailbreak_repro.cisr import inspect_cisr_artifact_version
from jailbreak_repro.run_experiment import default_out_dir, parse_args, response_config


def write_version_artifact(path: Path, format_version: str | None) -> None:
    if format_version is None:
        np.savez(path, threshold=np.array([0.5]))
    else:
        np.savez(path, format_version=np.array([format_version]))


class CISRVersioningTest(unittest.TestCase):
    def _args(self, detector: Path, defense: str):
        return parse_args(
            [
                "--model-preset",
                "mock",
                "--benchmark",
                "XSTest",
                "--defense",
                defense,
                "--cisr-detector",
                str(detector),
                "--judge-mode",
                "none",
            ]
        )

    def test_legacy_alias_resolves_to_artifact_version_and_separates_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v2 = root / "cisr2.npz"
            v3 = root / "cisr3.npz"
            write_version_artifact(v2, "CISR_v2_detector_v2")
            write_version_artifact(v3, "CISR_v3_detector_v1")

            args_v2 = self._args(v2, "cisr")
            args_v3 = self._args(v3, "cisr")

        self.assertEqual(args_v2.defense, "cisr2")
        self.assertEqual(args_v2.cisr_version, "cisr2")
        self.assertEqual(args_v3.defense, "cisr3")
        self.assertEqual(args_v3.cisr_version, "cisr3")
        self.assertIn("defense_cisr2", default_out_dir(args_v2).parts)
        self.assertIn("defense_cisr3", default_out_dir(args_v3).parts)
        self.assertNotEqual(default_out_dir(args_v2), default_out_dir(args_v3))
        self.assertEqual(response_config(args_v2)["cisr_version"], "cisr2")
        self.assertEqual(response_config(args_v3)["cisr_version"], "cisr3")

    def test_explicit_version_rejects_mismatched_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            detector = Path(tmp) / "detector.npz"
            write_version_artifact(detector, "CISR_v2_detector_v2")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self._args(detector, "cisr3")

    def test_legacy_artifact_without_format_is_cisr2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            detector = Path(tmp) / "legacy.npz"
            write_version_artifact(detector, None)
            self.assertEqual(inspect_cisr_artifact_version(detector), "cisr2")

    def test_v4_artifact_requires_explicit_detection_only_monitor_without_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            detector = Path(tmp) / "detector_v4.npz"
            write_version_artifact(detector, "CISR_v4_detector_v1")
            args = parse_args(
                [
                    "--model-preset",
                    "mock",
                    "--benchmark",
                    "XSTest",
                    "--defense",
                    "cisr4",
                    "--cisr-detector",
                    str(detector),
                    "--cisr4-review-action",
                    "monitor",
                    "--judge-mode",
                    "none",
                ]
            )
        self.assertEqual(args.defense, "cisr4")
        self.assertEqual(args.cisr_version, "cisr4")
        self.assertEqual(args.cisr4_review_action, "monitor")

    def test_v4_json_bundle_resolves_as_cisr4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "detector_bundle.json"
            bundle.write_text(
                json.dumps(
                    {
                        "format_version": "CISR_v4_detector_bundle_v1",
                        "branches": {},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(inspect_cisr_artifact_version(bundle), "cisr4")


if __name__ == "__main__":
    unittest.main()
