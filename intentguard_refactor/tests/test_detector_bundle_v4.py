from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_detector_bundle_v4 import main as build_bundle  # noqa: E402
from intentguard.detector import CISRDetectorBundle  # noqa: E402


def write_detector(path: Path, pooling: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rank = 3
    np.savez_compressed(
        path,
        format_version=np.array(["CISR_v4_detector_v1"]),
        feature_mode=np.array(["raw_rank3"]),
        model_id=np.array(["test-model"]),
        model_alias=np.array(["test"]),
        layer=np.array([2], dtype=np.int32),
        rank=np.array([rank], dtype=np.int32),
        pooling=np.array([pooling]),
        basis=np.eye(rank, 4),
        center=np.zeros(4),
        residual_center=np.zeros(4),
        feature_mean=np.zeros(rank),
        feature_std=np.ones(rank),
        role_categories=np.array([], dtype=str),
        weight_1=np.zeros((rank, 1)),
        bias_1=np.zeros(1),
        weight_2=np.ones((1, 1)),
        bias_2=np.zeros(1),
        threshold=np.array([0.8]),
        safe_threshold=np.array([0.2]),
        danger_threshold=np.array([0.8]),
        safe_route_enabled=np.array([True]),
        danger_route_enabled=np.array([True]),
        anchor_prompt=np.array([""]),
        uses_anchor=np.array([False]),
    )


def write_summary(path: Path, auc: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "splits": {
                    "validation": {
                        "auc": auc,
                        "average_precision": auc - 0.01,
                        "balanced_accuracy": auc - 0.02,
                        "view_stability": {"mean_probability_std": 0.03},
                    }
                }
            }
        ),
        encoding="utf-8",
    )


class CISRV4BundleBuildTest(unittest.TestCase):
    def test_selects_pooling_per_branch_using_validation_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = [
                ("text", "last", 0.80),
                ("text", "non_image_mean", 0.92),
                ("multimodal", "last", 0.75),
                ("multimodal", "image_mean", 0.95),
            ]
            argv = ["build_detector_bundle_v4.py"]
            for branch, pooling, auc in candidates:
                candidate_dir = root / branch / pooling
                summary = candidate_dir / "summary.json"
                detector = candidate_dir / "detector.npz"
                write_summary(summary, auc)
                write_detector(detector, pooling)
                argv.extend(
                    ["--candidate", branch, pooling, str(summary), str(detector)]
                )
            output = root / "bundle"
            argv.extend(["--out-dir", str(output), "--model-alias", "test"])

            with patch.object(sys, "argv", argv):
                build_bundle()

            bundle = CISRDetectorBundle.load(output / "detector_bundle.json")
            self.assertEqual(bundle.text_detector.pooling, "non_image_mean")
            self.assertEqual(bundle.multimodal_detector.pooling, "image_mean")


if __name__ == "__main__":
    unittest.main()
