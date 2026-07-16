from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_detector import choose_coverage_threshold, validate_protocol  # noqa: E402


def pair_rows(pair_key: str, template_id: str, split: str) -> list[dict]:
    common = {
        "pair_key": pair_key,
        "template_id": template_id,
        "evaluation_split": split,
        "intent_family": "cyber",
        "condition": template_id.split("::", 1)[0],
        "wrapper_family": "direct",
        "carrier_type": "text",
        "image_role": "none",
    }
    return [{**common, "id": f"{pair_key}-0", "label": 0}, {**common, "id": f"{pair_key}-1", "label": 1}]


class DetectionProtocolTest(unittest.TestCase):
    def test_conservative_threshold_meets_recall_lower_bound(self) -> None:
        labels = np.array([0] * 200 + [1] * 200)
        probabilities = np.concatenate([np.linspace(0.01, 0.40, 200), np.linspace(0.60, 0.99, 200)])

        threshold, selection = choose_coverage_threshold(
            labels,
            probabilities,
            target_tpr=0.95,
            target_fpr=0.10,
            confidence=0.95,
        )

        self.assertGreater(threshold, 0.40)
        self.assertTrue(selection["coverage_constraint_met"])
        self.assertTrue(selection["fpr_constraint_met"])
        self.assertGreaterEqual(selection["tpr_lower_confidence_bound"], 0.95)

    def test_threshold_reports_empirical_fallback_when_sample_is_too_small(self) -> None:
        labels = np.array([0] * 20 + [1] * 20)
        probabilities = np.concatenate([np.linspace(0.01, 0.40, 20), np.linspace(0.60, 0.99, 20)])

        _, selection = choose_coverage_threshold(
            labels,
            probabilities,
            target_tpr=0.99,
            target_fpr=0.10,
            confidence=0.95,
        )

        self.assertFalse(selection["coverage_constraint_met"])
        self.assertTrue(selection["fallback_to_empirical_coverage"])
        self.assertEqual(selection["tpr"], 1.0)

    def test_protocol_counts_strict_pairs_and_disjoint_templates(self) -> None:
        rows = []
        for index, split in enumerate(("train", "validation", "calibration", "test")):
            rows.extend(pair_rows(f"pair-{index}", f"condition-{index}::000", split))

        report = validate_protocol(rows)

        self.assertEqual(report["pair_count"], 4)
        self.assertEqual(report["malformed_pair_count"], 0)
        self.assertEqual(report["split_pair_counts"], {split: 1 for split in ("train", "validation", "calibration", "test")})

    def test_protocol_rejects_malformed_pair(self) -> None:
        rows = pair_rows("pair", "condition::000", "train")
        rows[0]["label"] = 1
        with self.assertRaisesRegex(ValueError, "Malformed counterfactual pairs"):
            validate_protocol(rows)

    def test_protocol_rejects_template_leakage(self) -> None:
        rows = pair_rows("pair-a", "condition::000", "train")
        rows.extend(pair_rows("pair-b", "condition::000", "test"))
        with self.assertRaisesRegex(ValueError, "Split leakage"):
            validate_protocol(rows)


if __name__ == "__main__":
    unittest.main()
