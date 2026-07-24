from __future__ import annotations

import unittest

import numpy as np

from counterfactual_risk_field.cnrf.calibration import worst_group_threshold
from counterfactual_risk_field.cnrf.schema import build_pair_records, validate_no_pack_leakage


class ProtocolTests(unittest.TestCase):
    def test_worst_group_threshold_protects_each_group(self) -> None:
        scores = np.array([-2.0, -1.0, -0.5, 0.0, -3.0, -2.5, 1.0, 2.0])
        groups = np.array(["a"] * 4 + ["b"] * 4)
        threshold, report = worst_group_threshold(scores, groups, alpha=0.25)
        for group in ("a", "b"):
            mask = groups == group
            self.assertLessEqual(float(np.mean(scores[mask] >= threshold)), 0.25)
        self.assertEqual(threshold, report["joint_threshold"])

    def test_pack_leakage_is_a_hard_error(self) -> None:
        rows = []
        for split, pair in (("train", "p1"), ("validation", "p2")):
            for label in (0, 1):
                rows.append(
                    {
                        "id": f"{pair}-{label}",
                        "label": label,
                        "pair_key": pair,
                        "composition_group": "shared-pack",
                        "evaluation_split": split,
                    }
                )
        pairs = build_pair_records(rows)
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_no_pack_leakage(pairs)


if __name__ == "__main__":
    unittest.main()
