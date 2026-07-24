from __future__ import annotations

import unittest

import numpy as np

from counterfactual_risk_field.cnrf.risk_field import CounterfactualRiskField


class RiskFieldTests(unittest.TestCase):
    def test_endpoints_have_signed_unit_coordinate(self) -> None:
        benign = np.array([[1.0, 0.0], [0.0, 1.0]])
        harmful = np.array([[0.8, 0.6], [0.6, 0.8]])
        field = CounterfactualRiskField(
            benign, harmful, ["p1", "p2"], ["g1", "g2"], k=1
        )
        safe = field.score(benign[[0]]).scores[0]
        unsafe = field.score(harmful[[0]]).scores[0]
        self.assertAlmostEqual(safe, -1.0, places=7)
        self.assertAlmostEqual(unsafe, 1.0, places=7)

    def test_retrieval_uses_distinct_pack_ids(self) -> None:
        benign = np.array([[1.0, 0.0], [0.99, 0.02], [0.0, 1.0]])
        harmful = np.array([[0.8, 0.6], [0.79, 0.61], [0.6, 0.8]])
        field = CounterfactualRiskField(
            benign,
            harmful,
            ["p1", "p2", "p3"],
            ["same", "same", "other"],
            k=2,
        )
        result = field.score(np.array([[0.9, 0.3]]))
        self.assertEqual(len(result.neighbor_pack_ids[0]), 2)
        self.assertEqual(len(set(result.neighbor_pack_ids[0])), 2)

    def test_leave_pack_out_prevents_self_match(self) -> None:
        benign = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        harmful = np.array([[0.8, 0.6], [0.6, 0.8], [-0.8, 0.6]])
        field = CounterfactualRiskField(
            benign, harmful, ["p1", "p2", "p3"], ["g1", "g2", "g3"], k=1
        )
        result = field.score(benign[[0]], exclude_pack_ids=["g1"])
        self.assertNotEqual(result.neighbor_pack_ids[0][0], "g1")


if __name__ == "__main__":
    unittest.main()
