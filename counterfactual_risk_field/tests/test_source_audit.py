from __future__ import annotations

import unittest

import numpy as np

from counterfactual_risk_field.cnrf.schema import PairRecord
from counterfactual_risk_field.cnrf.source_audit import audit_pair_representations


class SourceAuditTests(unittest.TestCase):
    def test_paired_difference_cancels_shared_source_offset(self) -> None:
        rng = np.random.default_rng(9)
        pairs = []
        activations = []
        for index in range(30):
            source = "left" if index < 15 else "right"
            offset = np.array([-4.0, 0.0, 0.0]) if source == "left" else np.array([4.0, 0.0, 0.0])
            content = rng.normal(0.0, 0.1, size=3)
            safe = offset + content + np.array([0.0, 0.0, -0.5])
            harmful = offset + content + np.array([0.0, 0.0, 0.5])
            benign_index = len(activations)
            activations.extend([safe, harmful])
            pairs.append(
                PairRecord(
                    pair_id=f"pair-{index}",
                    pack_id=f"pack-{index}",
                    split="reference",
                    modality="text",
                    benign_index=benign_index,
                    harmful_index=benign_index + 1,
                    semantic_source=source,
                    carrier_source=source,
                )
            )
        report = audit_pair_representations(
            np.asarray(activations), pairs, source_axis="semantic", seed=11
        )
        self.assertGreater(report["raw_endpoints"]["macro_f1"], 0.9)
        self.assertGreater(
            report["raw_endpoints"]["macro_f1"], report["arrows"]["macro_f1"] + 0.2
        )


if __name__ == "__main__":
    unittest.main()
