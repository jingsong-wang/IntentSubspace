from __future__ import annotations

import unittest

import numpy as np

from counterfactual_risk_field.cnrf.activations import ActivationData
from counterfactual_risk_field.cnrf.experiment import Experiment


class ExperimentTests(unittest.TestCase):
    def test_end_to_end_synthetic_protocol(self) -> None:
        rows = []
        vectors = []
        ids = []
        split_plan = ["train"] * 6 + ["validation"] * 3 + ["calibration"] * 3 + ["test"] * 3
        for pack_index, split in enumerate(split_plan):
            content = np.array([2.0, 0.0, 0.0, 0.0])
            for label, sign in ((0, -1.0), (1, 1.0)):
                sample_id = f"p{pack_index}-{label}"
                vector_layer_1 = content + np.array([0.0, 0.0, 0.0, sign])
                vector_layer_2 = content + np.array([0.0, 0.0, sign * 0.2, sign * 1.5])
                rows.append(
                    {
                        "id": sample_id,
                        "label": label,
                        "pair_key": f"pair-{pack_index}",
                        "composition_group": f"pack-{pack_index}",
                        "evaluation_split": split,
                        "modality_branch": "text",
                        "condition": "group-a" if pack_index % 2 else "group-b",
                        "source": "synthetic",
                    }
                )
                ids.append(sample_id)
                vectors.append(np.stack([vector_layer_1, vector_layer_2]))
        array = np.asarray(vectors, dtype=np.float32)[:, None, :, :]
        data = ActivationData(
            sample_ids=np.asarray(ids),
            activations=array,
            readout_valid=np.ones((len(ids), 1), dtype=bool),
            layers=np.array([1, 2], dtype=np.int32),
            readouts=np.array(["last"]),
            metadata={"model": "synthetic"},
        )
        experiment = Experiment(rows, data, k=2, alpha=0.2, max_reference_packs=6)
        summary, scores = experiment.run(readouts=["last"], layer_candidates=[1, 2])
        test = summary["branches"]["text"]["splits"]["test"]["overall"]
        self.assertGreaterEqual(test["tpr"], 0.99)
        self.assertLessEqual(test["fpr"], 0.01)
        self.assertEqual(len(scores), len(rows))


if __name__ == "__main__":
    unittest.main()
