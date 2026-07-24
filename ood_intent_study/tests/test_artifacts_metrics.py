from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ood_intent_study import ARTIFACT_VERSION
from ood_intent_study.artifacts import load_activation_table, read_shard
from ood_intent_study.audit_activations import audit_activation_shards
from ood_intent_study.extract import _write_shard
from ood_intent_study.analyze import _summarize_loso
from ood_intent_study.metrics import average_precision, balanced_threshold, rbf_mmd_1d, roc_auc, score_metrics


class ArtifactMetricTests(unittest.TestCase):
    @staticmethod
    def _activation_row(sample_id: str, value: float) -> dict[str, object]:
        return {
            "sample_id": sample_id,
            "activations": np.full((1, 1, 4), value, dtype=np.float32),
            "readout_valid": np.array([True]),
            "sequence_length": 10,
            "image_token_count": 0,
            "text_token_count": 10,
            "image_width": 0,
            "image_height": 0,
            "rendered_prompt_sha256": sample_id,
        }

    @staticmethod
    def _metadata() -> dict[str, str]:
        return {
            "artifact_version": ARTIFACT_VERSION,
            "manifest_sha256": "manifest",
            "run_fingerprint": "run",
            "model": "fixture",
        }

    def test_shards_round_trip_and_concatenate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {
                "artifact_version": ARTIFACT_VERSION,
                "manifest_sha256": "manifest",
                "run_fingerprint": "run",
                "model": "fixture",
            }
            for shard_index in range(2):
                batch = []
                for row_index in range(2):
                    value = shard_index * 2 + row_index
                    batch.append(
                        {
                            "sample_id": f"sample-{value}",
                            "activations": np.full((2, 3, 4), value, dtype=np.float32),
                            "readout_valid": np.array([True, value % 2 == 0]),
                            "sequence_length": 10 + value,
                            "image_token_count": value,
                            "text_token_count": 10,
                            "image_width": 32,
                            "image_height": 24,
                            "rendered_prompt_sha256": str(value),
                        }
                    )
                _write_shard(
                    root / f"shard_{shard_index:05d}.npz",
                    batch,
                    layers=[1, 2, 3],
                    readouts=["last", "image_mean"],
                    metadata=metadata,
                    storage_dtype="float16",
                )
            table = load_activation_table(root)
            self.assertEqual(table.activations.shape, (4, 2, 3, 4))
            self.assertEqual(table.sample_ids.tolist(), ["sample-0", "sample-1", "sample-2", "sample-3"])
            self.assertEqual(table.image_token_counts.tolist(), [0, 1, 2, 3])

    def test_binary_metrics_handle_ties(self) -> None:
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.3, 0.7, 0.9])
        self.assertEqual(roc_auc(y, scores), 1.0)
        threshold = balanced_threshold(y, scores)
        metrics = score_metrics(y, scores, threshold)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertGreater(rbf_mmd_1d(scores[:2], scores[2:]), 0.0)

    def test_float16_storage_fails_before_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard_00000.npz"
            with self.assertRaisesRegex(OverflowError, "FP16 activation storage would overflow"):
                _write_shard(
                    path,
                    [self._activation_row("large", 70_000.0)],
                    layers=[1],
                    readouts=["last"],
                    metadata=self._metadata(),
                    storage_dtype="float16",
                )
            self.assertFalse(path.exists())

    def test_bfloat16_storage_preserves_large_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard_00000.npz"
            _write_shard(
                path,
                [self._activation_row("large", 1_000_000.0)],
                layers=[1],
                readouts=["last"],
                metadata=self._metadata(),
                storage_dtype="bfloat16",
            )
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["activations"].dtype, np.uint16)
            shard = read_shard(path)
            self.assertTrue(np.isfinite(shard.activations).all())
            self.assertLess(abs(float(shard.activations[0, 0, 0, 0]) - 1_000_000.0) / 1_000_000.0, 0.01)
            audit = audit_activation_shards(Path(directory))
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["storage_dtype_shards"], {"bfloat16": 1})

    def test_reader_and_audit_reject_nonfinite_legacy_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "shard_00000.npz"
            _write_shard(
                path,
                [self._activation_row("broken", 1.0)],
                layers=[1],
                readouts=["last"],
                metadata=self._metadata(),
                storage_dtype="float32",
            )
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            arrays["activations"][0, 0, 0, 0] = np.inf
            with path.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            with self.assertRaisesRegex(ValueError, "legacy FP16 storage overflow"):
                read_shard(path)
            audit = audit_activation_shards(root)
            self.assertEqual(audit["status"], "FAIL")
            self.assertEqual(audit["affected_rows"], 1)
            self.assertEqual(audit["nonfinite_by_readout_layer"], {"last:layer_1": 1})

    def test_average_precision_is_invariant_to_tie_order(self) -> None:
        tied = np.array([0.5, 0.5])
        self.assertEqual(average_precision(np.array([1, 0]), tied), 0.5)
        self.assertEqual(average_precision(np.array([0, 1]), tied), 0.5)

    def test_loso_summary_uses_source_label_cells(self) -> None:
        rows = pd.DataFrame(
            [
                {"readout": "last", "layer": 1, "held_out_source": "harmful", "eligible": True, "positive_n": 4, "negative_n": 0, "tpr": 0.6, "tnr": np.nan, "ineligible_reason": ""},
                {"readout": "last", "layer": 1, "held_out_source": "benign", "eligible": True, "positive_n": 0, "negative_n": 4, "tpr": np.nan, "tnr": 0.8, "ineligible_reason": ""},
                {"readout": "last", "layer": 1, "held_out_source": "mixed", "eligible": True, "positive_n": 2, "negative_n": 2, "tpr": 0.7, "tnr": 0.9, "ineligible_reason": ""},
                {"readout": "last", "layer": 1, "held_out_source": "missing", "eligible": False, "ineligible_reason": "training_missing_class"},
            ]
        )
        summary = _summarize_loso(rows, iterations=0, seed=3).iloc[0]
        self.assertAlmostEqual(float(summary["harmful_macro_tpr"]), 0.65)
        self.assertAlmostEqual(float(summary["benign_macro_tnr"]), 0.85)
        self.assertAlmostEqual(float(summary["macro_balanced_source"]), 0.75)
        self.assertAlmostEqual(float(summary["worst_source_label_cell"]), 0.6)
        self.assertEqual(int(summary["ineligible_source_n"]), 1)


if __name__ == "__main__":
    unittest.main()
