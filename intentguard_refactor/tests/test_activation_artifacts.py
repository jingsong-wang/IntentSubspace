from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intentguard.artifacts import activation_archive_errors  # noqa: E402


ROWS = [
    {"id": "sample-0", "label": 0},
    {"id": "sample-1", "label": 1},
]


def write_archive(path: Path, **overrides: np.ndarray) -> None:
    values = {
        "activations": np.zeros((2, 1, 3), dtype=np.float32),
        "layers": np.array([0], dtype=np.int32),
        "ids": np.array(["sample-0", "sample-1"]),
        "labels": np.array([0, 1], dtype=np.int32),
        "conditions": np.array(["text", "text"]),
        "pair_keys": np.array(["pair-0", "pair-0"]),
        "image_roles": np.array(["none", "none"]),
        "metadata_json": np.array(
            json.dumps(
                {
                    "model": "test/model",
                    "backend": "generic_vlm",
                    "pooling": "last",
                    "multimodal_anchor": False,
                }
            )
        ),
    }
    values.update(overrides)
    np.savez_compressed(path, **values)


class ActivationArtifactTest(unittest.TestCase):
    def test_accepts_current_schema_with_matching_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activations.npz"
            write_archive(path)
            self.assertEqual(activation_archive_errors(path, expected_rows=ROWS), [])

    def test_rejects_legacy_archive_without_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activations.npz"
            write_archive(path)
            with np.load(path) as archive:
                values = {name: archive[name] for name in archive.files if name != "ids"}
            np.savez_compressed(path, **values)

            errors = activation_archive_errors(path, expected_rows=ROWS)

            self.assertTrue(any("missing required fields: ids" in error for error in errors))

    def test_rejects_dataset_order_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activations.npz"
            write_archive(path, ids=np.array(["sample-1", "sample-0"]))

            errors = activation_archive_errors(path, expected_rows=ROWS)

            self.assertTrue(any("sample ids do not exactly match" in error for error in errors))

    def test_requires_anchor_fields_for_v2_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activations.npz"
            write_archive(path)

            errors = activation_archive_errors(
                path,
                expected_rows=ROWS,
                require_multimodal_anchor=True,
            )

            self.assertTrue(any("anchor_activations" in error for error in errors))
            self.assertTrue(any("has_anchor" in error for error in errors))

    def test_rejects_empty_anchor_cache_and_wrong_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activations.npz"
            write_archive(
                path,
                anchor_activations=np.zeros((2, 1, 3), dtype=np.float32),
                has_anchor=np.array([False, False]),
            )

            errors = activation_archive_errors(
                path,
                expected_rows=ROWS,
                require_multimodal_anchor=True,
                expected_model="other/model",
                expected_backend="generic_vlm",
            )

            self.assertTrue(any("does not confirm multimodal-anchor" in error for error in errors))
            self.assertTrue(any("model mismatch" in error for error in errors))

    def test_rejects_pooling_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activations.npz"
            write_archive(path)

            errors = activation_archive_errors(
                path,
                expected_rows=ROWS,
                expected_pooling="image_mean",
            )

            self.assertTrue(any("pooling mismatch" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
