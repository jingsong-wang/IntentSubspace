from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ood_intent_study import ARTIFACT_VERSION
from ood_intent_study.analyze import main as analyze_main
from ood_intent_study.extract import _write_shard
from ood_intent_study.io_utils import canonical_json, sha256_text, write_jsonl_atomic
from ood_intent_study.tests.test_manifest import sample


class _FixtureProbe:
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        normalized = (X - self.mean) / self.scale
        self.direction = normalized[y == 1].mean(axis=0) - normalized[y == 0].mean(axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = ((X - self.mean) / self.scale) @ self.direction
        probability = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
        return np.column_stack([1.0 - probability, probability])


class AnalysisSmokeTests(unittest.TestCase):
    def test_synthetic_end_to_end_analysis(self) -> None:
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            activation_dir = root / "activations"
            manifest = root / "samples.jsonl"
            output = root / "analysis"
            rows = []
            batch = []
            index = 0
            for split, count in (("train", 16), ("validation", 8), ("test", 8)):
                for local in range(count):
                    label = local % 2
                    source = "XSTest" if split == "test" else f"source-{label}"
                    row = sample(f"s-{index}", source, f"semantic-{index}", f"group-{index}")
                    row = row.__class__(**{**row.to_dict(), "label": label, "label_name": "harmful" if label else "benign", "split": split, "split_group_id": f"split-{index}"})
                    rows.append(row.to_dict())
                    layer_values = []
                    for layer in range(3):
                        layer_values.append(rng.normal(size=6) + label * (layer + 1) * 0.8)
                    base = np.stack(layer_values, axis=0)
                    acts = np.stack([base, base + 0.01, base - 0.01], axis=0)
                    batch.append(
                        {
                            "sample_id": row.sample_id,
                            "activations": acts,
                            "readout_valid": np.array([True, True, True]),
                            "sequence_length": 12,
                            "image_token_count": 4,
                            "text_token_count": 8,
                            "image_width": 32,
                            "image_height": 32,
                            "rendered_prompt_sha256": str(index),
                        }
                    )
                    index += 1
            for local in range(8):
                row = sample(f"attack-{local}", "FigStep", f"attack-semantic-{local}", f"attack-group-{local}", attack=True)
                rows.append(row.to_dict())
                base = np.stack(
                    [rng.normal(size=6) + (layer + 1) * 0.2 for layer in range(3)], axis=0
                )
                acts = np.stack([base, base + 0.01, base - 0.01], axis=0)
                batch.append(
                    {
                        "sample_id": row.sample_id,
                        "activations": acts,
                        "readout_valid": np.array([True, True, True]),
                        "sequence_length": 20,
                        "image_token_count": 8,
                        "text_token_count": 12,
                        "image_width": 32,
                        "image_height": 32,
                        "rendered_prompt_sha256": str(index + local),
                    }
                )
            write_jsonl_atomic(manifest, rows)
            manifest_sha256 = sha256_text("\n".join(canonical_json(row) for row in rows))
            _write_shard(
                activation_dir / "shard_00000.npz",
                batch,
                layers=[1, 2, 3],
                readouts=["last", "non_image_mean", "image_mean"],
                metadata={
                    "artifact_version": ARTIFACT_VERSION,
                    "manifest_sha256": manifest_sha256,
                    "run_fingerprint": "fixture",
                    "model": "fixture",
                },
                storage_dtype="float32",
            )
            with (
                patch(
                    "ood_intent_study.analyze.fit_logistic",
                    side_effect=lambda X, y, seed: _FixtureProbe(X, y),
                ),
                patch("ood_intent_study.analyze.multiclass_centroid_macro_f1", return_value=float("nan")),
                patch("ood_intent_study.analyze.centroid_domain_auc", return_value=0.5),
                redirect_stdout(io.StringIO()),
            ):
                result = analyze_main(
                    [
                        "--activations",
                        str(activation_dir),
                        "--manifest",
                        str(manifest),
                        "--out-dir",
                        str(output),
                        "--bootstrap",
                        "0",
                        "--allow-incomplete",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertTrue((output / "analysis.json").is_file())
            self.assertTrue((output / "attack_shift_metrics.csv").is_file())
            self.assertTrue((output / "source_label_metrics.csv").is_file())
            self.assertTrue((output / "leave_one_source_out_label_metrics.csv").is_file())
            self.assertGreater((output / "common_panel_layer_metrics.csv").stat().st_size, 0)
            analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(analysis["modality_panel"], "all")
            self.assertEqual(analysis["coverage"]["modality_panel"], "all")
            self.assertEqual(analysis["common_multimodal_panel"]["rows"], 40)
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("## Standard Panel Composition", report)
            self.assertIn("| XSTest | 4 | 4 | 8 |", report)
            with (output / "source_label_metrics.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                source_labels = {row["source_label"] for row in csv.DictReader(handle)}
            self.assertIn("XSTest-safe", source_labels)
            self.assertIn("XSTest-unsafe", source_labels)
            with (output / "leave_one_source_out.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                xstest_loso = [
                    row
                    for row in csv.DictReader(handle)
                    if row["held_out_source"] == "XSTest"
                ]
            with (output / "leave_one_source_out_label_metrics.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                xstest_loso_labels = [
                    row
                    for row in csv.DictReader(handle)
                    if row["held_out_source"] == "XSTest"
                ]
            self.assertEqual(len(xstest_loso), 3)
            self.assertEqual(len(xstest_loso_labels), 6)
            self.assertEqual(
                {row["held_out_source_label"] for row in xstest_loso_labels},
                {"XSTest-safe", "XSTest-unsafe"},
            )
            self.assertEqual(set(analysis["common_multimodal_panel"]["selected_layers"]), {
                "last",
                "non_image_mean",
                "image_mean",
            })

    def test_text_and_multimodal_panels_refit_on_disjoint_rows(self) -> None:
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            activation_dir = root / "activations"
            manifest = root / "samples.jsonl"
            rows = []
            batch = []
            index = 0

            for split in ("train", "validation", "test"):
                for modality in ("text", "image_text"):
                    for label in (0, 1):
                        for _ in range(2):
                            source = (
                                "XSTest"
                                if modality == "text"
                                else ("MM-Vet" if label == 0 else "HADES")
                            )
                            row = sample(
                                f"panel-{index}",
                                source,
                                f"semantic-{index}",
                                f"group-{index}",
                            )
                            row = row.__class__(
                                **{
                                    **row.to_dict(),
                                    "label": label,
                                    "label_name": "harmful" if label else "benign",
                                    "modality": modality,
                                    "split": split,
                                    "split_group_id": f"split-{index}",
                                }
                            )
                            rows.append(row.to_dict())
                            base = np.stack(
                                [rng.normal(size=5) + label * (layer + 1) for layer in range(2)]
                            )
                            batch.append(
                                {
                                    "sample_id": row.sample_id,
                                    "activations": np.stack(
                                        [base, base + 0.02, base - 0.02], axis=0
                                    ),
                                    "readout_valid": np.array(
                                        [True, True, modality == "image_text"]
                                    ),
                                    "sequence_length": 16,
                                    "image_token_count": 4 if modality == "image_text" else 0,
                                    "text_token_count": 12,
                                    "image_width": 32 if modality == "image_text" else 0,
                                    "image_height": 32 if modality == "image_text" else 0,
                                    "rendered_prompt_sha256": str(index),
                                }
                            )
                            index += 1

            for local in range(4):
                row = sample(
                    f"panel-attack-{local}",
                    "FigStep",
                    f"attack-semantic-{local}",
                    f"attack-group-{local}",
                    attack=True,
                )
                row = row.__class__(**{**row.to_dict(), "modality": "image_text"})
                rows.append(row.to_dict())
                base = np.stack([rng.normal(size=5) + layer + 1 for layer in range(2)])
                batch.append(
                    {
                        "sample_id": row.sample_id,
                        "activations": np.stack([base, base + 0.02, base - 0.02], axis=0),
                        "readout_valid": np.array([True, True, True]),
                        "sequence_length": 20,
                        "image_token_count": 4,
                        "text_token_count": 16,
                        "image_width": 32,
                        "image_height": 32,
                        "rendered_prompt_sha256": f"attack-{local}",
                    }
                )

            write_jsonl_atomic(manifest, rows)
            manifest_sha256 = sha256_text("\n".join(canonical_json(row) for row in rows))
            _write_shard(
                activation_dir / "shard_00000.npz",
                batch,
                layers=[1, 2],
                readouts=["last", "non_image_mean", "image_mean"],
                metadata={
                    "artifact_version": ARTIFACT_VERSION,
                    "manifest_sha256": manifest_sha256,
                    "run_fingerprint": "panel-fixture",
                    "model": "fixture",
                },
                storage_dtype="float32",
            )

            with (
                patch(
                    "ood_intent_study.analyze.fit_logistic",
                    side_effect=lambda X, y, seed: _FixtureProbe(X, y),
                ),
                patch(
                    "ood_intent_study.analyze.multiclass_centroid_macro_f1",
                    return_value=float("nan"),
                ),
                patch("ood_intent_study.analyze.centroid_domain_auc", return_value=0.5),
                redirect_stdout(io.StringIO()),
            ):
                for panel in ("text_only", "multimodal_only"):
                    self.assertEqual(
                        analyze_main(
                            [
                                "--activations",
                                str(activation_dir),
                                "--manifest",
                                str(manifest),
                                "--out-dir",
                                str(root / panel),
                                "--modality-panel",
                                panel,
                                "--bootstrap",
                                "0",
                                "--skip-loso",
                                "--allow-incomplete",
                            ]
                        ),
                        0,
                    )

            text_analysis = json.loads((root / "text_only" / "analysis.json").read_text())
            image_analysis = json.loads(
                (root / "multimodal_only" / "analysis.json").read_text()
            )
            self.assertEqual(text_analysis["coverage"]["by_modality"], {"text": 12})
            self.assertEqual(
                image_analysis["coverage"]["by_modality"], {"image_text": 16}
            )
            self.assertEqual(
                text_analysis["coverage"]["standard_by_modality_label"],
                {"text": {"0": 6, "1": 6}},
            )
            self.assertEqual(
                image_analysis["coverage"]["standard_by_modality_label"],
                {"image_text": {"0": 6, "1": 6}},
            )
            self.assertEqual(
                image_analysis["coverage"]["standard_by_source_label"],
                {"HADES": {"1": 6}, "MM-Vet": {"0": 6}},
            )
            self.assertEqual(
                set(text_analysis["selected_layers"]), {"last", "non_image_mean"}
            )
            self.assertEqual(
                set(image_analysis["selected_layers"]),
                {"last", "non_image_mean", "image_mean"},
            )
            self.assertEqual(
                (root / "text_only" / "attack_shift_metrics.csv")
                .read_text(encoding="utf-8")
                .strip(),
                "",
            )
            self.assertGreater(
                (root / "multimodal_only" / "attack_shift_metrics.csv").stat().st_size,
                1,
            )


if __name__ == "__main__":
    unittest.main()
