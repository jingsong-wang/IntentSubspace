from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jailbreak_repro.prepare_representation_data import normalize_composition
from jailbreak_repro.summarize_representation_reproduction import collect
from jailbreak_repro.train_representation_detector import (
    ActivationArchive,
    activation_archive_logical_fingerprint,
    fit_mcd,
    load_activation_archive,
    parse_args,
    rcs_repository_threshold_indices,
    select_rcs_layers,
)


class RepresentationRepositoryReproductionTest(unittest.TestCase):
    def test_lightweight_activation_fingerprint_matches_full_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "activations.npz"
            np.savez_compressed(
                path,
                activations=np.zeros((2, 1, 3), dtype=np.float32),
                layers=np.array([1], dtype=np.int32),
                labels=np.array([0, 1], dtype=np.int32),
                ids=np.array(["safe", "unsafe"]),
                evaluation_splits=np.array(["train", "train"]),
                metadata_json=json.dumps({"model": "victim"}),
            )
            lightweight = activation_archive_logical_fingerprint(path)
            full = load_activation_archive(path).logical_fingerprint
        self.assertEqual(lightweight, full)

    def test_paper_rcs_resolves_released_defaults(self) -> None:
        args = parse_args(
            [
                "--activations",
                "missing.npz",
                "--method",
                "rcs-kcd",
                "--out",
                "detector.npz",
                "--protocol",
                "paper-rcs",
            ]
        )
        self.assertEqual(args.selection_split, "train")
        self.assertEqual(args.calibration_split, "train")
        self.assertEqual(args.dataset_field, "source")
        self.assertEqual(args.rcs_layer_selection, "official-composite")
        self.assertEqual(args.layer_selection_max_per_class, 1000)
        self.assertEqual(args.k, 40)
        self.assertEqual(args.mcd_min_cluster_size, 50)
        self.assertEqual(args.seed, 45)

    def test_released_threshold_sampling_is_per_source_and_deterministic(self) -> None:
        labels = np.array([0] * 120 + [0] * 80 + [1] * 130, dtype=np.int32)
        datasets = np.array(["safe_a"] * 120 + ["safe_b"] * 80 + ["unsafe"] * 130)
        train = np.ones(len(labels), dtype=bool)
        first = rcs_repository_threshold_indices(labels, datasets, train, 100, 45)
        second = rcs_repository_threshold_indices(labels, datasets, train, 100, 45)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 280)
        self.assertEqual(int(np.sum(datasets[first] == "safe_a")), 100)
        self.assertEqual(int(np.sum(datasets[first] == "safe_b")), 80)
        self.assertEqual(int(np.sum(datasets[first] == "unsafe")), 100)

    def test_released_mcd_covariance_builds_finite_precision(self) -> None:
        rng = np.random.default_rng(7)
        benign = rng.normal(-1.0, 0.5, size=(60, 4))
        malicious = rng.normal(1.0, 0.5, size=(60, 4))
        projected = np.vstack([benign, malicious]).astype(np.float32)
        labels = np.array([0] * 60 + [1] * 60, dtype=np.int32)
        datasets = np.array(["safe"] * 60 + ["unsafe"] * 60)
        arrays, metadata = fit_mcd(
            projected,
            labels,
            datasets,
            minimum_cluster_size=50,
            covariance_mode="released-analytical-shrinkage",
        )
        self.assertTrue(np.isfinite(arrays["benign_precisions"]).all())
        self.assertTrue(np.isfinite(arrays["malicious_precisions"]).all())
        self.assertEqual(metadata["mcd_covariance"], "released-analytical-shrinkage")
        self.assertEqual(metadata["benign_cluster_count"], 1)
        self.assertEqual(metadata["malicious_cluster_count"], 1)

    def test_official_composite_layer_selection_runs_on_frozen_training_split(self) -> None:
        if importlib.util.find_spec("scipy") is None or importlib.util.find_spec("sklearn") is None:
            self.skipTest("official RCS layer analysis requires scipy and scikit-learn")
        rng = np.random.default_rng(8)
        labels = np.array([0] * 12 + [1] * 12, dtype=np.int32)
        weak = rng.normal(0.0, 1.0, size=(24, 4))
        strong = rng.normal(0.0, 0.2, size=(24, 4))
        strong[labels == 0, 0] -= 2.0
        strong[labels == 1, 0] += 2.0
        activations = np.stack([weak, strong], axis=1).astype(np.float32)
        fields = {
            key: np.full(24, "", dtype=str)
            for key in (
                "ids",
                "evaluation_splits",
                "pair_keys",
                "conditions",
                "intent_families",
                "sources",
                "carrier_types",
                "image_roles",
                "image_paths",
                "prompt_texts",
                "label_names",
                "intent_ids",
            )
        }
        fields["ids"] = np.array([f"row-{index}" for index in range(24)])
        fields["evaluation_splits"] = np.full(24, "train")
        archive = ActivationArchive(
            path=Path("synthetic.npz"),
            activations=activations,
            layers=np.array([1, 2], dtype=np.int32),
            labels=labels,
            fields=fields,
            metadata={},
            logical_fingerprint="synthetic",
        )
        ranking = select_rcs_layers(
            archive,
            split="train",
            maximum_per_class=12,
            seed=45,
            mode="official-composite",
        )
        self.assertEqual(len(ranking), 2)
        self.assertIn("overall_score", ranking[0])
        self.assertEqual(ranking[0]["selection_sample_count"], 24)

    def test_normalizer_preserves_official_source_cluster(self) -> None:
        composition = {
            "train": {
                "JailbreakV-28K_llm_transfer_attack": [
                    {"txt": "unsafe request", "img": None, "toxicity": 1}
                ]
            },
            "test": {},
        }
        rows, missing = normalize_composition(composition, Path.cwd(), "absolute")
        self.assertEqual(missing, [])
        self.assertEqual(rows[0]["source"], "JailbreakV-28K")
        self.assertEqual(rows[0]["condition"], "JailbreakV-28K_llm_transfer_attack")
        self.assertEqual(rows[0]["evaluation_split"], "train")

    def test_external_summary_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "qwen" / "rcs-kcd" / "external" / "xstest" / "summary.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "representation_detector": {
                            "method": "rcs-kcd",
                            "protocol": "paper-rcs",
                            "labeled_count": 450,
                            "safe_count": 250,
                            "unsafe_count": 200,
                            "auroc": 0.9,
                            "tpr": 0.8,
                            "fpr": 0.1,
                            "detection_rate": 0.4,
                        }
                    }
                ),
                encoding="utf-8",
            )
            csdj = root / "qwen" / "rcs-kcd" / "external" / "csdj" / "summary.json"
            csdj.parent.mkdir(parents=True)
            csdj.write_text(
                json.dumps(
                    {
                        "representation_detector": {
                            "method": "rcs-kcd",
                            "protocol": "paper-rcs",
                            "scored_count": 100,
                            "labeled_count": 0,
                            "detection_rate": 0.7,
                            "tpr": None,
                            "fpr": None,
                        }
                    }
                ),
                encoding="utf-8",
            )
            rows = collect(root)
        self.assertEqual(len(rows), 2)
        by_source = {row["source"]: row for row in rows}
        self.assertAlmostEqual(by_source["xstest"]["false_negative_rate"], 0.2)
        self.assertAlmostEqual(by_source["csdj"]["false_negative_rate"], 0.3)
        self.assertEqual(by_source["csdj"]["unsafe_count"], 100)


if __name__ == "__main__":
    unittest.main()
