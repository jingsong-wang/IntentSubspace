from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from ood_intent_study.data import (
    RawRecord,
    assign_component_splits,
    load_source_records,
    normalize_record,
    preselect_records,
)
from ood_intent_study.audit_manifest import main as audit_main
from ood_intent_study.io_utils import (
    canonical_json,
    load_config,
    repo_root,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)
from ood_intent_study.schema import StudySample


def sample(name: str, source: str, semantic: str, group: str, attack: bool = False) -> StudySample:
    return StudySample(
        sample_id=name,
        source=source,
        source_kind="attack" if attack else "benchmark",
        source_record_id=name,
        label=1 if attack else 0,
        label_name="harmful" if attack else "benign",
        label_semantics="test",
        label_confidence="strong",
        label_provenance="explicit",
        source_role="user",
        prompt_text=name,
        semantic_text=semantic,
        image_path=None,
        image_exists=False,
        modality="text",
        category="",
        variant="",
        group_id=group,
        semantic_group_id=semantic,
        split_group_id=semantic,
        nuisance_group_id=group,
        split="external" if attack else "train",
        is_attack=attack,
        attack_name=source if attack else "",
        source_file="fixture.jsonl",
        source_row=0,
        prompt_sha256=name,
        semantic_sha256=semantic,
        image_sha256="",
    )


class ManifestTests(unittest.TestCase):
    def test_connected_split_groups_keep_group_and_semantic_links(self) -> None:
        rows = [
            sample("a", "A", "semantic:x", "group:1"),
            sample("b", "A", "semantic:y", "group:1"),
            sample("c", "B", "semantic:y", "group:2"),
            sample("d", "Attack", "semantic:y", "group:3", attack=True),
        ]
        assigned = assign_component_splits(
            rows,
            seed=7,
            fractions={"train": 0.7, "validation": 0.15, "test": 0.15},
        )
        self.assertEqual(len({row.split_group_id for row in assigned[:3]}), 1)
        self.assertEqual(len({row.split for row in assigned[:3]}), 1)
        self.assertEqual(assigned[3].split, "external")
        self.assertNotEqual(assigned[3].split_group_id, assigned[0].split_group_id)
        self.assertEqual(assigned[3].semantic_group_id, assigned[1].semantic_group_id)

    def test_figstep_adapter_retains_distinct_images(self) -> None:
        root = repo_root()
        config = load_config(root / "ood_intent_study" / "configs" / "default.json")
        spec = next(value for value in config["attacks"] if value["name"] == "FigStep")
        records, _ = load_source_records(root, spec)
        selected, duplicates = preselect_records(records, spec, maximum_per_label=3, seed=11)
        self.assertEqual(duplicates, 0)
        self.assertEqual(len(selected), 3)
        with tempfile.TemporaryDirectory() as directory:
            normalized = [
                normalize_record(
                    record,
                    root=root,
                    spec=spec,
                    seed=11,
                    fractions=config["split_fractions"],
                    asset_dir=Path(directory),
                    search_dirs=[],
                    external=True,
                )
                for record in selected
            ]
        self.assertEqual(len({row.sample_id for row in normalized}), 3)
        self.assertTrue(all(row.image_exists and row.modality == "image_text" for row in normalized))

    def test_vizwiz_vqa_asset_identity_and_context_aware_groups(self) -> None:
        root = repo_root()
        config = load_config(root / "ood_intent_study" / "configs" / "default.json")
        spec = next(value for value in config["datasets"] if value["name"] == "VizWiz-VQA")
        records, provenance = load_source_records(root, spec)
        self.assertEqual(len(records), 1000)
        self.assertEqual(provenance["selected_rows"], 1000)
        self.assertTrue(
            all(str(record.row["question_id"]).startswith("VizWiz_val_") for record in records)
        )

        first = records[0]
        second = records[1]
        fixtures = [
            RawRecord(
                {
                    **first.row,
                    "question_id": "VizWiz_fixture_a",
                    "image_id": "shared-image",
                    "txt": "What is this?",
                },
                first.source_file,
                0,
            ),
            RawRecord(
                {
                    **first.row,
                    "question_id": "VizWiz_fixture_b",
                    "image_id": "shared-image",
                    "txt": "What color is it?",
                },
                first.source_file,
                1,
            ),
            RawRecord(
                {
                    **second.row,
                    "question_id": "VizWiz_fixture_c",
                    "image_id": "different-image",
                    "txt": "What is this?",
                },
                second.source_file,
                2,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            normalized = [
                normalize_record(
                    record,
                    root=root,
                    spec=spec,
                    seed=11,
                    fractions=config["split_fractions"],
                    asset_dir=Path(directory),
                    search_dirs=[],
                    external=False,
                )
                for record in fixtures
            ]
        self.assertTrue(
            all(
                row.source == "VizWiz-VQA"
                and row.label == 0
                and row.label_confidence == "assumed"
                and row.modality == "image_text"
                and row.image_exists
                and row.metadata["dataset_identity"] == "VizWiz-VQA"
                and "answers" not in row.metadata
                for row in normalized
            )
        )
        self.assertEqual(normalized[0].group_id, normalized[1].group_id)
        self.assertNotEqual(normalized[0].semantic_group_id, normalized[2].semantic_group_id)
        assigned = assign_component_splits(
            normalized,
            seed=11,
            fractions=config["split_fractions"],
        )
        self.assertEqual(assigned[0].split_group_id, assigned[1].split_group_id)
        self.assertEqual(assigned[0].split, assigned[1].split)

    def test_jailbreakv_28k_attack_conditions_are_external_and_distinct(self) -> None:
        root = repo_root()
        config = load_config(root / "ood_intent_study" / "configs" / "default.json")
        expected = {
            "JailBreakV-28K-FigStep": ("figstep", 2000, {"figstep"}),
            "JailBreakV-28K-LLM-Transfer": (
                "llm_transfer_attack",
                20000,
                {"SD", "blank", "nature", "noise"},
            ),
            "JailBreakV-28K-Query-Related": (
                "query_related",
                6000,
                {"SD", "typo"},
            ),
        }
        normalized_by_source: dict[str, list[StudySample]] = {}
        with tempfile.TemporaryDirectory() as directory:
            for name, (attack_type, expected_rows, expected_styles) in expected.items():
                spec = next(value for value in config["attacks"] if value["name"] == name)
                records, provenance = load_source_records(root, spec)
                self.assertEqual(len(records), expected_rows)
                self.assertEqual(provenance["attack_type"], attack_type)
                self.assertEqual(set(provenance["by_image_style"]), expected_styles)
                selected, _ = preselect_records(
                    records, spec, maximum_per_label=4, seed=11
                )
                normalized_by_source[name] = [
                    normalize_record(
                        record,
                        root=root,
                        spec=spec,
                        seed=11,
                        fractions=config["split_fractions"],
                        asset_dir=Path(directory),
                        search_dirs=[],
                        external=True,
                    )
                    for record in selected
                ]
        for name, rows in normalized_by_source.items():
            self.assertEqual(len(rows), 4)
            self.assertTrue(
                all(
                    row.source == name
                    and row.label == 1
                    and row.label_confidence == "strong"
                    and row.modality == "image_text"
                    and row.image_exists
                    and row.is_attack
                    and row.split == "external"
                    and row.metadata["dataset_identity"] == "JailBreakV-28K"
                    and row.metadata["attack_type"]
                    and row.metadata["image_style"]
                    for row in rows
                )
            )
            self.assertTrue(all(row.prompt_text != row.semantic_text for row in rows))

    def test_config_declares_all_requested_sources(self) -> None:
        config = json.loads(
            (repo_root() / "ood_intent_study" / "configs" / "default.json").read_text(encoding="utf-8")
        )
        names = {row["name"] for row in config["datasets"] + config["attacks"]}
        self.assertEqual(
            names,
            {
                "AdvBench",
                "Alpaca",
                "DAN-Prompts",
                "OpenAssistant",
                "XSTest",
                "HADES",
                "MM-SafetyBench",
                "MM-Vet",
                "VizWiz-VQA",
                "FigStep",
                "JOOD",
                "CS-DJ",
                "JailBreakV-28K-FigStep",
                "JailBreakV-28K-LLM-Transfer",
                "JailBreakV-28K-Query-Related",
            },
        )

    def test_prepared_attack_sidecar_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "prepared" / "samples.jsonl"
            rows = [{"id": "one", "prompt_text": "carrier", "instruction": "goal"}]
            write_jsonl_atomic(manifest, rows)
            write_json_atomic(
                manifest.parent / "prepare.json",
                {
                    "logical_sha256": sha256_text("\n".join(canonical_json(row) for row in rows)),
                    "protocol_sha256": "protocol-hash",
                    "protocol": {
                        "attack": "CS-DJ",
                        "protocol_name": "CS-DJ-100",
                        "retrieval_pool": 100,
                        "selected_distraction_images": 9,
                    },
                },
            )
            spec = {
                "name": "CS-DJ",
                "reader": "best_run_jsonl",
                "preferred_path": "prepared/samples.jsonl",
                "require_preferred": True,
                "expected_protocol": {
                    "retrieval_pool": 100,
                    "selected_distraction_images": 9,
                },
            }
            records, audit = load_source_records(root, spec)
            self.assertEqual(len(records), 1)
            self.assertEqual(audit["selection_reason"], "verified_prepared_assets")
            self.assertEqual(records[0].row["prepared_protocol_name"], "CS-DJ-100")
            spec["expected_protocol"] = {"retrieval_pool": 10_000}
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                load_source_records(root, spec)

    def test_manifest_audit_checks_sidecar_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "samples.jsonl"
            rows = [sample("one", "source", "semantic", "group").to_dict()]
            write_jsonl_atomic(manifest, rows)
            fingerprint = sha256_text("\n".join(canonical_json(row) for row in rows))
            write_json_atomic(manifest.with_suffix(".manifest.json"), {"manifest_sha256": fingerprint})
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    audit_main(["--manifest", str(manifest), "--repo-root", str(root), "--require-sidecar"]),
                    0,
                )
            rows[0]["prompt_text"] = "changed"
            write_jsonl_atomic(manifest, rows)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    audit_main(["--manifest", str(manifest), "--repo-root", str(root), "--require-sidecar"]),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
