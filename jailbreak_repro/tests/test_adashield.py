from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from jailbreak_repro.adashield import (
    ADASHIELD_PAPER_CLIP_MODEL,
    ADASHIELD_POOL_FORMAT,
    AdaShieldConfig,
    build_pool_from_official_tables,
    compose_adashield_prompt,
    prepare_adashield_samples,
)
from jailbreak_repro.defenses import run_defense
from jailbreak_repro.io_utils import write_json
from jailbreak_repro.models import MockModelRunner
from jailbreak_repro.train_adashield import _extract_candidate


class AdaShieldTest(unittest.TestCase):
    def test_defender_candidate_parser_accepts_released_json_shape(self) -> None:
        parsed = _extract_candidate(
            'prefix {"improvement":"add a specific rule","prompt":"refuse unsafe image text"} suffix'
        )
        self.assertEqual(parsed, {"improvement": "add a specific rule", "prompt": "refuse unsafe image text"})

    def test_released_prompt_construction_duplicates_query(self) -> None:
        self.assertEqual(compose_adashield_prompt("Q", "P"), "QPQ")

    def test_static_mode_uses_the_supplied_prompt_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_path = root / "prompts" / "static_defense_prompt.txt"
            prompt_path.parent.mkdir(parents=True)
            prompt_path.write_text("official static shield", encoding="utf-8")
            samples = [{"id": "1", "prompt_text": "question", "image_path": "image.png"}]
            prepared = prepare_adashield_samples(
                samples,
                AdaShieldConfig(
                    source_dir=root,
                    artifact_dir=root / "artifacts",
                    mode="static",
                    static_prompt_file=prompt_path,
                ),
            )

        self.assertEqual(prepared[0]["adashield_variant"], "AdaShield-S")
        self.assertEqual(prepared[0]["adashield_defense_prompt"], "official static shield")
        self.assertTrue(prepared[0]["adashield_prompt_applied"])

    def test_modified_static_prompt_is_not_marked_as_paper_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            official = root / "prompts" / "static_defense_prompt.txt"
            official.parent.mkdir(parents=True)
            official.write_text("official", encoding="utf-8")
            custom = root / "custom.txt"
            custom.write_text("modified", encoding="utf-8")
            prepared = prepare_adashield_samples(
                [{"id": "1", "prompt_text": "q", "image_path": "image.png"}],
                AdaShieldConfig(
                    source_dir=root,
                    artifact_dir=root / "artifacts",
                    mode="static",
                    static_prompt_file=custom,
                ),
            )
        self.assertFalse(prepared[0]["adashield_paper_configuration"])

    def test_static_defense_passes_query_shield_query_to_victim(self) -> None:
        row = run_defense(
            defense="adashield",
            sample={
                "id": "1",
                "prompt_text": "Q",
                "image_path": "",
                "adashield_preprocessed": True,
                "adashield_variant": "AdaShield-S",
                "adashield_defense_prompt": "P",
                "adashield_prompt_applied": True,
                "adashield_paper_configuration": True,
            },
            runner=MockModelRunner(),
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.9,
        )
        self.assertEqual(row["model_prompt"], "QPQ")
        self.assertEqual(row["defense"], "adashield")
        self.assertFalse(row["paper_claim_compatible"])

    def test_adaptive_mode_never_falls_back_without_a_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "requires --adashield-prompt-pool"):
                prepare_adashield_samples(
                    [{"id": "1", "prompt_text": "q", "image_path": ""}],
                    AdaShieldConfig(
                        source_dir=root,
                        artifact_dir=root / "artifacts",
                        mode="adaptive",
                    ),
                )

    def test_adaptive_pool_is_bound_to_the_training_victim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "anchor.png"
            Image.new("RGB", (8, 8)).save(image_path)
            pool_path = root / "pool.json"
            write_json(
                pool_path,
                {
                    "format_version": ADASHIELD_POOL_FORMAT,
                    "victim_model": "victim-a",
                    "clip_model": ADASHIELD_PAPER_CLIP_MODEL,
                    "entries": [
                        {
                            "id": "one",
                            "query": "query",
                            "image_path": str(image_path),
                            "defense_prompt": "shield",
                        }
                    ],
                },
            )
            with self.assertRaisesRegex(ValueError, "trained for 'victim-a'"):
                prepare_adashield_samples(
                    [{"id": "1", "prompt_text": "q", "image_path": str(image_path)}],
                    AdaShieldConfig(
                        source_dir=root,
                        artifact_dir=root / "artifacts",
                        mode="adaptive",
                        prompt_pool=pool_path,
                        victim_model="victim-b",
                    ),
                )

    def test_official_tables_are_filtered_into_standard_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            image_path = source / "data" / "anchor.png"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(image_path)
            table = root / "tables" / "scenario" / "final_table.csv"
            table.parent.mkdir(parents=True)
            with table.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["final_judge_scores", "defense_prompt_list", "query", "image"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "final_judge_scores": 1,
                        "defense_prompt_list": "shield",
                        "query": "query",
                        "image": "data/anchor.png",
                    }
                )
                writer.writerow(
                    {
                        "final_judge_scores": 10,
                        "defense_prompt_list": "failed",
                        "query": "query",
                        "image": "data/anchor.png",
                    }
                )
            output = root / "pool.json"
            payload = build_pool_from_official_tables(
                table.parent.parent,
                output,
                victim_model="victim",
                source_dir=source,
            )

        self.assertEqual(payload["format_version"], ADASHIELD_POOL_FORMAT)
        self.assertEqual(payload["entry_count"], 1)
        self.assertEqual(payload["entries"][0]["defense_prompt"], "shield")


if __name__ == "__main__":
    unittest.main()
