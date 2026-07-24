from __future__ import annotations

import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intentguard.dataset_v3 import (
    build_dataset,
    classify_visual_semantic_role,
    render_clean_ocr_pair,
)


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 96), color).save(path)


class CISRV3DatasetTest(unittest.TestCase):
    def test_visual_role_uses_semantics_not_document_layout(self) -> None:
        self.assertEqual(
            classify_visual_semantic_role(Path("auth_doc.png")), "evidence_semantic"
        )
        self.assertEqual(
            classify_visual_semantic_role(Path("danger2.png")), "action_semantic"
        )
        self.assertEqual(
            classify_visual_semantic_role(Path("scenario.png")), "action_semantic"
        )
        self.assertEqual(
            classify_visual_semantic_role(Path("query_render.png")), "text_carrier_ocr"
        )

    def test_clean_ocr_pair_is_complete_and_matched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.png"
            benign = root / "benign.png"
            result = render_clean_ocr_pair(
                target,
                benign,
                "How can someone perform a sensitive action in a realistic setting?",
                "How can a responder prevent and report that sensitive action safely?",
                layout_index=1,
                width=640,
                height=640,
                preferred_size=32,
                minimum_size=18,
                font_path=None,
            )
            self.assertTrue(result.complete)
            self.assertGreater(result.font_size, 0)
            with Image.open(target) as target_image, Image.open(benign) as benign_image:
                self.assertEqual(target_image.size, benign_image.size)
                self.assertEqual(target_image.size, (640, 640))

    def test_builder_uses_existing_semantics_and_isolates_v3_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic = root / "imgs" / "topic"
            general = root / "imgs" / "general"
            write_image(topic / "danger.png", (150, 20, 20))
            write_image(topic / "danger2.png", (120, 30, 30))
            write_image(topic / "auth_doc.png", (230, 230, 220))
            write_image(topic / "auth_doc2.png", (220, 225, 230))
            write_image(general / "neutral.png", (20, 130, 70))
            config = {
                "intents": [
                    {
                        "intent_id": "test_intent",
                        "family": "test_family",
                        "target_intent_text": "perform a prohibited operation",
                        "benign_intent_text": "prevent and report the prohibited operation",
                        "image_dir": "topic",
                    }
                ],
                "dataset": {
                    "text_variants_per_distribution": 6,
                    "semantic_query_variants": 6,
                    "semantic_views_per_query": 2,
                    "ocr_variants": 6,
                    "mixed_query_variants": 6,
                    "mixed_query_views": 3,
                },
                "images": {
                    "root": "imgs",
                    "general_dir": "general",
                    "v3_generated_subdir": "_cisr_v3_generated",
                    "width": 640,
                    "height": 640,
                    "font_size": 30,
                    "minimum_font_size": 16,
                },
            }

            rows = build_dataset(config, root)

            evidence = [row for row in rows if "auth_doc" in str(row.get("image_path"))]
            self.assertTrue(evidence)
            self.assertTrue(
                all(row["visual_semantic_role"] == "evidence_semantic" for row in evidence)
            )
            semantic = [
                row
                for row in rows
                if row["visual_semantic_role"] in {"action_semantic", "evidence_semantic"}
            ]
            self.assertTrue(all(row["image_source"].startswith("existing_") for row in semantic))
            ocr = [row for row in rows if row["visual_semantic_role"] == "text_carrier_ocr"]
            self.assertTrue(ocr)
            self.assertTrue(all("/_cisr_v3_generated/ocr/" in row["image_path"] for row in ocr))
            self.assertTrue(all(row["ocr_render_complete"] is True for row in ocr))
            self.assertFalse(any("/_intentguard_generated/" in str(row.get("image_path")) for row in rows))
            self.assertFalse(
                any("toward provide" in str(row.get("prompt_text", "")) for row in rows)
            )

            labels_by_pair: dict[str, list[int]] = defaultdict(list)
            splits_by_view_group: dict[str, set[str]] = defaultdict(set)
            for row in rows:
                labels_by_pair[row["pair_key"]].append(int(row["label"]))
                if row["view_group"]:
                    splits_by_view_group[row["view_group"]].add(row["evaluation_split"])
            self.assertTrue(all(sorted(labels) == [0, 1] for labels in labels_by_pair.values()))
            self.assertTrue(all(len(splits) == 1 for splits in splits_by_view_group.values()))
            self.assertEqual(
                {row["evaluation_split"] for row in rows},
                {"train", "validation", "calibration", "test"},
            )


if __name__ == "__main__":
    unittest.main()
