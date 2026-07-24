from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intentguard.dataset_v4 import build_dataset, render_figstep_style_pair  # noqa: E402


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 96), color).save(path)


class CISRV4DatasetTest(unittest.TestCase):
    def test_figstep_style_renderer_produces_matched_clean_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.png"
            benign = root / "benign.png"
            result = render_figstep_style_pair(
                target,
                benign,
                "Steps to perform a prohibited operation.",
                "Steps to prevent and report a prohibited operation.",
                layout_index=0,
                width=640,
                height=640,
                preferred_size=32,
                minimum_size=18,
                font_path=None,
            )
            self.assertTrue(result.complete)
            with Image.open(target) as left, Image.open(benign) as right:
                self.assertEqual(left.size, right.size)
                self.assertEqual(left.size, (640, 640))

    def test_builder_isolates_modalities_and_adds_paired_figstep_style_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic = root / "imgs" / "topic"
            general = root / "imgs" / "general"
            write_image(topic / "danger.png", (150, 20, 20))
            write_image(topic / "auth_doc.png", (230, 230, 220))
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
                    "semantic_views_per_query": 1,
                    "ocr_variants": 6,
                    "mixed_query_variants": 6,
                    "mixed_query_views": 2,
                    "figstep_style_ocr_variants": 6,
                },
                "images": {
                    "root": "imgs",
                    "general_dir": "general",
                    "v3_generated_subdir": "_cisr_v4_generated",
                    "v4_generated_subdir": "_cisr_v4_generated",
                    "width": 640,
                    "height": 640,
                    "font_size": 30,
                    "minimum_font_size": 16,
                },
            }

            rows = build_dataset(config, root)

            styled = [
                row
                for row in rows
                if row.get("attack_style_family") == "typographic_list_completion"
            ]
            self.assertEqual(len(styled), 12)
            self.assertEqual({int(row["label"]) for row in styled}, {0, 1})
            self.assertTrue(all(row["ocr_render_complete"] for row in styled))
            self.assertTrue(
                all("/_cisr_v4_generated/figstep_style_ocr/" in row["image_path"] for row in styled)
            )
            text = [row for row in rows if row["modality_branch"] == "text"]
            multimodal = [row for row in rows if row["modality_branch"] == "multimodal"]
            self.assertTrue(text)
            self.assertTrue(multimodal)
            self.assertTrue(all(not row.get("image_path") for row in text))
            self.assertTrue(all(row.get("image_path") for row in multimodal))
            self.assertEqual(
                {row["evaluation_split"] for row in styled},
                {"train", "validation", "calibration", "test"},
            )


if __name__ == "__main__":
    unittest.main()
