from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jailbreak_repro.attacks import (
    _infer_csdj_aux_model_source,
    _resolve_csdj_image_dir,
    _resolve_csdj_selected_image,
    load_csdj_samples,
)


class CSDJPathResolutionTest(unittest.TestCase):
    def test_qwen_auxiliary_model_uses_modelscope_in_auto_mode(self) -> None:
        self.assertEqual(
            _infer_csdj_aux_model_source("Qwen/Qwen2.5-3B-Instruct", "auto"),
            "modelscope",
        )
        self.assertEqual(
            _infer_csdj_aux_model_source("Qwen/Qwen2.5-3B-Instruct", "hf"),
            "hf",
        )

    def test_prefers_repository_data_images_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            data_images = source / "data" / "images"
            legacy_images = source / "llava_images"
            data_images.mkdir(parents=True)
            legacy_images.mkdir()

            self.assertEqual(_resolve_csdj_image_dir(source, None), data_images.resolve())

    def test_falls_back_to_official_legacy_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            legacy_images = source / "llava_images"
            legacy_images.mkdir()

            self.assertEqual(_resolve_csdj_image_dir(source, None), legacy_images.resolve())

    def test_explicit_image_directory_has_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            override = Path(tmp) / "external-images"
            (source / "data" / "images").mkdir(parents=True)
            override.mkdir()

            self.assertEqual(_resolve_csdj_image_dir(source, override), override.resolve())

    def test_relocates_absolute_map_paths_by_basename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            image_dir = source / "data" / "images"
            image_dir.mkdir(parents=True)
            image = image_dir / "sample.jpg"
            image.touch()

            old_server_path = "/old/server/CS-DJ-main/llava_images/sample.jpg"
            self.assertEqual(
                _resolve_csdj_selected_image(old_server_path, image_dir, source),
                image.resolve(),
            )

    def test_builds_official_twelve_panel_attack_image(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "CS-DJ-main"
            instructions_dir = source / "instructions"
            image_dir = source / "data" / "images"
            instructions_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)

            instruction = "Test instruction for CS-DJ composition."
            (instructions_dir / "Test.json").write_text(
                json.dumps([{"instruction": instruction, "keywords": "test"}]),
                encoding="utf-8",
            )

            image_names = []
            for index in range(9):
                image_name = f"source_{index}.jpg"
                Image.new("RGB", (32, 24), color=(index * 20, 80, 160)).save(image_dir / image_name)
                image_names.append(image_name)

            image_map_path = root / "image-map.json"
            image_map_path.write_text(json.dumps({instruction: image_names}), encoding="utf-8")
            subquestions_path = root / "subquestions.json"
            subquestions_path.write_text(
                json.dumps({instruction: ["First test", "Second test", "Third test"]}),
                encoding="utf-8",
            )

            samples = load_csdj_samples(
                source_dir=source,
                max_samples=1,
                artifact_dir=root / "artifacts",
                image_map_path=image_map_path,
                subquestions_file=subquestions_path,
                aux_model="none",
            )

            self.assertEqual(len(samples), 1)
            output_path = Path(samples[0]["image_path"])
            self.assertTrue(output_path.is_file())
            with Image.open(output_path) as output:
                self.assertEqual(output.size, (1500, 2120))
            self.assertEqual(len(samples[0]["selected_distraction_images"]), 9)
            self.assertEqual(samples[0]["sub_question_list"], ["First test", "Second test", "Third test"])

            first_mtime = output_path.stat().st_mtime_ns
            resumed = load_csdj_samples(
                source_dir=source,
                max_samples=1,
                artifact_dir=root / "artifacts",
                image_map_path=image_map_path,
                subquestions_file=subquestions_path,
                aux_model="none",
            )
            self.assertEqual(Path(resumed[0]["image_path"]).stat().st_mtime_ns, first_mtime)


if __name__ == "__main__":
    unittest.main()
