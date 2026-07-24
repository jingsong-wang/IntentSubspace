from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ood_intent_study.analyze import (
    _filter_analysis_rows,
    _validate_standard_split_labels,
)
from ood_intent_study.artifacts import ActivationTable
from ood_intent_study.io_utils import sha256_text
from ood_intent_study.panels import modality_panel_mask, source_display_name
from ood_intent_study.run import _panel_output_dir, _parse_modality_panels
from ood_intent_study.visualize import (
    SOURCE_STYLES,
    _add_source_display,
    _analysis_panel_selection,
)


def _activation_table() -> ActivationTable:
    count = 4
    return ActivationTable(
        sample_ids=np.array(["text-safe", "text-unsafe", "image-safe", "image-unsafe"]),
        activations=np.zeros((count, 1, 1, 3), dtype=np.float32),
        readout_valid=np.ones((count, 1), dtype=bool),
        layers=np.array([1]),
        readouts=np.array(["last"]),
        sequence_lengths=np.full(count, 8),
        image_token_counts=np.array([0, 0, 4, 4]),
        text_token_counts=np.full(count, 8),
        image_widths=np.array([0, 0, 32, 32]),
        image_heights=np.array([0, 0, 32, 32]),
        rendered_prompt_sha256=np.array(["a", "b", "c", "d"]),
        metadata={},
    )


def _panel_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["text-safe", "text-unsafe", "image-safe", "image-unsafe"],
            "source": ["XSTest", "XSTest", "MM-Vet", "HADES"],
            "modality": ["text", "text", "image_text", "image_text"],
            "label": [0, 1, 0, 1],
            "label_confidence": ["strong", "strong", "assumed", "derived"],
            "is_attack": [False, False, False, False],
            "split": ["train", "train", "train", "train"],
        }
    )


class ModalityPanelTests(unittest.TestCase):
    def test_panel_masks_and_joint_activation_filter(self) -> None:
        frame = _panel_frame()
        table = _activation_table()
        self.assertEqual(
            modality_panel_mask(frame["modality"], "text_only").tolist(),
            [True, True, False, False],
        )
        filtered, filtered_frame = _filter_analysis_rows(
            table, frame, "text_only", strong_label_sensitivity=True
        )
        self.assertEqual(filtered.sample_ids.tolist(), ["text-safe", "text-unsafe"])
        self.assertEqual(filtered_frame["sample_id"].tolist(), filtered.sample_ids.tolist())
        self.assertEqual(filtered.activations.shape[0], 2)

    def test_missing_standard_classes_fail_before_fitting(self) -> None:
        rows = []
        for split in ("train", "validation", "test"):
            rows.append({"is_attack": False, "split": split, "label": 1})
        frame = pd.DataFrame(rows)
        with self.assertRaisesRegex(ValueError, "every non-attack split must contain both labels"):
            _validate_standard_split_labels(frame, "multimodal_only", False)

    def test_visualization_recreates_panel_fingerprint(self) -> None:
        frame = _panel_frame()
        sample_ids = _activation_table().sample_ids
        expected = sha256_text("\n".join(sorted(sample_ids[:2].tolist())))
        analysis = {
            "modality_panel": "text_only",
            "coverage": {
                "modality_panel": "text_only",
                "strong_label_sensitivity": False,
                "analysis_sample_ids_sha256": expected,
            },
        }
        keep, panel, strong = _analysis_panel_selection(frame, sample_ids, analysis)
        self.assertEqual(keep.tolist(), [True, True, False, False])
        self.assertEqual(panel, "text_only")
        self.assertFalse(strong)

    def test_xstest_groups_and_source_styles_are_distinct(self) -> None:
        self.assertEqual(source_display_name("XSTest", 0), "XSTest-safe")
        self.assertEqual(source_display_name("XSTest", 1), "XSTest-unsafe")
        displayed = _add_source_display(_panel_frame())
        self.assertEqual(
            displayed.loc[displayed["source"].eq("XSTest"), "source_display"].tolist(),
            ["XSTest-safe", "XSTest-unsafe"],
        )
        colors = [style.color for style in SOURCE_STYLES.values()]
        self.assertEqual(len(colors), len(set(colors)))
        self.assertIn("VizWiz-VQA", SOURCE_STYLES)
        self.assertTrue(
            {
                "JailBreakV-28K-FigStep",
                "JailBreakV-28K-LLM-Transfer",
                "JailBreakV-28K-Query-Related",
            }.issubset(SOURCE_STYLES)
        )
        self.assertNotEqual(SOURCE_STYLES["XSTest-safe"], SOURCE_STYLES["XSTest-unsafe"])

    def test_source_style_validation_is_limited_to_selected_panel(self) -> None:
        frame = _panel_frame()
        frame.loc[2, "source"] = "future-image-source"
        text_mask = modality_panel_mask(frame["modality"], "text_only")
        displayed = _add_source_display(frame, text_mask)
        self.assertEqual(displayed.loc[2, "source_display"], "future-image-source")
        with self.assertRaisesRegex(ValueError, "future-image-source"):
            _add_source_display(frame)

    def test_panel_orchestrator_parsing_and_layout(self) -> None:
        panels = _parse_modality_panels("all,text_only,multimodal_only,text_only")
        self.assertEqual(panels, ["all", "text_only", "multimodal_only"])
        self.assertEqual(
            _panel_output_dir(Path("run"), "analysis", "text_only", "model", True),
            Path("run/analysis_panels/text_only/model"),
        )
        with self.assertRaisesRegex(ValueError, "Unknown modality panels"):
            _parse_modality_panels("audio_only")


if __name__ == "__main__":
    unittest.main()
