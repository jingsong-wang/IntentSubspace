from __future__ import annotations

from collections.abc import Iterable

import numpy as np


MODALITY_PANELS = ("all", "text_only", "multimodal_only")
PANEL_MODALITY = {
    "all": None,
    "text_only": "text",
    "multimodal_only": "image_text",
}
PANEL_TITLES = {
    "all": "all modalities",
    "text_only": "text only",
    "multimodal_only": "multimodal only",
}


def modality_panel_mask(modalities: Iterable[object], panel: str) -> np.ndarray:
    if panel not in PANEL_MODALITY:
        raise ValueError(f"Unknown modality panel {panel!r}; expected one of {MODALITY_PANELS}")
    values = np.asarray(list(modalities)).astype(str)
    modality = PANEL_MODALITY[panel]
    return np.ones(len(values), dtype=bool) if modality is None else values == modality


def source_display_name(source: object, label: object) -> str:
    name = str(source)
    if name != "XSTest":
        return name
    return "XSTest-unsafe" if int(label) == 1 else "XSTest-safe"
