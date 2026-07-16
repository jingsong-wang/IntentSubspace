from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REQUIRED_ACTIVATION_FIELDS = frozenset(
    {
        "activations",
        "layers",
        "ids",
        "labels",
        "conditions",
        "pair_keys",
        "image_roles",
        "metadata_json",
    }
)
MULTIMODAL_ANCHOR_FIELDS = frozenset(
    {
        "anchor_activations",
        "has_anchor",
    }
)


def _first_mismatch(actual: list[Any], expected: list[Any]) -> str:
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
        if actual_value != expected_value:
            return f"index {index}: archive={actual_value!r}, dataset={expected_value!r}"
    return f"length: archive={len(actual)}, dataset={len(expected)}"


def activation_archive_errors(
    path: Path,
    *,
    expected_rows: Iterable[dict[str, Any]] | None = None,
    require_multimodal_anchor: bool = False,
    expected_model: str | None = None,
    expected_backend: str | None = None,
) -> list[str]:
    """Return compatibility errors without loading the large activation tensor."""
    path = Path(path)
    if not path.is_file():
        return ["archive does not exist"]
    if path.stat().st_size == 0:
        return ["archive is empty"]

    rows = list(expected_rows) if expected_rows is not None else None
    required = set(REQUIRED_ACTIVATION_FIELDS)
    if require_multimodal_anchor:
        required.update(MULTIMODAL_ANCHOR_FIELDS)

    errors: list[str] = []
    try:
        with np.load(path, allow_pickle=True) as archive:
            available = set(archive.files)
            missing = sorted(required - available)
            if missing:
                errors.append(f"missing required fields: {', '.join(missing)}")
                return errors

            ids_array = np.asarray(archive["ids"])
            labels_array = np.asarray(archive["labels"])
            layers_array = np.asarray(archive["layers"])
            try:
                metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            except Exception as exc:
                errors.append(f"metadata_json is invalid: {type(exc).__name__}: {exc}")
                metadata = {}
            if ids_array.ndim != 1:
                errors.append(f"ids must be one-dimensional, got shape={ids_array.shape}")
            if labels_array.ndim != 1:
                errors.append(f"labels must be one-dimensional, got shape={labels_array.shape}")
            if layers_array.ndim != 1 or layers_array.size == 0:
                errors.append(f"layers must be a non-empty one-dimensional array, got shape={layers_array.shape}")

            if ids_array.ndim == 1 and labels_array.ndim == 1 and len(ids_array) != len(labels_array):
                errors.append(
                    f"ids/labels length mismatch: ids={len(ids_array)}, labels={len(labels_array)}"
                )

            if require_multimodal_anchor:
                if metadata.get("multimodal_anchor") is not True:
                    errors.append("metadata does not confirm multimodal-anchor extraction")
                has_anchor = np.asarray(archive["has_anchor"])
                if has_anchor.ndim != 1:
                    errors.append(
                        f"has_anchor must be one-dimensional, got shape={has_anchor.shape}"
                    )
                elif ids_array.ndim == 1 and len(has_anchor) != len(ids_array):
                    errors.append(
                        f"ids/has_anchor length mismatch: ids={len(ids_array)}, "
                        f"has_anchor={len(has_anchor)}"
                    )

            if expected_model is not None and metadata.get("model") != expected_model:
                errors.append(
                    f"model mismatch: archive={metadata.get('model')!r}, expected={expected_model!r}"
                )
            if expected_backend is not None and metadata.get("backend") != expected_backend:
                errors.append(
                    f"backend mismatch: archive={metadata.get('backend')!r}, "
                    f"expected={expected_backend!r}"
                )

            if rows is not None and ids_array.ndim == 1:
                expected_ids = [str(row["id"]) for row in rows]
                actual_ids = ids_array.astype(str).tolist()
                if actual_ids != expected_ids:
                    errors.append(
                        "sample ids do not exactly match the current dataset ("
                        + _first_mismatch(actual_ids, expected_ids)
                        + ")"
                    )

                expected_labels = [int(row["label"]) for row in rows]
                if labels_array.ndim == 1:
                    actual_labels = labels_array.astype(int).tolist()
                    if actual_labels != expected_labels:
                        errors.append(
                            "labels do not exactly match the current dataset ("
                            + _first_mismatch(actual_labels, expected_labels)
                            + ")"
                        )
    except Exception as exc:
        errors.append(f"archive cannot be read: {type(exc).__name__}: {exc}")
    return errors


def format_activation_archive_error(path: Path, errors: Iterable[str]) -> str:
    details = "\n".join(f"  - {error}" for error in errors)
    return (
        f"Incompatible activation archive: {path}\n{details}\n"
        "Re-run the activation extraction stage with the current code. The archive "
        "cannot be migrated safely when sample ids are absent because row alignment "
        "cannot be verified."
    )
