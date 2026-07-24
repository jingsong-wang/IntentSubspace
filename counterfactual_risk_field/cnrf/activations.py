from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ActivationData:
    sample_ids: np.ndarray
    activations: np.ndarray  # sample x readout x layer x hidden
    readout_valid: np.ndarray  # sample x readout
    layers: np.ndarray
    readouts: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        if self.activations.ndim != 4:
            raise ValueError(f"activations must be [sample, readout, layer, hidden], got {self.activations.shape}")
        n, readout_n, layer_n, _ = self.activations.shape
        if self.sample_ids.shape != (n,):
            raise ValueError("sample_ids do not align with activations")
        if self.readout_valid.shape != (n, readout_n):
            raise ValueError("readout_valid does not align with activations")
        if self.layers.shape != (layer_n,) or self.readouts.shape != (readout_n,):
            raise ValueError("layer/readout axes do not align with activations")
        if len(set(self.sample_ids.astype(str).tolist())) != n:
            raise ValueError("activation artifact contains duplicate sample ids")
        if not np.isfinite(self.activations).all():
            raise ValueError("activation artifact contains non-finite values")


def _decode_bfloat16(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def load_activation_data(
    path: Path | str,
    *,
    selected_readouts: Iterable[str] | None = None,
    selected_layers: Iterable[int] | None = None,
) -> ActivationData:
    source = Path(path).expanduser().resolve()
    readout_filter = (
        tuple(dict.fromkeys(str(value) for value in selected_readouts))
        if selected_readouts is not None
        else None
    )
    layer_filter = (
        tuple(dict.fromkeys(int(value) for value in selected_layers))
        if selected_layers is not None
        else None
    )
    if source.is_dir():
        data = _load_shards(
            source,
            selected_readouts=readout_filter,
            selected_layers=layer_filter,
        )
    elif source.suffix.lower() == ".npz":
        data = _load_legacy_npz(
            source,
            selected_readouts=readout_filter,
            selected_layers=layer_filter,
        )
    else:
        raise ValueError(f"Expected an activation shard directory or .npz archive, got {source}")
    data.validate()
    return data


def _axis_positions(
    available: np.ndarray,
    requested: tuple[Any, ...] | None,
    *,
    axis_name: str,
) -> np.ndarray:
    if requested is None:
        return np.arange(len(available), dtype=np.int64)
    positions: list[int] = []
    values = available.tolist()
    for value in requested:
        matches = [index for index, candidate in enumerate(values) if candidate == value]
        if len(matches) != 1:
            raise ValueError(
                f"Requested {axis_name} {value!r} is not present exactly once; "
                f"available={values}"
            )
        positions.append(matches[0])
    return np.asarray(positions, dtype=np.int64)


def _load_legacy_npz(
    path: Path,
    *,
    selected_readouts: tuple[str, ...] | None,
    selected_layers: tuple[int, ...] | None,
) -> ActivationData:
    with np.load(path, allow_pickle=True) as archive:
        values = np.asarray(archive["activations"])
        if values.ndim != 3:
            raise ValueError(f"Legacy activation array must be [sample, layer, hidden], got {values.shape}")
        ids_key = "ids" if "ids" in archive else "sample_ids"
        sample_ids = np.asarray(archive[ids_key]).astype(str)
        layers = np.asarray(archive["layers"], dtype=np.int32)
        metadata = (
            json.loads(str(np.asarray(archive["metadata_json"]).item()))
            if "metadata_json" in archive
            else {}
        )
        readout = str(metadata.get("pooling") or "last")
        readouts = np.asarray([readout]).astype(str)
        readout_positions = _axis_positions(
            readouts,
            selected_readouts,
            axis_name="readout",
        )
        layer_positions = _axis_positions(
            layers,
            selected_layers,
            axis_name="layer",
        )
        selected_values = values[:, layer_positions, :][:, None, :, :]
    return ActivationData(
        sample_ids=sample_ids,
        activations=selected_values[:, readout_positions, :, :].astype(
            np.float32, copy=False
        ),
        readout_valid=np.ones((len(sample_ids), len(readout_positions)), dtype=bool),
        layers=layers[layer_positions],
        readouts=readouts[readout_positions],
        metadata=metadata,
    )


def _load_shards(
    directory: Path,
    *,
    selected_readouts: tuple[str, ...] | None,
    selected_layers: tuple[int, ...] | None,
) -> ActivationData:
    paths = sorted(directory.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.npz files found in {directory}")
    ids: list[np.ndarray] = []
    activations: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    first_layers: np.ndarray | None = None
    first_readouts: np.ndarray | None = None
    first_metadata: dict[str, Any] | None = None
    readout_positions: np.ndarray | None = None
    layer_positions: np.ndarray | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            raw = np.asarray(archive["activations"])
            value = _decode_bfloat16(raw) if metadata.get("storage_dtype") == "bfloat16" else raw
            layers = np.asarray(archive["layers"], dtype=np.int32)
            readouts = np.asarray(archive["readouts"]).astype(str)
            if first_layers is None:
                first_layers, first_readouts, first_metadata = layers, readouts, metadata
                readout_positions = _axis_positions(
                    readouts,
                    selected_readouts,
                    axis_name="readout",
                )
                layer_positions = _axis_positions(
                    layers,
                    selected_layers,
                    axis_name="layer",
                )
            elif not np.array_equal(layers, first_layers) or not np.array_equal(readouts, first_readouts):
                raise ValueError(f"Activation shard axes disagree in {path}")
            assert readout_positions is not None and layer_positions is not None
            ids.append(np.asarray(archive["sample_ids"]).astype(str))
            activations.append(
                value[:, readout_positions, :, :][:, :, layer_positions, :].astype(
                    np.float32, copy=False
                )
            )
            valid.append(
                np.asarray(archive["readout_valid"], dtype=bool)[:, readout_positions]
            )
    assert (
        first_layers is not None
        and first_readouts is not None
        and first_metadata is not None
        and readout_positions is not None
        and layer_positions is not None
    )
    return ActivationData(
        sample_ids=np.concatenate(ids),
        activations=np.concatenate(activations, axis=0),
        readout_valid=np.concatenate(valid, axis=0),
        layers=first_layers[layer_positions],
        readouts=first_readouts[readout_positions],
        metadata=first_metadata,
    )


def align_rows(rows: list[dict[str, Any]], data: ActivationData) -> np.ndarray:
    from .schema import row_id

    position = {value: index for index, value in enumerate(data.sample_ids.astype(str).tolist())}
    missing = [row_id(row) for row in rows if row_id(row) not in position]
    if missing:
        raise ValueError(f"Activation artifact is missing {len(missing)} manifest rows: {missing[:20]}")
    return np.asarray([position[row_id(row)] for row in rows], dtype=np.int64)
