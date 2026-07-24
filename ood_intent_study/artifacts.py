from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import ARTIFACT_VERSION


SUPPORTED_ARTIFACT_VERSIONS = {
    "layerwise_activation_shard_v1",
    ARTIFACT_VERSION,
}

REQUIRED_KEYS = {
    "sample_ids",
    "activations",
    "readout_valid",
    "layers",
    "readouts",
    "sequence_lengths",
    "image_token_counts",
    "text_token_counts",
    "image_widths",
    "image_heights",
    "rendered_prompt_sha256",
    "metadata_json",
}


@dataclass
class ActivationTable:
    sample_ids: np.ndarray
    activations: np.ndarray
    readout_valid: np.ndarray
    layers: np.ndarray
    readouts: np.ndarray
    sequence_lengths: np.ndarray
    image_token_counts: np.ndarray
    text_token_counts: np.ndarray
    image_widths: np.ndarray
    image_heights: np.ndarray
    rendered_prompt_sha256: np.ndarray
    metadata: dict[str, Any]


def shard_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("shard_*.npz"))


def read_shard(path: Path) -> ActivationTable:
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_KEYS - set(data.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("artifact_version") not in SUPPORTED_ARTIFACT_VERSIONS:
            raise ValueError(
                f"Unsupported artifact version in {path}: {metadata.get('artifact_version')!r}"
            )
        raw_activations = data["activations"]
        storage_dtype = metadata.get("storage_dtype")
        if storage_dtype == "bfloat16":
            if raw_activations.dtype != np.uint16:
                raise ValueError(
                    f"{path} declares bfloat16 storage but contains {raw_activations.dtype}"
                )
            activations = _bfloat16_bits_to_float32(raw_activations)
        else:
            activations = raw_activations
        if not bool(np.isfinite(activations).all()):
            nonfinite = ~np.isfinite(activations)
            affected_rows = int(np.sum(nonfinite.reshape(len(activations), -1).any(axis=1)))
            raise ValueError(
                f"{path} contains {int(nonfinite.sum())} non-finite activation values "
                f"across {affected_rows} rows. This is commonly caused by legacy FP16 "
                "storage overflow; the lost values cannot be repaired. Re-extract to a new "
                "directory with --storage-dtype float32 or bfloat16."
            )
        return ActivationTable(
            sample_ids=data["sample_ids"].astype(str),
            activations=activations,
            readout_valid=data["readout_valid"].astype(bool),
            layers=data["layers"].astype(np.int32),
            readouts=data["readouts"].astype(str),
            sequence_lengths=data["sequence_lengths"].astype(np.int32),
            image_token_counts=data["image_token_counts"].astype(np.int32),
            text_token_counts=data["text_token_counts"].astype(np.int32),
            image_widths=data["image_widths"].astype(np.int32),
            image_heights=data["image_heights"].astype(np.int32),
            rendered_prompt_sha256=data["rendered_prompt_sha256"].astype(str),
            metadata=metadata,
        )


def _bfloat16_bits_to_float32(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def iter_shards(directory: Path) -> Iterator[ActivationTable]:
    paths = shard_paths(directory)
    if not paths:
        raise FileNotFoundError(f"No activation shards found in {directory}")
    for path in paths:
        yield read_shard(path)


def load_activation_table(directory: Path) -> ActivationTable:
    shards = list(iter_shards(directory))
    first = shards[0]
    for index, shard in enumerate(shards[1:], start=1):
        if not np.array_equal(shard.layers, first.layers):
            raise ValueError(f"Layer mismatch in shard index {index}")
        if not np.array_equal(shard.readouts, first.readouts):
            raise ValueError(f"Readout mismatch in shard index {index}")
        for field in ("manifest_sha256", "run_fingerprint", "runtime_identity_sha256", "model"):
            if shard.metadata.get(field) != first.metadata.get(field):
                raise ValueError(f"Metadata mismatch for {field!r} in shard index {index}")
    sample_ids = np.concatenate([shard.sample_ids for shard in shards])
    if len(sample_ids) != len(set(sample_ids.tolist())):
        raise ValueError("Activation shards contain duplicate sample IDs")
    return ActivationTable(
        sample_ids=sample_ids,
        activations=np.concatenate([shard.activations for shard in shards], axis=0),
        readout_valid=np.concatenate([shard.readout_valid for shard in shards], axis=0),
        layers=first.layers,
        readouts=first.readouts,
        sequence_lengths=np.concatenate([shard.sequence_lengths for shard in shards]),
        image_token_counts=np.concatenate([shard.image_token_counts for shard in shards]),
        text_token_counts=np.concatenate([shard.text_token_counts for shard in shards]),
        image_widths=np.concatenate([shard.image_widths for shard in shards]),
        image_heights=np.concatenate([shard.image_heights for shard in shards]),
        rendered_prompt_sha256=np.concatenate([shard.rendered_prompt_sha256 for shard in shards]),
        metadata=first.metadata,
    )


def completed_sample_ids(directory: Path) -> set[str]:
    completed: set[str] = set()
    for path in shard_paths(directory):
        with np.load(path, allow_pickle=False) as data:
            completed.update(data["sample_ids"].astype(str).tolist())
    return completed
