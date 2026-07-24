from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import _bfloat16_bits_to_float32, shard_paths
from .io_utils import write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit activation shards for numeric corruption.")
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def audit_activation_shards(directory: Path) -> dict[str, Any]:
    paths = shard_paths(directory)
    if not paths:
        raise FileNotFoundError(f"No activation shards found in {directory}")

    dtype_counts: Counter[str] = Counter()
    storage_counts: Counter[str] = Counter()
    nonfinite_locations: Counter[str] = Counter()
    contaminated: list[dict[str, Any]] = []
    affected_ids: list[str] = []
    total_rows = 0
    total_values = 0
    nan_values = 0
    positive_inf_values = 0
    negative_inf_values = 0
    maximum_finite_abs = 0.0

    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            raw = data["activations"]
            metadata = json.loads(str(data["metadata_json"].item()))
            storage_dtype = str(metadata.get("storage_dtype") or raw.dtype)
            activations = (
                _bfloat16_bits_to_float32(raw)
                if storage_dtype == "bfloat16" and raw.dtype == np.uint16
                else raw
            )
            sample_ids = data["sample_ids"].astype(str)
            readouts = data["readouts"].astype(str)
            layers = data["layers"].astype(int)

        finite = np.isfinite(activations)
        row_bad = ~finite.reshape(len(activations), -1).all(axis=1)
        shard_nan = int(np.isnan(activations).sum())
        shard_positive_inf = int(np.isposinf(activations).sum())
        shard_negative_inf = int(np.isneginf(activations).sum())
        finite_values = activations[finite]
        shard_max = float(np.max(np.abs(finite_values))) if len(finite_values) else 0.0

        total_rows += len(activations)
        total_values += activations.size
        nan_values += shard_nan
        positive_inf_values += shard_positive_inf
        negative_inf_values += shard_negative_inf
        maximum_finite_abs = max(maximum_finite_abs, shard_max)
        dtype_counts[str(raw.dtype)] += 1
        storage_counts[storage_dtype] += 1
        if bool(row_bad.any()):
            local_ids = sample_ids[row_bad].tolist()
            affected_ids.extend(local_ids)
            local_locations: dict[str, int] = {}
            if activations.ndim == 4:
                counts = np.sum(~finite, axis=(0, 3))
                for readout_index, readout in enumerate(readouts):
                    for layer_index, layer in enumerate(layers):
                        count = int(counts[readout_index, layer_index])
                        if count:
                            key = f"{readout}:layer_{int(layer)}"
                            local_locations[key] = count
                            nonfinite_locations[key] += count
            contaminated.append(
                {
                    "shard": str(path),
                    "rows": len(activations),
                    "affected_rows": int(row_bad.sum()),
                    "nan_values": shard_nan,
                    "positive_inf_values": shard_positive_inf,
                    "negative_inf_values": shard_negative_inf,
                    "maximum_finite_abs": shard_max,
                    "nonfinite_by_readout_layer": local_locations,
                    "affected_sample_id_examples": local_ids[:8],
                }
            )

    return {
        "schema_version": "activation_numeric_audit_v1",
        "status": "FAIL" if contaminated else "PASS",
        "directory": str(directory),
        "shards": len(paths),
        "rows": total_rows,
        "values": total_values,
        "raw_dtype_shards": dict(sorted(dtype_counts.items())),
        "storage_dtype_shards": dict(sorted(storage_counts.items())),
        "maximum_finite_abs": maximum_finite_abs,
        "nan_values": nan_values,
        "positive_inf_values": positive_inf_values,
        "negative_inf_values": negative_inf_values,
        "contaminated_shards": len(contaminated),
        "affected_rows": len(set(affected_ids)),
        "affected_sample_id_examples": sorted(set(affected_ids))[:20],
        "nonfinite_by_readout_layer": dict(sorted(nonfinite_locations.items())),
        "contaminated_shard_details": contaminated,
        "repairable_without_reinference": False if contaminated else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit_activation_shards(args.activations.expanduser().resolve())
    if args.out:
        write_json_atomic(args.out.expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
