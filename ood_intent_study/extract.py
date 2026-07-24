from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from tqdm import tqdm

import numpy as np

from . import ARTIFACT_VERSION
from .artifacts import completed_sample_ids, read_shard, shard_paths
from .audit_activations import audit_activation_shards
from .io_utils import (
    canonical_json,
    read_jsonl,
    repo_root,
    sha256_file,
    sha256_text,
    stable_fraction,
    write_json_atomic,
)
from .model_backends import (
    decoder_layers,
    forward_kwargs,
    load_model_and_processor,
    parse_layers,
    prepare_input,
    representation_forward_model,
    runtime_metadata,
)
from .schema import StudySample


SUPPORTED_READOUTS = {"last", "all_mean", "non_image_mean", "image_mean"}


class LayerPoolCapture:
    def __init__(self, readouts: list[str], layers: list[int]) -> None:
        unknown = set(readouts) - SUPPORTED_READOUTS
        if unknown:
            raise ValueError(f"Unsupported readouts: {sorted(unknown)}")
        self.readouts = readouts
        self.layers = layers
        self.attention_mask: Any = None
        self.image_mask: Any = None
        self.values: dict[int, list[Any]] = {}
        self.valid: dict[int, list[bool]] = {}

    def prepare(self, attention_mask: Any, image_mask: Any) -> None:
        self.attention_mask = attention_mask
        self.image_mask = image_mask
        self.values = {}
        self.valid = {}

    def hook(self, layer: int) -> Callable[..., None]:
        def capture(_: Any, __: Any, output: Any) -> None:
            import torch

            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(hidden) or hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError(f"Layer {layer} produced unsupported hidden shape")
            attention = self.attention_mask[0].to(hidden.device).bool()
            image = self.image_mask[0].to(hidden.device).bool() & attention
            if hidden.shape[1] != attention.shape[0]:
                raise RuntimeError(
                    f"Layer {layer} sequence length {hidden.shape[1]} != mask {attention.shape[0]}"
                )
            non_image = attention & ~image
            layer_values: list[Any] = []
            layer_valid: list[bool] = []
            for readout in self.readouts:
                if readout == "last":
                    positions = torch.nonzero(attention, as_tuple=False).flatten()
                    is_valid = positions.numel() > 0
                    value = hidden[0, positions[-1]] if is_valid else hidden.new_zeros(hidden.shape[-1])
                elif readout == "all_mean":
                    is_valid = bool(attention.any().item())
                    value = hidden[0, attention].float().mean(dim=0) if is_valid else hidden.new_zeros(hidden.shape[-1])
                elif readout == "non_image_mean":
                    is_valid = bool(non_image.any().item())
                    value = hidden[0, non_image].float().mean(dim=0) if is_valid else hidden.new_zeros(hidden.shape[-1])
                elif readout == "image_mean":
                    is_valid = bool(image.any().item())
                    value = hidden[0, image].float().mean(dim=0) if is_valid else hidden.new_zeros(hidden.shape[-1])
                else:  # pragma: no cover - guarded in __init__
                    raise AssertionError(readout)
                layer_values.append(value.detach().to(device="cpu", dtype=torch.float32))
                layer_valid.append(is_valid)
            self.values[layer] = layer_values
            self.valid[layer] = layer_valid

        return capture

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        missing = set(self.layers) - set(self.values)
        if missing:
            raise RuntimeError(f"Hooks did not capture layers: {sorted(missing)}")
        by_layer = [np.stack([value.numpy() for value in self.values[layer]], axis=0) for layer in self.layers]
        activations = np.stack(by_layer, axis=1)  # readout x layer x hidden
        valid = np.array(
            [[self.valid[layer][readout_index] for layer in self.layers] for readout_index in range(len(self.readouts))],
            dtype=bool,
        )
        if not np.all(valid == valid[:, :1]):
            raise RuntimeError("Readout validity changed across layers for the same input")
        return activations, valid[:, 0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pooled representations from every decoder layer.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True, help="Stable output name, e.g. qwen25vl7b.")
    parser.add_argument("--model", required=True, help="Hub ID or local model path.")
    parser.add_argument("--model-source", choices=["auto", "hf", "modelscope"], default="auto")
    parser.add_argument("--model-revision")
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--backend", choices=["qwen2_5_vl", "gemma3"], required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--readouts", default="last,non_image_mean,image_mean")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--storage-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Use float32 for formal runs; bfloat16 preserves exponent range at half the storage.",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", choices=["auto", "eager", "sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--source", action="append", help="Optional source filter; repeatable.")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--max-consecutive-same-error",
        type=int,
        default=3,
        help="Abort after this many consecutive identical sample errors; 0 disables the circuit breaker.",
    )
    parser.add_argument("--empty-cache-every", type=int, default=0)
    return parser.parse_args(argv)


def _load_manifest(path: Path) -> tuple[list[StudySample], dict[str, Any]]:
    rows = [StudySample.from_dict(row) for row in read_jsonl(path)]
    metadata_path = path.with_suffix(".manifest.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    calculated = sha256_text("\n".join(canonical_json(row.to_dict()) for row in rows))
    expected = metadata.get("manifest_sha256")
    if expected and expected != calculated:
        raise ValueError(f"Manifest fingerprint mismatch: expected={expected}, calculated={calculated}")
    metadata["manifest_sha256"] = calculated
    return rows, metadata


def _round_robin_limit(rows: list[StudySample], maximum: int | None, seed: int) -> list[StudySample]:
    if maximum is None or maximum <= 0 or len(rows) <= maximum:
        return rows
    buckets: dict[str, list[StudySample]] = defaultdict(list)
    for row in rows:
        buckets[row.source].append(row)
    for source_rows in buckets.values():
        source_rows.sort(key=lambda item: stable_fraction(item.sample_id, seed))
    selected: list[StudySample] = []
    sources = sorted(buckets)
    while sources and len(selected) < maximum:
        for source in list(sources):
            if len(selected) >= maximum:
                break
            if buckets[source]:
                selected.append(buckets[source].pop())
            if not buckets[source]:
                sources.remove(source)
    return sorted(selected, key=lambda item: item.sample_id)


def _failed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    output: set[str] = set()
    for row in read_jsonl(path):
        if row.get("sample_id"):
            output.add(str(row["sample_id"]))
    return output


def _append_error(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()


def _write_shard(
    path: Path,
    batch: list[dict[str, Any]],
    layers: list[int],
    readouts: list[str],
    metadata: dict[str, Any],
    storage_dtype: str,
) -> None:
    activations = np.stack([row["activations"] for row in batch]).astype(np.float32, copy=False)
    finite = np.isfinite(activations)
    if not bool(finite.all()):
        affected = np.where(~finite.reshape(len(batch), -1).all(axis=1))[0]
        sample_ids = [batch[int(index)]["sample_id"] for index in affected[:8]]
        raise FloatingPointError(
            "Model pooling produced non-finite activations before storage conversion; "
            f"affected_sample_ids={sample_ids}"
        )
    if storage_dtype == "float16":
        limit = float(np.finfo(np.float16).max)
        row_max = np.max(np.abs(activations), axis=tuple(range(1, activations.ndim)))
        overflow = np.where(row_max > limit)[0]
        if len(overflow):
            sample_ids = [batch[int(index)]["sample_id"] for index in overflow[:8]]
            raise OverflowError(
                "FP16 activation storage would overflow finite model outputs. "
                f"max_abs={float(row_max.max()):.8g}, fp16_max={limit:.8g}, "
                f"affected_sample_ids={sample_ids}. Re-run with --storage-dtype float32 "
                "(recommended) or bfloat16."
            )
        stored_activations = activations.astype(np.float16)
    elif storage_dtype == "bfloat16":
        stored_activations = _float32_to_bfloat16_bits(activations)
        nonfinite_encoding = (stored_activations & np.uint16(0x7F80)) == np.uint16(0x7F80)
        if bool(nonfinite_encoding.any()):
            raise OverflowError(
                "BFloat16 activation storage would overflow finite float32 pooling outputs; "
                "re-run with --storage-dtype float32."
            )
    elif storage_dtype == "float32":
        stored_activations = activations
    else:  # pragma: no cover - argparse guards this in production.
        raise ValueError(f"Unsupported storage_dtype={storage_dtype!r}")

    stored_metadata = {
        **metadata,
        "storage_dtype": storage_dtype,
        "activation_array_dtype": str(stored_activations.dtype),
    }
    arrays = {
        "sample_ids": np.array([row["sample_id"] for row in batch]),
        "activations": stored_activations,
        "readout_valid": np.stack([row["readout_valid"] for row in batch]).astype(bool),
        "layers": np.array(layers, dtype=np.int32),
        "readouts": np.array(readouts),
        "sequence_lengths": np.array([row["sequence_length"] for row in batch], dtype=np.int32),
        "image_token_counts": np.array([row["image_token_count"] for row in batch], dtype=np.int32),
        "text_token_counts": np.array([row["text_token_count"] for row in batch], dtype=np.int32),
        "image_widths": np.array([row["image_width"] for row in batch], dtype=np.int32),
        "image_heights": np.array([row["image_height"] for row in batch], dtype=np.int32),
        "rendered_prompt_sha256": np.array([row["rendered_prompt_sha256"] for row in batch]),
        "metadata_json": np.array(canonical_json(stored_metadata)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _float32_to_bfloat16_bits(values: np.ndarray) -> np.ndarray:
    """Round finite float32 values to IEEE bfloat16 encoded as uint16."""
    array = np.ascontiguousarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def _next_shard_index(directory: Path) -> int:
    paths = shard_paths(directory)
    if not paths:
        return 0
    return max(int(path.stem.split("_")[-1]) for path in paths) + 1


def _implementation_identity() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    relevant = {
        "__init__.py",
        "artifacts.py",
        "extract.py",
        "io_utils.py",
        "model_backends.py",
        "schema.py",
    }
    files = {
        path.name: sha256_file(path)
        for path in sorted(package_root.glob("*.py"))
        if path.is_file() and path.name in relevant
    }
    return {
        "files": files,
        "sha256": sha256_text(canonical_json(files)),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.shard_size < 1:
        raise ValueError("--shard-size must be positive")
    if args.max_consecutive_same_error < 0:
        raise ValueError("--max-consecutive-same-error cannot be negative")
    readouts = [value.strip() for value in args.readouts.split(",") if value.strip()]
    if not readouts or len(readouts) != len(set(readouts)):
        raise ValueError("--readouts must contain unique names")

    manifest_path = args.manifest.expanduser().resolve()
    rows, manifest_metadata = _load_manifest(manifest_path)
    if args.source:
        wanted = set(args.source)
        rows = [row for row in rows if row.source in wanted]
    rows = _round_robin_limit(rows, args.max_samples, args.seed)
    if not rows:
        raise ValueError("No manifest rows remain after filtering")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_path = out_dir / "errors.jsonl"
    if shard_paths(out_dir) and not args.resume:
        raise FileExistsError(f"Activation shards already exist in {out_dir}; pass --resume")

    processor, model, resolved_model = load_model_and_processor(args)
    inference_model = representation_forward_model(model)
    modules, layer_path = decoder_layers(model)
    layers = parse_layers(args.layers, len(modules))
    selection_sha = sha256_text("\n".join(row.sample_id for row in rows))
    runtime = runtime_metadata(model, processor, inference_model, resolved_model, layer_path, args)
    runtime["implementation"] = _implementation_identity()
    runtime_identity = {
        key: runtime.get(key)
        for key in (
            "model",
            "resolved_model",
            "model_source",
            "model_revision",
            "backend",
            "dtype",
            "device_map",
            "hf_device_map",
            "attn_implementation",
            "trust_remote_code",
            "decoder_layer_path",
            "model_class",
            "forward_model_class",
            "processor_class",
            "checkpoint_commit",
            "checkpoint_layout_sha256",
            "config_sha256",
            "chat_template_sha256",
            "packages",
            "hardware",
            "implementation",
        )
    }
    runtime_identity_sha256 = sha256_text(canonical_json(runtime_identity))
    run_contract = {
        "manifest_sha256": manifest_metadata["manifest_sha256"],
        "selection_sha256": selection_sha,
        "model_name": args.model_name,
        "model": args.model,
        "resolved_model": resolved_model,
        "backend": args.backend,
        "layers": layers,
        "readouts": readouts,
        "storage_dtype": args.storage_dtype,
        "runtime_identity_sha256": runtime_identity_sha256,
    }
    run_fingerprint = sha256_text(canonical_json(run_contract))

    run_path = out_dir / "run.json"
    if run_path.is_file():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous.get("run_fingerprint") != run_fingerprint:
            raise ValueError(
                "Existing run fingerprint differs. Use a new output directory rather than mixing shards."
            )
    for path in shard_paths(out_dir):
        shard = read_shard(path)
        if shard.metadata.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"Shard {path} does not belong to this extraction run")

    capture = LayerPoolCapture(readouts=readouts, layers=layers)
    handles = [modules[layer - 1].register_forward_hook(capture.hook(layer)) for layer in layers]
    completed = completed_sample_ids(out_dir)
    failed = _failed_ids(errors_path) if not args.retry_failures else set()
    failed.difference_update(completed)
    selected_ids = {row.sample_id for row in rows}
    write_json_atomic(
        run_path,
        {
            **run_contract,
            "schema_version": "layerwise_extraction_run_v1",
            "run_fingerprint": run_fingerprint,
            "status": "running",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "selected_rows": len(rows),
            "completed_rows": len(completed & selected_ids),
            "failed_rows": len(failed & selected_ids),
            "shards": len(shard_paths(out_dir)),
            "runtime": runtime,
        },
    )
    shard_index = _next_shard_index(out_dir)
    pending: list[dict[str, Any]] = []
    shard_metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "run_fingerprint": run_fingerprint,
        "runtime_identity_sha256": runtime_identity_sha256,
        "manifest_sha256": manifest_metadata["manifest_sha256"],
        "selection_sha256": selection_sha,
        "model": args.model,
        "model_name": args.model_name,
        "backend": args.backend,
        "layers": layers,
        "readouts": readouts,
        "pooling_location": "decoder_block_output_gpu",
        "layer_numbering": "one_based_decoder_block_output",
        "runtime": runtime,
    }

    import torch

    processed_this_run = 0
    image_hash_cache: dict[str, str] = {}
    previous_error_signature = ""
    consecutive_same_errors = 0
    try:
        with torch.inference_mode():
            for sample in tqdm(rows, desc="Extracting Activations", total=len(rows)):
                if sample.sample_id in completed or sample.sample_id in failed:
                    continue
                try:
                    image_path: Path | None = None
                    if sample.image_path:
                        candidate = Path(sample.image_path)
                        image_path = candidate if candidate.is_absolute() else args.repo_root.expanduser().resolve() / candidate
                        if not image_path.is_file():
                            raise FileNotFoundError(f"Missing image: {image_path}")
                        cache_key = str(image_path.resolve())
                        actual_hash = image_hash_cache.get(cache_key)
                        if actual_hash is None:
                            actual_hash = sha256_file(image_path)
                            image_hash_cache[cache_key] = actual_hash
                        if sample.image_sha256 and actual_hash != sample.image_sha256:
                            raise ValueError(
                                f"Image fingerprint changed for {sample.sample_id}: "
                                f"manifest={sample.image_sha256}, actual={actual_hash}"
                            )
                    prepared = prepare_input(
                        processor,
                        model,
                        backend=args.backend,
                        prompt=sample.prompt_text,
                        image_path=image_path,
                        forward_model=inference_model,
                    )
                    capture.prepare(prepared.attention_mask, prepared.image_mask)
                    output = inference_model(
                        **prepared.model_inputs,
                        **forward_kwargs(inference_model),
                    )
                    activations, valid = capture.finalize()
                    pending.append(
                        {
                            "sample_id": sample.sample_id,
                            "activations": activations,
                            "readout_valid": valid,
                            "sequence_length": prepared.sequence_length,
                            "image_token_count": prepared.image_token_count,
                            "text_token_count": prepared.sequence_length - prepared.image_token_count,
                            "image_width": prepared.image_width,
                            "image_height": prepared.image_height,
                            "rendered_prompt_sha256": sha256_text(prepared.rendered_prompt),
                        }
                    )
                    processed_this_run += 1
                    previous_error_signature = ""
                    consecutive_same_errors = 0
                    del output, prepared, activations
                    if len(pending) >= args.shard_size:
                        _write_shard(
                            out_dir / f"shard_{shard_index:05d}.npz",
                            pending,
                            layers,
                            readouts,
                            shard_metadata,
                            args.storage_dtype,
                        )
                        completed.update(row["sample_id"] for row in pending)
                        pending.clear()
                        shard_index += 1
                    if args.empty_cache_every and processed_this_run % args.empty_cache_every == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except Exception as exc:
                    signature = f"{type(exc).__name__}:{exc}"
                    if signature == previous_error_signature:
                        consecutive_same_errors += 1
                    else:
                        previous_error_signature = signature
                        consecutive_same_errors = 1
                    _append_error(
                        errors_path,
                        {
                            "sample_id": sample.sample_id,
                            "source": sample.source,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "time": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    failed.add(sample.sample_id)
                    if args.fail_fast:
                        raise
                    if (
                        args.max_consecutive_same_error
                        and consecutive_same_errors >= args.max_consecutive_same_error
                    ):
                        raise RuntimeError(
                            "Aborting after repeated identical extraction errors; inspect errors.jsonl. "
                            f"Last signature: {signature}"
                        ) from exc
            if pending:
                _write_shard(
                    out_dir / f"shard_{shard_index:05d}.npz",
                    pending,
                    layers,
                    readouts,
                    shard_metadata,
                    args.storage_dtype,
                )
                completed.update(row["sample_id"] for row in pending)
                pending.clear()
    except Exception as exc:
        write_json_atomic(
            run_path,
            {
                **run_contract,
                "schema_version": "layerwise_extraction_run_v1",
                "run_fingerprint": run_fingerprint,
                "status": "failed",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "selected_rows": len(rows),
                "completed_rows": len(completed & selected_ids),
                "failed_rows": len(failed & selected_ids),
                "shards": len(shard_paths(out_dir)),
                "fatal_error_type": type(exc).__name__,
                "fatal_error": str(exc),
                "runtime": runtime,
            },
        )
        raise
    finally:
        for handle in handles:
            handle.remove()

    try:
        numeric_audit = audit_activation_shards(out_dir)
        write_json_atomic(out_dir / "numeric_audit.json", numeric_audit)
    except Exception as exc:
        write_json_atomic(
            run_path,
            {
                **run_contract,
                "schema_version": "layerwise_extraction_run_v1",
                "run_fingerprint": run_fingerprint,
                "status": "failed",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "selected_rows": len(rows),
                "completed_rows": len(completed & selected_ids),
                "failed_rows": len(failed & selected_ids),
                "shards": len(shard_paths(out_dir)),
                "fatal_error_type": type(exc).__name__,
                "fatal_error": f"Post-extraction numeric audit failed: {exc}",
                "runtime": runtime,
            },
        )
        raise
    extraction_complete = (
        len(completed & selected_ids) == len(rows)
        and not (failed & selected_ids)
        and numeric_audit["status"] == "PASS"
    )
    run_summary = {
        **run_contract,
        "schema_version": "layerwise_extraction_run_v1",
        "run_fingerprint": run_fingerprint,
        "status": "complete" if extraction_complete else "incomplete",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "selected_rows": len(rows),
        "completed_rows": len(completed & selected_ids),
        "failed_rows": len(failed & selected_ids),
        "shards": len(shard_paths(out_dir)),
        "numeric_audit": {
            key: numeric_audit[key]
            for key in (
                "status",
                "values",
                "maximum_finite_abs",
                "nan_values",
                "positive_inf_values",
                "negative_inf_values",
                "contaminated_shards",
                "affected_rows",
            )
        },
        "runtime": runtime,
    }
    write_json_atomic(run_path, run_summary)
    print(canonical_json(run_summary))
    return 0 if extraction_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
