from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import canonical_json, read_jsonl, sha256_file, write_json_atomic
from .metrics import balanced_threshold, standardized_centroid_probe


PANELS = ("all", "text_only", "multimodal_only")
MODELS = ("qwen25vl7b", "gemma3_12b")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Break a leave-one-source-out result into source-internal variants and "
            "categories at each panel's validation-selected last-token layer."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", default="MM-SafetyBench")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--panels", nargs="+", choices=PANELS, default=list(PANELS))
    return parser.parse_args(argv)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "source"


def _panel_mask(rows: list[dict[str, Any]], panel: str) -> np.ndarray:
    if panel == "all":
        return np.ones(len(rows), dtype=bool)
    modality = "text" if panel == "text_only" else "image_text"
    return np.asarray([str(row["modality"]) == modality for row in rows], dtype=bool)


def _selected_layers(root: Path, models: list[str], panels: list[str]) -> dict[str, dict[str, int]]:
    selected: dict[str, dict[str, int]] = {}
    for model in models:
        selected[model] = {}
        for panel in panels:
            path = root / "analysis_panels" / panel / model / "analysis.json"
            analysis = json.loads(path.read_text(encoding="utf-8"))
            selected[model][panel] = int(analysis["selected_layers"]["last"]["layer"])
    return selected


def _load_requested_layers(
    activation_dir: Path,
    requested_layers: set[int],
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    sample_chunks: list[np.ndarray] = []
    valid_chunks: list[np.ndarray] = []
    layer_chunks: dict[int, list[np.ndarray]] = {layer: [] for layer in requested_layers}
    expected_layers: np.ndarray | None = None
    expected_readouts: np.ndarray | None = None

    shards = sorted(activation_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No activation shards found in {activation_dir}")
    for shard_path in shards:
        with np.load(shard_path, allow_pickle=False) as data:
            layers = data["layers"].astype(int)
            readouts = data["readouts"].astype(str)
            if expected_layers is None:
                expected_layers = layers
                expected_readouts = readouts
            elif not np.array_equal(layers, expected_layers) or not np.array_equal(
                readouts, expected_readouts
            ):
                raise ValueError(f"Shard schema mismatch at {shard_path}")
            matches = np.flatnonzero(readouts == "last")
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one last readout in {shard_path}")
            readout_index = int(matches[0])
            raw = data["activations"]
            sample_chunks.append(data["sample_ids"].astype(str))
            valid_chunks.append(data["readout_valid"][:, readout_index].astype(bool))
            for layer in requested_layers:
                layer_matches = np.flatnonzero(layers == layer)
                if len(layer_matches) != 1:
                    raise ValueError(f"Layer {layer} is absent or duplicated in {shard_path}")
                layer_chunks[layer].append(
                    raw[:, readout_index, int(layer_matches[0]), :].astype(np.float32, copy=True)
                )
    sample_ids = np.concatenate(sample_chunks)
    if len(sample_ids) != len(set(sample_ids.tolist())):
        raise ValueError(f"Duplicate sample IDs in {activation_dir}")
    return (
        sample_ids,
        {layer: np.concatenate(chunks, axis=0) for layer, chunks in layer_chunks.items()},
        np.concatenate(valid_chunks),
    )


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _aggregate(scores: pd.DataFrame, group_field: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped = [("all", scores)] if group_field == "overall" else scores.groupby(group_field, dropna=False)
    for subgroup, subset in grouped:
        correct_n = int(subset["correct"].sum())
        total = int(len(subset))
        ci_low, ci_high = _wilson_interval(correct_n, total)
        output.append(
            {
                "grouping": group_field,
                "subgroup": str(subgroup),
                "n": total,
                "correct_n": correct_n,
                "label_recall": correct_n / total,
                "label_recall_wilson_95_low": ci_low,
                "label_recall_wilson_95_high": ci_high,
                "score_mean": float(subset["score"].mean()),
                "score_median": float(subset["score"].median()),
                "margin_mean": float(subset["margin"].mean()),
                "margin_median": float(subset["margin"].median()),
            }
        )
    return output


def _expected_loso_recall(root: Path, panel: str, model: str, layer: int, source: str) -> float:
    frame = pd.read_csv(root / "analysis_panels" / panel / model / "leave_one_source_out.csv")
    match = frame[
        (frame["readout"] == "last")
        & (frame["layer"] == layer)
        & (frame["held_out_source"] == source)
        & frame["eligible"].astype(bool)
    ]
    if len(match) != 1:
        raise ValueError(
            f"Expected one eligible LOSO row for panel={panel}, model={model}, "
            f"layer={layer}, source={source}; found {len(match)}"
        )
    row = match.iloc[0]
    if float(row.get("positive_n", 0)) > 0:
        return float(row["tpr"])
    return float(row["tnr"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    out_dir = (args.out_dir or (root / "diagnostics")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "samples.jsonl"
    manifest_rows = list(read_jsonl(manifest_path))
    manifest_by_id = {str(row["sample_id"]): row for row in manifest_rows}
    selected = _selected_layers(root, list(args.models), list(args.panels))
    score_outputs: list[pd.DataFrame] = []
    aggregate_outputs: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []

    for model in args.models:
        sample_ids, layer_arrays, valid = _load_requested_layers(
            root / "activations" / model,
            set(selected[model].values()),
        )
        missing = [sample_id for sample_id in sample_ids if sample_id not in manifest_by_id]
        if missing:
            raise ValueError(f"Manifest is missing activation IDs: {missing[:5]}")
        rows = [manifest_by_id[sample_id] for sample_id in sample_ids]
        labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
        sources = np.asarray([str(row["source"]) for row in rows])
        splits = np.asarray([str(row["split"]) for row in rows])
        groups = np.asarray([str(row["split_group_id"]) for row in rows])
        attacks = np.asarray([bool(row["is_attack"]) for row in rows], dtype=bool)

        for panel in args.panels:
            layer = selected[model][panel]
            X = layer_arrays[layer]
            keep = _panel_mask(rows, panel) & valid
            holdout = keep & ~attacks & (sources == args.source)
            if not holdout.any():
                continue
            held_out_groups = set(groups[holdout].tolist())
            related = np.isin(groups, list(held_out_groups))
            train = keep & ~attacks & ~related & (splits == "train")
            calibration = keep & ~attacks & ~related & (splits == "validation")
            if len(np.unique(labels[train])) != 2 or len(np.unique(labels[calibration])) != 2:
                raise ValueError(
                    f"Panel {panel}, model {model} cannot fit/calibrate after holding out {args.source}"
                )

            _, calibration_scores, direction, scale = standardized_centroid_probe(
                X[train], labels[train], X[calibration]
            )
            train_mean = X[train].mean(axis=0)
            holdout_scores = ((X[holdout] - train_mean) / scale) @ direction
            threshold = balanced_threshold(labels[calibration], calibration_scores)
            heldout_rows = [row for row, included in zip(rows, holdout) if included]
            heldout_labels = labels[holdout]
            predictions = holdout_scores >= threshold
            correct = predictions == heldout_labels.astype(bool)
            score_frame = pd.DataFrame(
                {
                    "sample_id": [str(row["sample_id"]) for row in heldout_rows],
                    "model": model,
                    "panel": panel,
                    "selected_layer": layer,
                    "readout": "last",
                    "held_out_source": args.source,
                    "label": heldout_labels,
                    "modality": [str(row.get("modality", "")) for row in heldout_rows],
                    "variant": [str(row.get("variant", "")) for row in heldout_rows],
                    "category": [str(row.get("category", "")) for row in heldout_rows],
                    "original_split": [str(row.get("split", "")) for row in heldout_rows],
                    "split_group_id": [str(row.get("split_group_id", "")) for row in heldout_rows],
                    "score": holdout_scores,
                    "threshold": threshold,
                    "margin": holdout_scores - threshold,
                    "predicted_harmful": predictions.astype(int),
                    "correct": correct.astype(int),
                }
            )
            score_outputs.append(score_frame)
            for grouping in ("overall", "variant", "category"):
                for row in _aggregate(score_frame, grouping):
                    aggregate_outputs.append(
                        {
                            "model": model,
                            "panel": panel,
                            "selected_layer": layer,
                            "readout": "last",
                            "held_out_source": args.source,
                            "held_out_label": int(heldout_labels[0]),
                            **row,
                        }
                    )

            observed = float(correct.mean())
            expected = _expected_loso_recall(root, panel, model, layer, args.source)
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    f"LOSO verification failed for panel={panel}, model={model}: "
                    f"subgroup diagnostic={observed}, saved analysis={expected}"
                )
            verification_rows.append(
                {
                    "model": model,
                    "panel": panel,
                    "selected_layer": layer,
                    "observed_label_recall": observed,
                    "saved_label_recall": expected,
                    "exact_match": True,
                }
            )

    scores = pd.concat(score_outputs, ignore_index=True) if score_outputs else pd.DataFrame()
    aggregates = pd.DataFrame(aggregate_outputs)
    source_slug = _safe_slug(args.source)
    scores_path = out_dir / f"{source_slug}_loso_scores.csv"
    aggregate_path = out_dir / f"{source_slug}_loso_subgroups.csv"
    scores.to_csv(scores_path, index=False)
    aggregates.to_csv(aggregate_path, index=False)
    summary = {
        "schema_version": "source_loso_subgroup_diagnostics_v1",
        "source": args.source,
        "models": list(args.models),
        "panels": list(args.panels),
        "readout": "last",
        "selected_layers": selected,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "score_rows": int(len(scores)),
        "aggregate_rows": int(len(aggregates)),
        "verification": verification_rows,
        "outputs": {
            "scores": str(scores_path),
            "subgroups": str(aggregate_path),
        },
        "interpretation_notes": [
            "Every source-internal subgroup is evaluated by a probe that held out the entire source.",
            "The layer is selected by the corresponding panel's ordinary validation protocol before this diagnostic.",
            "Category cells are exploratory because several contain only a small number of samples.",
            "Wilson intervals treat rows as Bernoulli observations and do not correct for repeated semantic variants.",
        ],
    }
    write_json_atomic(out_dir / f"{source_slug}_loso_subgroups.json", summary)
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
