from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from counterfactual_risk_field.cnrf.activations import align_rows, load_activation_data
from counterfactual_risk_field.cnrf.io import read_jsonl
from counterfactual_risk_field.cnrf.schema import (
    build_pair_records,
    evaluation_group,
    modality,
    protocol_split,
    row_label,
)

from .io_utils import repo_root, write_json
from .representation_detectors import (
    RepresentationDetector,
    save_representation_artifact,
)


FORMAT_VERSION = "cnrf_oracle_unified_artifact_build_v1"


def _resolve_from_root(path: Path) -> Path:
    value = path.expanduser()
    return value.resolve() if value.is_absolute() else (repo_root() / value).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _select_records(
    oracle_dir: Path,
    *,
    policy: str,
    max_fpr: float,
    selection_target: str,
    budget: int,
) -> dict[str, dict[str, Any]]:
    source = oracle_dir / "best_pack_subsets.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"Missing CNRF Oracle candidates: {source}")
    matches = [
        record
        for record in read_jsonl(source)
        if str(record.get("policy")) == policy
        and abs(float(record.get("max_fpr", -1.0)) - max_fpr) < 1e-12
        and str(record.get("selection_target")) == selection_target
        and int(record.get("budget", -1)) == budget
    ]
    selected: dict[str, dict[str, Any]] = {}
    for record in matches:
        branch = str(record.get("branch") or "")
        if branch in selected:
            raise ValueError(f"Oracle output contains duplicate matching records for {branch}")
        selected[branch] = record
    if set(selected) != {"text", "image_text"}:
        raise ValueError(
            "Unified CNRF Oracle requires exactly text and image_text records; "
            f"found={sorted(selected)}"
        )
    return selected


def _activation_run_metadata(activation_dir: Path) -> dict[str, Any]:
    run_path = activation_dir / "run.json"
    return _read_json(run_path) if run_path.is_file() else {}


def _array_prefix(branch: str, index: int) -> str:
    return f"cnrf_{branch}_{index}"


def build_artifact(
    *,
    manifest_path: Path,
    activation_dir: Path,
    oracle_dir: Path,
    protocol_config: Path,
    output_path: Path,
    model_id: str | None,
    policy: str,
    max_fpr: float,
    selection_target: str,
    budget: int,
) -> tuple[RepresentationDetector, dict[str, Any]]:
    records = _select_records(
        oracle_dir,
        policy=policy,
        max_fpr=max_fpr,
        selection_target=selection_target,
        budget=budget,
    )
    requested_readouts = sorted(
        {
            str(view["readout"])
            for record in records.values()
            for view in record["views"]
        }
    )
    requested_layers = sorted(
        {
            int(view["layer"])
            for record in records.values()
            for view in record["views"]
        }
    )
    rows = list(read_jsonl(manifest_path))
    activation_data = load_activation_data(
        activation_dir,
        selected_readouts=requested_readouts,
        selected_layers=requested_layers,
    )
    order = align_rows(rows, activation_data)
    values = activation_data.activations[order]
    valid = activation_data.readout_valid[order]
    readout_positions = {
        str(value): index for index, value in enumerate(activation_data.readouts.astype(str))
    }
    layer_positions = {
        int(value): index for index, value in enumerate(activation_data.layers.astype(int))
    }
    pairs = build_pair_records(rows)
    arrays: dict[str, np.ndarray] = {}
    branch_metadata: dict[str, Any] = {}
    hidden_dim = int(values.shape[-1])
    for branch, record in sorted(records.items()):
        selected_packs = set(str(value) for value in record["selected_packs"])
        view_metadata: list[dict[str, Any]] = []
        for view_index, source_view in enumerate(record["views"]):
            readout = str(source_view["readout"])
            layer = int(source_view["layer"])
            readout_position = readout_positions[readout]
            layer_position = layer_positions[layer]
            selected_pairs = [
                pair
                for pair in pairs
                if pair.split == "reference"
                and pair.modality == branch
                and pair.pack_id in selected_packs
                and valid[pair.benign_index, readout_position]
                and valid[pair.harmful_index, readout_position]
            ]
            if not selected_pairs:
                raise ValueError(f"No reference pairs remain for {branch}/{readout}/layer{layer}")
            prefix = _array_prefix(branch, view_index)
            arrays[f"{prefix}_benign"] = values[
                [pair.benign_index for pair in selected_pairs],
                readout_position,
                layer_position,
                :,
            ].astype(np.float32, copy=False)
            arrays[f"{prefix}_harmful"] = values[
                [pair.harmful_index for pair in selected_pairs],
                readout_position,
                layer_position,
                :,
            ].astype(np.float32, copy=False)
            arrays[f"{prefix}_pair_ids"] = np.asarray(
                [pair.pair_id for pair in selected_pairs]
            ).astype(str)
            arrays[f"{prefix}_pack_ids"] = np.asarray(
                [pair.pack_id for pair in selected_pairs]
            ).astype(str)
            view_metadata.append(
                {
                    "readout": readout,
                    "layer": layer,
                    "center": float(source_view["center"]),
                    "scale": float(source_view["scale"]),
                    "support_radius": float(source_view["support_radius"]),
                    "pairs": int(source_view["pairs"]),
                    "packs": int(source_view["packs"]),
                    "array_prefix": prefix,
                }
            )
        point = record["selection_point"]
        selection_groups = {
            str(group): {
                "n": int(metrics["n"]),
                "positive_n": int(metrics["positive_n"]),
                "negative_n": int(metrics["negative_n"]),
                "tpr": None if metrics.get("tpr") is None else float(metrics["tpr"]),
                "fpr": None if metrics.get("fpr") is None else float(metrics["fpr"]),
            }
            for group, metrics in point["by_group"]["groups"].items()
        }
        branch_metadata[branch] = {
            "candidate_id": str(record["candidate_id"]),
            "threshold": float(point["threshold"]),
            "selected_packs": sorted(selected_packs),
            "selection_objective_tpr": float(point["objective_tpr"]),
            "selection_worst_empirical_fpr": float(point["worst_empirical_fpr"]),
            "selection_groups": selection_groups,
            "views": view_metadata,
        }

    protocol = _read_json(protocol_config)
    risk = protocol["risk_field"]
    activation_run = _activation_run_metadata(activation_dir)
    resolved_model_id = str(model_id or activation_run.get("model") or "")
    if not resolved_model_id:
        raise ValueError("Could not infer model id; pass --model-id explicitly")
    first_branch = branch_metadata["image_text"]
    first_view = first_branch["views"][0]
    metadata = {
        "method": "cnrf-oracle",
        "model_id": resolved_model_id,
        "model_revision": activation_run.get("model_revision"),
        "backend": activation_run.get("backend") or activation_data.metadata.get("backend"),
        "layer": int(first_view["layer"]),
        "pooling": str(first_view["readout"]),
        "threshold": float(first_branch["threshold"]),
        "hidden_dim": hidden_dim,
        "protocol": "cnrf_oracle_unified_macro_harmful_v1",
        "implementation_basis": (
            "Unified cross-benchmark CNRF Oracle pack selection with frozen modality branches"
        ),
        "paper_training_protocol": False,
        "core_algorithm_compatible": True,
        "oracle_only": True,
        "label_leakage_scope": "test/external labels select one cross-benchmark pack bank and threshold",
        "per_benchmark_oracle": False,
        "selection_policy": policy,
        "selection_max_fpr": max_fpr,
        "selection_target": selection_target,
        "selection_budget": budget,
        "k": int(risk["k"]),
        "min_arrow_norm": float(risk["min_arrow_norm"]),
        "score_clip": None if risk.get("score_clip") is None else float(risk["score_clip"]),
        "fusion_policy": "supported_max",
        "support_policy": "abstain_safe",
        "decision_rule": "score_greater_than_or_equal",
        "source_manifest": str(manifest_path),
        "source_activations": str(activation_dir),
        "source_oracle": str(oracle_dir),
        "branches": branch_metadata,
    }
    detector = save_representation_artifact(output_path, metadata, arrays)
    verification = verify_artifact(detector, rows, values, valid, readout_positions, layer_positions)
    summary = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "oracle_only": True,
        "per_benchmark_oracle": False,
        "artifact": str(detector.path),
        "artifact_fingerprint": detector.fingerprint,
        "model_id": detector.model_id,
        "selection": {
            "policy": policy,
            "max_fpr": max_fpr,
            "target": selection_target,
            "budget": budget,
        },
        "branches": {
            branch: {
                "candidate_id": value["candidate_id"],
                "threshold": value["threshold"],
                "pack_count": len(value["selected_packs"]),
                "views": value["views"],
            }
            for branch, value in branch_metadata.items()
        },
        "verification": verification,
    }
    write_json(output_path.with_suffix(".build.json"), summary)
    return detector, summary


def verify_artifact(
    detector: RepresentationDetector,
    rows: list[dict[str, Any]],
    values: np.ndarray,
    valid: np.ndarray,
    readout_positions: dict[str, int],
    layer_positions: dict[int, int],
) -> dict[str, Any]:
    """Re-score the frozen oracle evaluation rows and check stored group rates."""

    report: dict[str, Any] = {}
    mismatches: list[str] = []
    for branch, branch_config in detector.metadata["branches"].items():
        branch_rows = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if modality(row) == branch and protocol_split(row) in {"test", "external"}
            ],
            dtype=np.int64,
        )
        fused = np.full(len(branch_rows), -1.0e9, dtype=np.float64)
        supported_any = np.zeros(len(branch_rows), dtype=bool)
        for view_index, view in enumerate(branch_config["views"]):
            readout_position = readout_positions[str(view["readout"])]
            layer_position = layer_positions[int(view["layer"])]
            locally_valid = valid[branch_rows, readout_position]
            if not np.any(locally_valid):
                continue
            result = detector.cnrf_field(branch, view_index).score(
                values[
                    branch_rows[locally_valid], readout_position, layer_position, :
                ]
            )
            standardized = (
                result.scores - float(view["center"])
            ) / float(view["scale"])
            supported = result.nearest_midpoint_distance <= float(view["support_radius"])
            local_positions = np.flatnonzero(locally_valid)[supported]
            fused[local_positions] = np.maximum(fused[local_positions], standardized[supported])
            supported_any[local_positions] = True
        detected = fused >= float(branch_config["threshold"])
        expected_groups = branch_config.get("selection_groups") or {}
        group_values = np.asarray([evaluation_group(rows[index]) for index in branch_rows])
        labels = np.asarray([row_label(rows[index]) for index in branch_rows], dtype=int)
        groups: dict[str, Any] = {}
        for group in sorted(set(group_values.tolist())):
            mask = group_values == group
            positives = mask & (labels == 1)
            negatives = mask & (labels == 0)
            groups[group] = {
                "n": int(mask.sum()),
                "positive_n": int(positives.sum()),
                "negative_n": int(negatives.sum()),
                "tpr": float(np.mean(detected[positives])) if np.any(positives) else None,
                "fpr": float(np.mean(detected[negatives])) if np.any(negatives) else None,
            }
        for group, expected in expected_groups.items():
            actual = groups.get(group)
            if actual is None:
                mismatches.append(f"{branch}/{group}: group missing")
                continue
            for key in ("n", "positive_n", "negative_n"):
                if int(actual[key]) != int(expected[key]):
                    mismatches.append(
                        f"{branch}/{group}/{key}: actual={actual[key]} expected={expected[key]}"
                    )
            for key in ("tpr", "fpr"):
                left, right = actual[key], expected[key]
                if left is None or right is None:
                    if left is not right:
                        mismatches.append(
                            f"{branch}/{group}/{key}: actual={left} expected={right}"
                        )
                elif abs(float(left) - float(right)) > 1e-12:
                    mismatches.append(
                        f"{branch}/{group}/{key}: actual={left} expected={right}"
                    )
        report[branch] = {
            "evaluation_rows": int(len(branch_rows)),
            "support_coverage": float(np.mean(supported_any)) if len(branch_rows) else None,
            "matches_oracle_selection_report": not any(
                value.startswith(f"{branch}/") for value in mismatches
            ),
            "groups": groups,
        }
    if mismatches:
        preview = "; ".join(mismatches[:10])
        raise ValueError(f"Frozen CNRF artifact does not reproduce its Oracle record: {preview}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze one unified cross-benchmark CNRF Oracle as an online defense artifact."
    )
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--activations", type=Path)
    parser.add_argument("--oracle-dir", type=Path)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=Path("counterfactual_risk_field/configs/protocol_v2_diverse_axes.json"),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--policy", default="abstain_safe", choices=["abstain_safe"])
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--selection-target", default="macro_harmful")
    parser.add_argument("--budget", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work = _resolve_from_root(args.work)
    manifest = _resolve_from_root(args.manifest) if args.manifest else work / "experiment.jsonl"
    activations = (
        _resolve_from_root(args.activations)
        if args.activations
        else work / "activations" / args.model_tag
    )
    oracle_dir = (
        _resolve_from_root(args.oracle_dir)
        if args.oracle_dir
        else repo_root()
        / "jailbreak_repro"
        / "runs"
        / "cnrf_oracle"
        / args.model_tag
        / work.name
        / "raw"
    )
    output = (
        _resolve_from_root(args.out)
        if args.out
        else repo_root()
        / "jailbreak_repro"
        / "runs"
        / "cnrf_oracle"
        / args.model_tag
        / work.name
        / "detector"
        / "cnrf_oracle_unified.npz"
    )
    detector, summary = build_artifact(
        manifest_path=manifest.resolve(),
        activation_dir=activations.resolve(),
        oracle_dir=oracle_dir.resolve(),
        protocol_config=_resolve_from_root(args.protocol_config),
        output_path=output.resolve(),
        model_id=args.model_id,
        policy=args.policy,
        max_fpr=float(args.max_fpr),
        selection_target=args.selection_target,
        budget=int(args.budget),
    )
    print(
        json.dumps(
            {
                "artifact": str(detector.path),
                "fingerprint": detector.fingerprint,
                "model_id": detector.model_id,
                "branches": summary["branches"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
