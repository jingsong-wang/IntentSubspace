from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from intentguard.io import read_json, read_jsonl, write_json, write_jsonl
from intentguard.subspace import load_subspace, project_scores


DEFAULT_HARD_REFUSAL = (
    "I can't help with that request. I can help with safe, high-level prevention, "
    "reporting, de-escalation, or harm-reduction information instead."
)


def score_by_id(activations_path: Path, subspace_path: Path, layer: str | int) -> dict[str, dict[str, Any]]:
    data = np.load(activations_path, allow_pickle=True)
    ids = data["ids"].astype(str)
    layers = data["layers"].astype(int)
    subspace = load_subspace(subspace_path, layer)
    matches = np.where(layers == subspace["layer"])[0]
    if len(matches) != 1:
        raise ValueError(f"Activations do not contain layer {subspace['layer']}; available={layers.tolist()}")
    X = data["activations"][:, int(matches[0]), :]
    coords, scores = project_scores(X, subspace["center"], subspace["basis"])
    return {
        sample_id: {
            "layer": int(subspace["layer"]),
            "score": float(scores[i]),
            "coords": [float(v) for v in coords[i].tolist()],
            "subspace": str(subspace_path),
        }
        for i, sample_id in enumerate(ids.tolist())
    }


def normalized_prompt(row: dict[str, Any]) -> str:
    return str(row.get("prompt") or row.get("prompt_text") or row.get("user_prompt") or "").strip()


def apply_one(row: dict[str, Any], intent_score: dict[str, Any], refusal_score: dict[str, Any], thresholds: dict[str, Any], hard_refusal: str) -> dict[str, Any]:
    intent_threshold = float(thresholds["intent"]["threshold"])
    refusal_threshold = float(thresholds["refusal"]["threshold"])
    intent_detected = bool(float(intent_score["score"]) >= intent_threshold)
    refusal_low = bool(float(refusal_score["score"]) < refusal_threshold)
    triggered = bool(intent_detected and refusal_low)
    original_response = str(row.get("response") or "")
    final_response = hard_refusal if triggered else original_response
    final_source = "hard_refusal_template" if triggered else str(row.get("response_source", "model"))
    return {
        **row,
        "prompt": normalized_prompt(row),
        "original_response": original_response,
        "original_response_source": str(row.get("response_source", "model")),
        "intent_score": intent_score["score"],
        "intent_coords": intent_score["coords"],
        "intent_layer": intent_score["layer"],
        "intent_threshold": intent_threshold,
        "intent_detected": intent_detected,
        "refusal_score": refusal_score["score"],
        "refusal_coords": refusal_score["coords"],
        "refusal_layer": refusal_score["layer"],
        "refusal_threshold": refusal_threshold,
        "refusal_low": refusal_low,
        "intervention_triggered": triggered,
        "intervention_policy": "hard_refusal_template",
        "post_intervention_response": final_response,
        "response": final_response,
        "response_source": final_source,
        "detection_result": {
            "intent": {
                "score": intent_score["score"],
                "threshold": intent_threshold,
                "detected": intent_detected,
                "layer": intent_score["layer"],
            },
            "refusal": {
                "score": refusal_score["score"],
                "threshold": refusal_threshold,
                "low_refusal_tendency": refusal_low,
                "layer": refusal_score["layer"],
            },
            "triggered": triggered,
            "rule": thresholds.get("trigger_rule", "intent high and refusal low"),
        },
    }


def summarize(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    n = len(rows)
    detected = sum(1 for r in rows if r.get("intent_detected"))
    refusal_low = sum(1 for r in rows if r.get("refusal_low"))
    triggered = sum(1 for r in rows if r.get("intervention_triggered"))
    return {
        "n": n,
        "trigger_rule": thresholds.get("trigger_rule"),
        "intent_layer": thresholds["intent"]["layer"],
        "refusal_layer": thresholds["refusal"]["layer"],
        "intent_threshold": thresholds["intent"]["threshold"],
        "refusal_threshold": thresholds["refusal"]["threshold"],
        "intent_detected": detected,
        "intent_detected_rate": detected / n if n else float("nan"),
        "refusal_low": refusal_low,
        "refusal_low_rate": refusal_low / n if n else float("nan"),
        "intervention_triggered": triggered,
        "intervention_triggered_rate": triggered / n if n else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Original generation JSONL.")
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--intent-subspace", type=Path, required=True)
    parser.add_argument("--refusal-subspace", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--intent-layer", default="selected")
    parser.add_argument("--refusal-layer", default="selected")
    parser.add_argument("--hard-refusal", default=DEFAULT_HARD_REFUSAL)
    parser.add_argument("--allow-missing-activations", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    thresholds = read_json(args.thresholds)
    intent_scores = score_by_id(args.activations, args.intent_subspace, args.intent_layer)
    refusal_scores = score_by_id(args.activations, args.refusal_subspace, args.refusal_layer)

    output_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        sample_id = str(row.get("id"))
        if sample_id not in intent_scores or sample_id not in refusal_scores:
            missing.append(sample_id)
            if args.allow_missing_activations:
                output_rows.append({**row, "intervention_triggered": False, "detection_missing": True})
                continue
            raise KeyError(f"Missing activation score for sample id={sample_id}")
        output_rows.append(apply_one(row, intent_scores[sample_id], refusal_scores[sample_id], thresholds, args.hard_refusal))

    write_jsonl(args.out, output_rows)
    summary_path = args.summary_out or args.out.with_name(args.out.stem + "_summary.json")
    summary = summarize(output_rows, thresholds)
    if missing:
        summary["missing_activation_ids"] = missing
    write_json(summary_path, summary)
    print(f"Wrote intervention results to {args.out}")
    print(f"Triggered {summary['intervention_triggered']} / {summary['n']} samples")


if __name__ == "__main__":
    main()
