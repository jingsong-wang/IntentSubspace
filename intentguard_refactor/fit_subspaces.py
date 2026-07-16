from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from intentguard.io import read_jsonl, write_json
from intentguard.subspace import fit_all_layers, save_subspace


def array_or_default(data: Any, key: str, n: int, default: str = "") -> np.ndarray:
    if key in data:
        return data[key].astype(str)
    return np.array([default] * n)


def groups_from(data: Any, group_by: str, mask: np.ndarray | None = None) -> np.ndarray | None:
    n = len(data["ids"])
    if group_by == "none":
        return None
    if group_by == "condition":
        arr = array_or_default(data, "conditions", n, "unknown_condition")
    elif group_by == "intent_family":
        arr = array_or_default(data, "intent_families", n, "unknown_family")
    elif group_by == "condition_intent_family":
        cond = array_or_default(data, "conditions", n, "unknown_condition")
        fam = array_or_default(data, "intent_families", n, "unknown_family")
        arr = np.array([f"{c}::{f}" for c, f in zip(cond.tolist(), fam.tolist())])
    else:
        raise ValueError(f"Unsupported group-by: {group_by}")
    return arr[mask] if mask is not None else arr


def infer_refusal_label(row: dict[str, Any], harmful_score_threshold: int) -> int | None:
    if row.get("refusal_label") is not None:
        return int(row["refusal_label"])
    # RSS predicts a safe refusal/pivot, not the presence of refusal wording.
    # A response that refuses but still leaks actionable harmful content is a
    # negative example for the safety route.
    if row.get("judge_harmful") is True:
        return 0
    if (
        row.get("judge_harmfulness_score") is not None
        and int(row["judge_harmfulness_score"]) >= harmful_score_threshold
    ):
        return 0
    if row.get("refusal_heuristic") is not None:
        return 1 if bool(row["refusal_heuristic"]) else 0
    refusal_state = str(row.get("refusal_state", "")).lower()
    if refusal_state == "refused":
        return 1
    if refusal_state in {"not_refused", "refused_but_leaky"}:
        return 0
    outcome = str(row.get("response_outcome", "")).lower()
    if outcome == "refusal_safe_pivot":
        return 1
    if outcome == "judge_safe_unclear_refusal":
        return None
    if outcome in {"harmful_compliance", "safe_nonrefusal"}:
        return 0
    return None


def load_refusal_labels(path: Path, harmful_score_threshold: int) -> dict[str, int]:
    labels: dict[str, int] = {}
    for row in read_jsonl(path):
        label = infer_refusal_label(row, harmful_score_threshold)
        if label is None:
            continue
        labels[str(row.get("id"))] = int(label)
    return labels


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            vals.append(str(value))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def compact_layer_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        cv = result["cv"]
        rows.append(
            {
                "layer": result["layer"],
                "auc": cv.get("auc"),
                "ap": cv.get("average_precision"),
                "cv_bal_acc": cv.get("cross_validated_balanced_accuracy"),
                "full_auc": result["full"].get("auc"),
                "mean_delta_norm": result["diagnostics"].get("mean_delta_norm"),
            }
        )
    return rows


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# IntentGuard-LRH Subspace Fit Report",
        "",
        f"Activations: `{summary['activations']}`",
        f"Selection group-by: `{summary['group_by']}`",
        "",
    ]
    for key, title in [("intent", "Risk Detection Subspace S_I"), ("refusal", "Refusal Tendency Subspace S_R")]:
        if key not in summary:
            continue
        section = summary[key]
        lines.extend(
            [
                f"## {title}",
                "",
                f"Output: `{section['path']}`",
                f"Selected layer: `{section['selected_layer']}`",
                f"Selected CV AUROC: `{section['selected_auc']:.4f}`",
                f"Selected CV balanced accuracy: `{section['selected_balanced_accuracy']:.4f}`",
                "",
                markdown_table(section["layers"], ["layer", "auc", "ap", "cv_bal_acc", "full_auc", "mean_delta_norm"]),
                "",
            ]
        )
    (out_dir / "subspace_selection_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--intent-rank", type=int, default=3)
    parser.add_argument("--refusal-rank", type=int, default=2)
    parser.add_argument("--group-by", choices=["condition", "intent_family", "condition_intent_family", "none"], default="condition")
    parser.add_argument("--refusal-labels", type=Path, help="JSONL from generation or judge output. If omitted, S_R is skipped.")
    parser.add_argument("--harmful-score-threshold", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data = np.load(args.activations, allow_pickle=True)
    activations = data["activations"]
    layers = data["layers"].astype(int)
    ids = data["ids"].astype(str)
    labels = data["labels"].astype(int)
    pair_keys = data["pair_keys"].astype(str)
    groups = groups_from(data, args.group_by)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    intent_bases, intent_centers, intent_results, intent_selected = fit_all_layers(
        activations=activations,
        layers=layers,
        y=labels,
        rank=args.intent_rank,
        mode="intent",
        groups=groups,
        pair_keys=pair_keys,
        seed=args.seed,
    )
    intent_path = args.out_dir / "intent_subspace.npz"
    save_subspace(intent_path, layers, intent_bases, intent_centers, args.intent_rank, "intent", intent_selected, args.activations)

    summary: dict[str, Any] = {
        "activations": str(args.activations),
        "group_by": args.group_by,
        "intent": {
            "path": str(intent_path),
            "rank": args.intent_rank,
            "selected_layer": int(intent_selected["layer"]),
            "selected_layer_index": int(intent_selected["layer_index"]),
            "selected_auc": float(intent_selected["cv"].get("auc", float("nan"))),
            "selected_balanced_accuracy": float(intent_selected["cv"].get("cross_validated_balanced_accuracy", float("nan"))),
            "layers": compact_layer_rows(intent_results),
        },
    }

    if args.refusal_labels is not None:
        refusal_by_id = load_refusal_labels(args.refusal_labels, args.harmful_score_threshold)
        mask = np.array([sample_id in refusal_by_id for sample_id in ids], dtype=bool)
        refusal_y = np.array([refusal_by_id[sample_id] for sample_id in ids[mask]], dtype=int)
        if len(refusal_y) == 0 or len(set(refusal_y.tolist())) < 2:
            raise ValueError(
                f"Refusal labels must cover both classes. Covered={len(refusal_y)}, counts={dict(Counter(refusal_y.tolist()))}"
            )
        refusal_groups = groups_from(data, args.group_by, mask=mask)
        refusal_bases, refusal_centers, refusal_results, refusal_selected = fit_all_layers(
            activations=activations[mask],
            layers=layers,
            y=refusal_y,
            rank=args.refusal_rank,
            mode="refusal",
            groups=refusal_groups,
            pair_keys=None,
            seed=args.seed,
        )
        refusal_path = args.out_dir / "refusal_subspace.npz"
        save_subspace(refusal_path, layers, refusal_bases, refusal_centers, args.refusal_rank, "refusal", refusal_selected, args.activations)
        summary["refusal"] = {
            "path": str(refusal_path),
            "rank": args.refusal_rank,
            "label_source": str(args.refusal_labels),
            "label_counts": dict(Counter(refusal_y.tolist())),
            "covered_samples": int(mask.sum()),
            "selected_layer": int(refusal_selected["layer"]),
            "selected_layer_index": int(refusal_selected["layer_index"]),
            "selected_auc": float(refusal_selected["cv"].get("auc", float("nan"))),
            "selected_balanced_accuracy": float(refusal_selected["cv"].get("cross_validated_balanced_accuracy", float("nan"))),
            "layers": compact_layer_rows(refusal_results),
        }

    write_json(args.out_dir / "subspace_selection.json", summary)
    write_report(args.out_dir, summary)
    print(f"Wrote S_I to {intent_path}")
    if "refusal" in summary:
        print(f"Wrote S_R to {summary['refusal']['path']}")
    print(f"Wrote selection summary to {args.out_dir / 'subspace_selection.json'}")


if __name__ == "__main__":
    main()
