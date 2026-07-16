from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from intentguard.io import read_jsonl, write_json, write_jsonl


def by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in rows}


def compact_judge(row: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    if row is None:
        return {
            f"{prefix}_judge_harmfulness_score": None,
            f"{prefix}_judge_harmful": None,
            f"{prefix}_response_outcome": None,
            f"{prefix}_refusal_state": None,
            f"{prefix}_judge_parse_error": None,
        }
    return {
        f"{prefix}_judge_harmfulness_score": row.get("judge_harmfulness_score"),
        f"{prefix}_judge_harmful": row.get("judge_harmful"),
        f"{prefix}_response_outcome": row.get("response_outcome"),
        f"{prefix}_refusal_state": row.get("refusal_state"),
        f"{prefix}_judge_parse_error": row.get("judge_parse_error"),
    }


def merge_rows(detection_rows: list[dict[str, Any]], original_judge: dict[str, dict[str, Any]], intervention_judge: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    for row in detection_rows:
        sample_id = str(row.get("id"))
        original = original_judge.get(sample_id)
        post = intervention_judge.get(sample_id)
        merged.append(
            {
                "id": sample_id,
                "label": row.get("label"),
                "label_name": row.get("label_name"),
                "condition": row.get("condition"),
                "intent_family": row.get("intent_family"),
                "intent_id": row.get("intent_id"),
                "image_role": row.get("image_role"),
                "image_path": row.get("image_path"),
                "prompt": row.get("prompt") or row.get("prompt_text"),
                "intent_score": row.get("intent_score"),
                "intent_threshold": row.get("intent_threshold"),
                "intent_detected": row.get("intent_detected"),
                "refusal_score": row.get("refusal_score"),
                "refusal_threshold": row.get("refusal_threshold"),
                "refusal_low": row.get("refusal_low"),
                "intervention_triggered": row.get("intervention_triggered"),
                "original_response_source": row.get("original_response_source"),
                "final_response_source": row.get("response_source"),
                **compact_judge(original, "original"),
                **compact_judge(post, "post"),
                "harmful_after_intervention_reduced": (
                    bool(original and original.get("judge_harmful"))
                    and bool(post is not None)
                    and not bool(post.get("judge_harmful"))
                ),
            }
        )
    return merged


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    original_harmful = sum(1 for r in rows if r.get("original_judge_harmful"))
    post_harmful = sum(1 for r in rows if r.get("post_judge_harmful"))
    triggered = sum(1 for r in rows if r.get("intervention_triggered"))
    reduced = sum(1 for r in rows if r.get("harmful_after_intervention_reduced"))
    summary: dict[str, Any] = {
        "n": n,
        "original_judge_harmful": original_harmful,
        "original_judge_harmful_rate": original_harmful / n if n else float("nan"),
        "post_judge_harmful": post_harmful,
        "post_judge_harmful_rate": post_harmful / n if n else float("nan"),
        "intervention_triggered": triggered,
        "intervention_triggered_rate": triggered / n if n else float("nan"),
        "harmful_after_intervention_reduced": reduced,
        "by_label_name": {},
        "by_intent_family": {},
        "by_condition": {},
    }
    for key, out_key in [
        ("label_name", "by_label_name"),
        ("intent_family", "by_intent_family"),
        ("condition", "by_condition"),
    ]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(key, ""))].append(row)
        for group, items in sorted(grouped.items()):
            summary[out_key][group] = {
                "n": len(items),
                "triggered": sum(1 for r in items if r.get("intervention_triggered")),
                "original_harmful": sum(1 for r in items if r.get("original_judge_harmful")),
                "post_harmful": sum(1 for r in items if r.get("post_judge_harmful")),
                "post_outcomes": dict(Counter(str(r.get("post_response_outcome")) for r in items)),
            }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True, help="Output from apply_intervention.py.")
    parser.add_argument("--original-judge", type=Path, required=True)
    parser.add_argument("--post-judge", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()

    detection_rows = read_jsonl(args.detections)
    merged = merge_rows(detection_rows, by_id(read_jsonl(args.original_judge)), by_id(read_jsonl(args.post_judge)))
    write_jsonl(args.out, merged)
    if args.csv_out:
        write_csv(args.csv_out, merged)
    summary = summarize(merged)
    write_json(args.summary_out or args.out.with_name(args.out.stem + "_summary.json"), summary)
    print(f"Wrote sample audit to {args.out}")


if __name__ == "__main__":
    main()
