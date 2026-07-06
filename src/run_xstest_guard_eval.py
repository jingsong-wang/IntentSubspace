import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: Any):
        return iterable

from run_hades_dynamic_guard import (
    decode_generated,
    first_param_device,
    infer_num_layers,
    load_qwen_vl,
    load_subspace,
    move_batch_to_device,
    parse_layers,
    pool_hidden,
    project_activation,
    resolve_threshold,
    response_refused,
)


def deterministic_split(raw_id: str, label: str, seed: int, calibration_ratio: float) -> str:
    key = f"{seed}::{label}::{raw_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], byteorder="big") / float(2**64)
    return "calibration" if value < calibration_ratio else "test"


def load_xstest_rows(
    csv_path: Path,
    max_samples: int | None = None,
    split: str = "all",
    split_seed: int = 7,
    calibration_ratio: float = 0.3,
) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out = []
    for row in rows:
        label = str(row.get("label", "")).strip().lower()
        if label not in {"safe", "unsafe"}:
            raise ValueError(f"Unsupported XSTest label for id={row.get('id')}: {label!r}")
        raw_id = str(row.get("id", "")).strip()
        assigned_split = deterministic_split(raw_id, label, split_seed, calibration_ratio)
        if split != "all" and assigned_split != split:
            continue
        out.append(
            {
                "id": f"xstest_{raw_id}",
                "raw_id": raw_id,
                "dataset": "XSTest",
                "split": assigned_split,
                "prompt": str(row.get("prompt", "")).strip(),
                "type": str(row.get("type", "")).strip(),
                "label": label,
                "label_int": 1 if label == "unsafe" else 0,
                "focus": str(row.get("focus", "")).strip(),
                "note": str(row.get("note", "")).strip(),
                "raw": row,
            }
        )
    if max_samples is not None:
        out = out[:max_samples]
    return out


def build_text_messages(row: dict, system_prompt: str | None = None) -> list[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": row["prompt"]}],
        }
    )
    return messages


def process_text_messages(processor: Any, messages: list[dict]) -> tuple[dict, str]:
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(
        text=[rendered],
        padding=True,
        return_tensors="pt",
    )
    return batch, rendered


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        wins += np.sum(p > neg)
        wins += 0.5 * np.sum(p == neg)
    return float(wins / total)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    tp = 0
    precisions = []
    for i, label in enumerate(y_sorted, start=1):
        if label == 1:
            tp += 1
            precisions.append(tp / i)
    return float(np.mean(precisions)) if precisions else 0.0


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    unsafe_total = int(np.sum(y_true == 1))
    safe_total = int(np.sum(y_true == 0))
    total = int(len(y_true))
    unsafe_acc = tp / unsafe_total if unsafe_total else float("nan")
    safe_acc = tn / safe_total if safe_total else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = unsafe_acc
    f1 = 2 * precision * recall / (precision + recall) if precision == precision and (precision + recall) else float("nan")
    out = {
        "n": total,
        "safe_n": safe_total,
        "unsafe_n": unsafe_total,
        "accuracy": (tp + tn) / total if total else float("nan"),
        "balanced_accuracy": 0.5 * (safe_acc + unsafe_acc),
        "safe_accuracy": safe_acc,
        "unsafe_accuracy": unsafe_acc,
        "precision_unsafe": precision,
        "recall_unsafe": recall,
        "f1_unsafe": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    if scores is not None:
        out["auc"] = binary_auc(y_true, scores)
        out["average_precision"] = average_precision(y_true, scores)
    return out


def run_one(
    row: dict,
    processor: Any,
    model: Any,
    subspace: dict,
    pooling: str,
    threshold: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    system_prompt: str | None,
) -> dict:
    import torch

    messages = build_text_messages(row, system_prompt=system_prompt)
    batch, rendered = process_text_messages(processor, messages)
    batch = move_batch_to_device(dict(batch), first_param_device(model))

    with torch.no_grad():
        out = model(**batch, output_hidden_states=True, use_cache=False)
        hidden = pool_hidden(
            out.hidden_states[subspace["layer"]].detach().float().cpu(),
            batch.get("attention_mask", None).detach().cpu() if "attention_mask" in batch else None,
            pooling,
        ).numpy()

    coords, primary_score, subspace_norm = project_activation(hidden, subspace)
    subspace_pred_unsafe = bool(primary_score >= threshold)

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0.0}
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    with torch.no_grad():
        generated = model.generate(**batch, **gen_kwargs)
    response = decode_generated(processor, batch, generated)
    model_pred_unsafe = response_refused(response)

    label_int = int(row["label_int"])
    return {
        "id": row["id"],
        "raw_id": row["raw_id"],
        "dataset": row["dataset"],
        "type": row["type"],
        "label": row["label"],
        "label_int": label_int,
        "focus": row["focus"],
        "note": row["note"],
        "split": row["split"],
        "prompt": row["prompt"],
        "rendered_prompt": rendered,
        "primary_score": primary_score,
        "subspace_norm": subspace_norm,
        "coords": [float(v) for v in coords.tolist()],
        "threshold": threshold,
        "subspace_pred_unsafe": subspace_pred_unsafe,
        "subspace_correct": bool(int(subspace_pred_unsafe) == label_int),
        "model_pred_unsafe_by_refusal": model_pred_unsafe,
        "model_correct_by_refusal": bool(int(model_pred_unsafe) == label_int),
        "response": response,
    }


def group_metrics(results: list[dict], pred_key: str) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        grouped[row["type"]].append(row)

    out = {}
    for group, rows in sorted(grouped.items()):
        y_true = np.array([r["label_int"] for r in rows], dtype=np.int32)
        y_pred = np.array([int(r[pred_key]) for r in rows], dtype=np.int32)
        scores = np.array([r["primary_score"] for r in rows], dtype=np.float32) if pred_key == "subspace_pred_unsafe" else None
        out[group] = confusion_metrics(y_true, y_pred, scores=scores)
    return out


def summarize(results: list[dict], threshold_info: dict, args: argparse.Namespace) -> dict:
    y_true = np.array([r["label_int"] for r in results], dtype=np.int32)
    subspace_pred = np.array([int(r["subspace_pred_unsafe"]) for r in results], dtype=np.int32)
    model_pred = np.array([int(r["model_pred_unsafe_by_refusal"]) for r in results], dtype=np.int32)
    subspace_scores = np.array([r["primary_score"] for r in results], dtype=np.float32)

    return {
        "n": len(results),
        "label_counts": dict(Counter(r["label"] for r in results)),
        "type_counts": dict(Counter(r["type"] for r in results)),
        "split": args.split,
        "split_seed": args.split_seed,
        "calibration_ratio": args.calibration_ratio,
        "model": args.model,
        "subspace": str(args.subspace),
        "subspace_layer": args.score_layer,
        "pooling": args.pooling,
        "threshold": threshold_info,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "subspace_metrics": confusion_metrics(y_true, subspace_pred, scores=subspace_scores),
        "model_refusal_metrics": confusion_metrics(y_true, model_pred),
        "subspace_by_type": group_metrics(results, "subspace_pred_unsafe"),
        "model_refusal_by_type": group_metrics(results, "model_pred_unsafe_by_refusal"),
        "score_distribution": {
            "safe_mean": float(np.mean(subspace_scores[y_true == 0])),
            "safe_median": float(np.median(subspace_scores[y_true == 0])),
            "unsafe_mean": float(np.mean(subspace_scores[y_true == 1])),
            "unsafe_median": float(np.median(subspace_scores[y_true == 1])),
        },
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_outputs(results: list[dict], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "xstest_results.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (out_dir / "xstest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fields = [
        "id",
        "raw_id",
        "type",
        "label",
        "focus",
        "note",
        "split",
        "primary_score",
        "subspace_norm",
        "threshold",
        "subspace_pred_unsafe",
        "subspace_correct",
        "model_pred_unsafe_by_refusal",
        "model_correct_by_refusal",
        "prompt",
        "response",
    ]
    with (out_dir / "xstest_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fields})

    lines = [
        "# XSTest Intent-Subspace Safety Evaluation",
        "",
        f"Samples: `{summary['n']}`",
        f"Model: `{summary['model']}`",
        f"Subspace: `{summary['subspace']}`",
        f"Threshold: `{summary['threshold']['threshold']:.6f}` from `{summary['threshold']['source']}`",
        "",
        "## Overall Metrics",
        "",
        "| detector | accuracy | balanced_accuracy | safe_accuracy | unsafe_accuracy | precision_unsafe | recall_unsafe | f1_unsafe | auc | ap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, key in [("subspace", "subspace_metrics"), ("model_refusal", "model_refusal_metrics")]:
        row = summary[key]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    fmt(row["accuracy"]),
                    fmt(row["balanced_accuracy"]),
                    fmt(row["safe_accuracy"]),
                    fmt(row["unsafe_accuracy"]),
                    fmt(row["precision_unsafe"]),
                    fmt(row["recall_unsafe"]),
                    fmt(row["f1_unsafe"]),
                    fmt(row.get("auc", float("nan"))),
                    fmt(row.get("average_precision", float("nan"))),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Score Distribution",
            "",
            "| label | mean | median |",
            "| --- | ---: | ---: |",
            f"| safe | {summary['score_distribution']['safe_mean']:.4f} | {summary['score_distribution']['safe_median']:.4f} |",
            f"| unsafe | {summary['score_distribution']['unsafe_mean']:.4f} | {summary['score_distribution']['unsafe_median']:.4f} |",
            "",
            "## By Type",
            "",
            "| type | detector | n | balanced_accuracy | safe_accuracy | unsafe_accuracy | fp | fn |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for detector_name, section in [("subspace", "subspace_by_type"), ("model_refusal", "model_refusal_by_type")]:
        for group, row in summary[section].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        group,
                        detector_name,
                        str(row["n"]),
                        fmt(row["balanced_accuracy"]),
                        fmt(row["safe_accuracy"]),
                        fmt(row["unsafe_accuracy"]),
                        str(row["fp"]),
                        str(row["fn"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- XSTest contains both `safe` and `unsafe` prompts, so false positives on safe prompts are measured directly.",
            "- `model_refusal` treats a refusal-style response as an unsafe prediction; this is a lightweight heuristic and should be manually audited for final claims.",
            "- `subspace` uses the selected hidden-layer primary coordinate and the calibrated threshold.",
            "",
            "## Output Files",
            "",
            "- `xstest_results.jsonl`: full prompts, responses, scores, and predictions.",
            "- `xstest_scores.csv`: compact table for error analysis and plotting.",
            "- `xstest_summary.json`: machine-readable metrics.",
        ]
    )
    (out_dir / "xstest_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data", type=Path, default=Path("benchmark/XSTest/xstest_prompts.csv"))
    parser.add_argument("--subspace", type=Path, required=True)
    parser.add_argument("--calibration-activations", type=Path)
    parser.add_argument(
        "--threshold-json",
        type=Path,
        help="JSON produced by calibrate_subspace_threshold.py. Overrides --threshold/--calibration-activations.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--score-layer", default="28", help="Layer id in the fitted subspace, or `last`.")
    parser.add_argument("--pooling", choices=["last", "mean"], default="last")
    parser.add_argument("--threshold", default="auto", help="`auto` or a numeric primary-coordinate threshold.")
    parser.add_argument("--split", choices=["all", "calibration", "test"], default="all")
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--calibration-ratio", type=float, default=0.3)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--system-prompt")
    args = parser.parse_args()

    args.data = args.data.resolve()
    args.subspace = args.subspace.resolve()
    if args.calibration_activations is not None:
        args.calibration_activations = args.calibration_activations.resolve()
    if args.threshold_json is not None:
        args.threshold_json = args.threshold_json.resolve()

    rows = load_xstest_rows(
        args.data,
        max_samples=args.max_samples,
        split=args.split,
        split_seed=args.split_seed,
        calibration_ratio=args.calibration_ratio,
    )
    subspace = load_subspace(args.subspace, args.score_layer)
    threshold_info = resolve_threshold(args, subspace)

    processor, model = load_qwen_vl(args.model, args.dtype, args.device, args.trust_remote_code)
    num_layers = infer_num_layers(model)
    parse_layers(str(subspace["layer"]), num_layers)

    print(f"Loaded {len(rows)} XSTest rows from {args.data}")
    print(f"Labels: {dict(Counter(row['label'] for row in rows))}")
    print(f"Types: {dict(Counter(row['type'] for row in rows))}")
    print(
        f"Using layer {subspace['layer']} threshold={threshold_info['threshold']:.6f} "
        f"source={threshold_info['source']}"
    )

    results = []
    for row in tqdm(rows, desc="xstest-guard-eval"):
        results.append(
            run_one(
                row=row,
                processor=processor,
                model=model,
                subspace=subspace,
                pooling=args.pooling,
                threshold=float(threshold_info["threshold"]),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                system_prompt=args.system_prompt,
            )
        )

    summary = summarize(results, threshold_info, args)
    write_outputs(results, summary, args.out_dir)
    print(f"Wrote XSTest outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
