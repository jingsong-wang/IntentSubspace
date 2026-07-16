import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: Any):
        return iterable

from extract_activations import (
    apply_text_chat_template,
    build_text_prompt,
    dtype_from_arg,
    first_param_device,
    generic_vlm_message,
    generic_vlm_process_messages,
    infer_backend,
    load_generic_vlm_backend,
    load_qwen_vl_backend,
    load_text_backend,
    move_batch_to_device,
    qwen_message,
    qwen_process_messages,
    resolve_image_for_runtime,
    resolve_model_path,
)
from run_hades_dynamic_guard import response_refused


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def decode_generated(processor_or_tokenizer: Any, batch: dict[str, Any], generated_ids: Any) -> str:
    input_len = int(batch["input_ids"].shape[1])
    generated_trimmed = generated_ids[:, input_len:]
    decoded = processor_or_tokenizer.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip() if decoded else ""


def normalized_prompt(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt_text") or row.get("prompt") or row.get("user_prompt") or "").strip()
    if prompt:
        return prompt
    surrogate = str(row.get("image_prompt") or row.get("image_caption") or row.get("ocr_source_text") or "").strip()
    if surrogate:
        return f"[image-carried request] {surrogate}"
    return "[image-only sample]"


def maybe_insert_system(messages: list[dict[str, Any]], system_prompt: str | None) -> list[dict[str, Any]]:
    if not system_prompt:
        return messages
    return [{"role": "system", "content": system_prompt}, *messages]


def build_text_batch(row: dict[str, Any], tokenizer: Any, system_prompt: str | None) -> tuple[dict[str, Any], str]:
    content = build_text_prompt(row, include_image_surrogate=True)
    if system_prompt:
        content = f"{system_prompt}\n\n{content}"
    rendered = apply_text_chat_template(tokenizer, content)
    batch = tokenizer(rendered, return_tensors="pt", padding=False, truncation=False)
    return dict(batch), rendered


def build_batch(
    row: dict[str, Any],
    backend: str,
    processor_or_tokenizer: Any,
    image_base_dir: Path,
    allow_image_surrogate: bool,
    system_prompt: str | None,
) -> tuple[dict[str, Any], str]:
    if backend == "text":
        return build_text_batch(row, processor_or_tokenizer, system_prompt)
    if backend == "qwen2_5_vl":
        messages = qwen_message(
            row,
            use_image_surrogate_when_missing=allow_image_surrogate,
            image_base_dir=image_base_dir,
        )
        messages = maybe_insert_system(messages, system_prompt)
        batch, rendered = qwen_process_messages(processor_or_tokenizer, messages)
        return dict(batch), rendered
    if backend == "generic_vlm":
        messages, images = generic_vlm_message(
            row,
            use_image_surrogate_when_missing=allow_image_surrogate,
            image_base_dir=image_base_dir,
        )
        messages = maybe_insert_system(messages, system_prompt)
        batch, rendered = generic_vlm_process_messages(processor_or_tokenizer, messages, images)
        return dict(batch), rendered
    raise ValueError(f"Unsupported backend: {backend}")


def resolve_output_image_path(row: dict[str, Any], image_base_dir: Path) -> str:
    image_path = row.get("image_path")
    if not image_path:
        return ""
    return resolve_image_for_runtime(str(image_path), image_base_dir)


def run_one(
    row: dict[str, Any],
    row_index: int,
    backend: str,
    processor_or_tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    batch, rendered = build_batch(
        row=row,
        backend=backend,
        processor_or_tokenizer=processor_or_tokenizer,
        image_base_dir=args.image_base_dir,
        allow_image_surrogate=args.allow_image_surrogate,
        system_prompt=args.system_prompt,
    )
    batch = move_batch_to_device(batch, first_param_device(model))

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
    }
    if args.temperature > 0.0:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        generated = model.generate(**batch, **gen_kwargs)
    response = decode_generated(processor_or_tokenizer, batch, generated)
    refused = response_refused(response)
    label = int(row.get("label", 0))

    return {
        "row_index": row_index,
        "id": str(row.get("id", row_index)),
        "pair_key": str(row.get("pair_key", "")),
        "dataset": str(row.get("dataset", "intent_probe")),
        "source": str(row.get("source", "")),
        "model": args.model_alias or args.model,
        "model_id": args.model,
        "backend": backend,
        "label": label,
        "label_name": str(row.get("label_name", "target" if label == 1 else "benign_control")),
        "condition": str(row.get("condition", "")),
        "intent_family": str(row.get("intent_family", "")),
        "intent_id": str(row.get("intent_id", "")),
        "intent_text": str(row.get("intent_text", "")),
        "image_role": str(row.get("image_role", "")),
        "image_path": resolve_output_image_path(row, args.image_base_dir),
        "prompt": normalized_prompt(row),
        "prompt_text": str(row.get("prompt_text", "")),
        "ocr_source_text": str(row.get("ocr_source_text", "")),
        "rendered_prompt": rendered,
        "response_source": "model",
        "response": response,
        "refusal_heuristic": refused,
        "heuristic_attack_success": bool(label == 1 and not refused),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }


def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    total = len(results)
    refused = sum(1 for row in results if row["refusal_heuristic"])
    nonrefused = total - refused
    summary: dict[str, Any] = {
        "n": total,
        "model": args.model,
        "model_alias": args.model_alias,
        "backend": args.resolved_backend,
        "data": str(args.data),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "overall": {
            "refusal_heuristic_count": refused,
            "refusal_heuristic_rate": refused / total if total else float("nan"),
            "nonrefusal_count": nonrefused,
            "nonrefusal_rate": nonrefused / total if total else float("nan"),
        },
        "by_label": {},
        "by_condition": {},
        "by_intent_family": {},
    }
    for key, out_key in [
        ("label_name", "by_label"),
        ("condition", "by_condition"),
        ("intent_family", "by_intent_family"),
    ]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in results:
            groups[str(row.get(key, ""))].append(row)
        for group, rows in sorted(groups.items()):
            n = len(rows)
            r = sum(1 for row in rows if row["refusal_heuristic"])
            summary[out_key][group] = {
                "n": n,
                "refusal_heuristic_rate": r / n if n else float("nan"),
                "label_counts": dict(Counter(str(row.get("label_name", "")) for row in rows)),
            }
    return summary


def write_outputs(results: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "generation_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (out_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_fields = [
        "row_index",
        "id",
        "model",
        "label_name",
        "condition",
        "intent_family",
        "intent_id",
        "image_role",
        "refusal_heuristic",
        "heuristic_attack_success",
        "image_path",
    ]
    with (out_dir / "generation_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in csv_fields})

    lines = [
        "# Training Probe Generation Report",
        "",
        f"Samples: `{summary['n']}`",
        f"Model: `{summary['model']}`",
        f"Backend: `{summary['backend']}`",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in summary["overall"].items():
        lines.append(f"| {key} | {value:.4f} |" if isinstance(value, float) else f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `generation_results.jsonl`: prompts, metadata, responses, and refusal heuristic.",
            "- `generation_scores.csv`: compact refusal summary.",
            "- `generation_summary.json`: aggregate generation behavior.",
        ]
    )
    (out_dir / "generation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face/ModelScope model id or local path.")
    parser.add_argument("--model-alias", default="")
    parser.add_argument("--model-source", choices=["hf", "modelscope"], default="hf")
    parser.add_argument("--model-revision")
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["auto", "text", "qwen2_5_vl", "generic_vlm"], default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--system-prompt")
    parser.add_argument("--allow-image-surrogate", action="store_true")
    parser.add_argument("--image-base-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    args.data = args.data.resolve()
    args.image_base_dir = args.image_base_dir.resolve()
    if args.model_cache_dir is not None:
        args.model_cache_dir = args.model_cache_dir.resolve()
    args.resolved_model = resolve_model_path(args)
    args.resolved_backend = infer_backend(args)

    rows = load_jsonl(args.data)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    if args.resolved_backend == "text":
        processor_or_tokenizer, model = load_text_backend(args)
    elif args.resolved_backend == "qwen2_5_vl":
        processor_or_tokenizer, model = load_qwen_vl_backend(args)
    elif args.resolved_backend == "generic_vlm":
        processor_or_tokenizer, model = load_generic_vlm_backend(args)
    else:
        raise ValueError(f"Unsupported backend: {args.resolved_backend}")

    results = []
    for row_index, row in enumerate(tqdm(rows, desc=f"generate-{args.model_alias or args.model}")):
        results.append(
            run_one(
                row=row,
                row_index=row_index,
                backend=args.resolved_backend,
                processor_or_tokenizer=processor_or_tokenizer,
                model=model,
                args=args,
            )
        )

    summary = summarize(results, args)
    write_outputs(results, summary, args.out_dir)
    print(f"Wrote training probe generations to {args.out_dir}")


if __name__ == "__main__":
    main()
