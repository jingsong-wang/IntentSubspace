import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def image_surrogate_text(row: dict) -> str | None:
    return row.get("image_prompt") or row.get("image_caption")


def build_text_prompt(row: dict, include_image_surrogate: bool = True) -> str:
    parts = []
    if row.get("prompt_text"):
        parts.append(row["prompt_text"].strip())
    if include_image_surrogate:
        surrogate = image_surrogate_text(row)
        if surrogate:
            parts.append(f"[Image prompt surrogate: {surrogate.strip()}]")
    if not parts:
        parts.append("[Image-only sample; no typed user text is present.]")
    return "\n".join(parts)


def apply_text_chat_template(tokenizer: Any, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"User: {content}\nAssistant:"


def resolve_image_for_runtime(path: str, image_base_dir: Path) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((image_base_dir / p).resolve())


def qwen_message(row: dict, use_image_surrogate_when_missing: bool, image_base_dir: Path) -> list[dict]:
    content = []
    image_path = row.get("image_path")
    if image_path:
        content.append({"type": "image", "image": resolve_image_for_runtime(str(image_path), image_base_dir)})

    text = row.get("prompt_text") or ""
    if use_image_surrogate_when_missing and not image_path:
        surrogate = image_surrogate_text(row)
        if surrogate:
            text = (text + "\n" if text else "") + f"[Image prompt surrogate: {surrogate}]"
    if text:
        content.append({"type": "text", "text": text})
    if not content:
        content.append({"type": "text", "text": "[Image-only sample; image path is missing.]"})
    return [{"role": "user", "content": content}]


def parse_layers(layer_arg: str, num_layers: int) -> list[int]:
    if layer_arg == "all":
        return list(range(1, num_layers + 1))

    aliases = {
        "early": max(1, num_layers // 4),
        "mid": max(1, num_layers // 2),
        "late": max(1, (3 * num_layers) // 4),
        "last": num_layers,
    }
    layers = []
    for item in layer_arg.split(","):
        item = item.strip()
        if item in aliases:
            layers.append(aliases[item])
        else:
            idx = int(item)
            if idx < 0:
                idx = num_layers + 1 + idx
            layers.append(idx)

    for idx in layers:
        if idx < 0 or idx > num_layers:
            raise ValueError(f"Layer {idx} out of range 0..{num_layers}")
    return sorted(set(layers))


def infer_num_layers(model: Any) -> int:
    candidates = [
        getattr(model.config, "num_hidden_layers", None),
        getattr(getattr(model.config, "text_config", None), "num_hidden_layers", None),
        getattr(getattr(model.config, "llm_config", None), "num_hidden_layers", None),
    ]
    for value in candidates:
        if value:
            return int(value)
    raise RuntimeError("Could not infer num_hidden_layers from model config.")


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor | None, pooling: str) -> torch.Tensor:
    # hidden: [1, seq, dim]
    if attention_mask is None:
        valid = hidden[0]
    else:
        mask = attention_mask[0].bool().to(hidden.device)
        valid = hidden[0, mask]
    if pooling == "last":
        return valid[-1]
    if pooling == "mean":
        return valid.mean(dim=0)
    raise ValueError(f"Unknown pooling mode: {pooling}")


def dtype_from_arg(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return "auto"


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def first_param_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_text_backend(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if args.device == "auto" else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_from_arg(args.dtype),
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if args.device != "auto":
        model.to(torch.device(args.device))
    model.eval()
    return tokenizer, model


def load_qwen_vl_backend(args):
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Qwen2.5-VL support requires a recent transformers build with "
            "Qwen2_5_VLForConditionalGeneration."
        ) from exc

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    device_map = "auto" if args.device == "auto" else None
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype_from_arg(args.dtype),
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if args.device != "auto":
        model.to(torch.device(args.device))
    model.eval()
    return processor, model


def qwen_process_messages(processor: Any, messages: list[dict]) -> tuple[dict, str]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError(
            "Install qwen-vl-utils to process Qwen2.5-VL image messages: "
            "python -m pip install qwen-vl-utils"
        ) from exc

    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    batch = processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return batch, rendered


def extract_text_rows(args, rows: list[dict], layers: list[int], tokenizer: Any, model: torch.nn.Module) -> tuple[np.ndarray, list[str]]:
    acts = []
    rendered_prompts = []
    with torch.no_grad():
        for row in tqdm(rows, desc="extract-text"):
            content = build_text_prompt(row, include_image_surrogate=True)
            rendered = apply_text_chat_template(tokenizer, content)
            rendered_prompts.append(rendered)
            batch = tokenizer(rendered, return_tensors="pt", padding=False, truncation=False)
            batch = move_batch_to_device(batch, first_param_device(model))
            out = model(**batch, output_hidden_states=True, use_cache=False)
            sample_layers = []
            for layer_idx in layers:
                pooled = pool_hidden(
                    out.hidden_states[layer_idx].detach().float().cpu(),
                    batch.get("attention_mask", None).detach().cpu() if "attention_mask" in batch else None,
                    args.pooling,
                )
                sample_layers.append(pooled.numpy())
            acts.append(np.stack(sample_layers, axis=0))
    return np.stack(acts, axis=0), rendered_prompts


def extract_qwen_vl_rows(args, rows: list[dict], layers: list[int], processor: Any, model: torch.nn.Module) -> tuple[np.ndarray, list[str]]:
    acts = []
    rendered_prompts = []
    with torch.no_grad():
        for row in tqdm(rows, desc="extract-qwen2.5-vl"):
            messages = qwen_message(row, use_image_surrogate_when_missing=args.allow_image_surrogate, image_base_dir=args.image_base_dir)
            batch, rendered = qwen_process_messages(processor, messages)
            rendered_prompts.append(rendered)
            batch = move_batch_to_device(dict(batch), first_param_device(model))
            out = model(**batch, output_hidden_states=True, use_cache=False)
            sample_layers = []
            for layer_idx in layers:
                pooled = pool_hidden(
                    out.hidden_states[layer_idx].detach().float().cpu(),
                    batch.get("attention_mask", None).detach().cpu() if "attention_mask" in batch else None,
                    args.pooling,
                )
                sample_layers.append(pooled.numpy())
            acts.append(np.stack(sample_layers, axis=0))
    return np.stack(acts, axis=0), rendered_prompts


def infer_backend(args) -> str:
    if args.backend != "auto":
        return args.backend
    name = args.model.lower()
    if "qwen2.5-vl" in name or "qwen2_5_vl" in name or "vl-" in name:
        return "qwen2_5_vl"
    return "text"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model name or local path.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=["auto", "text", "qwen2_5_vl"], default="auto")
    parser.add_argument("--layers", default="mid,last", help="Comma list, e.g. 12,24,last or aliases early,mid,late,last or all.")
    parser.add_argument("--pooling", choices=["last", "mean"], default="last")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--allow-image-surrogate",
        action="store_true",
        help="For VLM backend, use image_prompt text when image_path is empty. Useful before real images are generated.",
    )
    parser.add_argument("--image-base-dir", type=Path, default=Path("."), help="Base directory used to resolve relative image_path entries.")
    args = parser.parse_args()
    args.image_base_dir = args.image_base_dir.resolve()

    rows = load_jsonl(args.data)
    backend = infer_backend(args)

    if backend == "text":
        tokenizer, model = load_text_backend(args)
        num_layers = infer_num_layers(model)
        layers = parse_layers(args.layers, num_layers)
        activations, rendered_prompts = extract_text_rows(args, rows, layers, tokenizer, model)
    elif backend == "qwen2_5_vl":
        processor, model = load_qwen_vl_backend(args)
        num_layers = infer_num_layers(model)
        layers = parse_layers(args.layers, num_layers)
        activations, rendered_prompts = extract_qwen_vl_rows(args, rows, layers, processor, model)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        activations=activations,
        layers=np.array(layers, dtype=np.int32),
        ids=np.array([r["id"] for r in rows]),
        labels=np.array([int(r["label"]) for r in rows], dtype=np.int32),
        conditions=np.array([r["condition"] for r in rows]),
        pair_keys=np.array([r["pair_key"] for r in rows]),
        intent_ids=np.array([r.get("intent_id", "") for r in rows]),
        intent_texts=np.array([r.get("intent_text", "") for r in rows]),
        intent_families=np.array([r.get("intent_family", "") for r in rows]),
        image_roles=np.array([r.get("image_role", "") for r in rows]),
        image_paths=np.array([r.get("image_path") or "" for r in rows]),
        sources=np.array([r.get("source", "") for r in rows]),
        prompts=np.array(rendered_prompts),
        metadata_json=json.dumps(
            {"model": args.model, "backend": backend, "pooling": args.pooling, "layers": layers},
            ensure_ascii=False,
        ),
    )
    print(f"Wrote activations {activations.shape} to {args.out}")


if __name__ == "__main__":
    main()
