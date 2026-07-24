from __future__ import annotations

import importlib.metadata
import hashlib
import inspect
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class PreparedInput:
    model_inputs: dict[str, Any]
    rendered_prompt: str
    attention_mask: Any
    image_mask: Any
    sequence_length: int
    image_token_count: int
    image_width: int
    image_height: int


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (
        "torch",
        "transformers",
        "accelerate",
        "qwen-vl-utils",
        "Pillow",
        "numpy",
        "modelscope",
        "safetensors",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype={name!r}")
    return mapping[name]


def resolve_model_reference(
    model: str,
    model_source: str,
    revision: str | None,
    cache_dir: Path | None,
) -> str:
    local = Path(model).expanduser()
    if local.exists():
        return str(local.resolve())
    if model_source in {"hf", "auto"}:
        return model
    if model_source != "modelscope":
        raise ValueError(f"Unsupported model_source={model_source!r}")
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise ImportError("ModelScope loading requires the modelscope package") from exc
    kwargs: dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir.expanduser().resolve())
    return str(snapshot_download(model, **kwargs))


def _loader_kwargs(args: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype(args.dtype),
        "device_map": args.device_map,
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if args.attn_implementation != "auto":
        kwargs["attn_implementation"] = args.attn_implementation
    if args.model_source != "modelscope" and args.model_revision:
        kwargs["revision"] = args.model_revision
    if args.model_cache_dir:
        kwargs["cache_dir"] = str(args.model_cache_dir.expanduser().resolve())
    return kwargs


def load_model_and_processor(args: Any) -> tuple[Any, Any, str]:
    import transformers
    from transformers import AutoProcessor

    resolved = resolve_model_reference(
        args.model,
        model_source=args.model_source,
        revision=args.model_revision,
        cache_dir=args.model_cache_dir,
    )
    processor_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.model_source != "modelscope" and args.model_revision:
        processor_kwargs["revision"] = args.model_revision
    if args.model_cache_dir:
        processor_kwargs["cache_dir"] = str(args.model_cache_dir.expanduser().resolve())
    processor = AutoProcessor.from_pretrained(resolved, **processor_kwargs)
    kwargs = _loader_kwargs(args)
    if args.backend == "qwen2_5_vl":
        cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
        if cls is None:
            raise ImportError("Installed transformers lacks Qwen2_5_VLForConditionalGeneration")
    elif args.backend == "gemma3":
        cls = getattr(transformers, "Gemma3ForConditionalGeneration", None)
        if cls is None:
            cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if cls is None:
            raise ImportError("Installed transformers lacks Gemma3ForConditionalGeneration")
    else:
        raise ValueError(f"Unsupported backend={args.backend!r}")
    model = cls.from_pretrained(resolved, **kwargs)
    model.eval()
    return processor, model, resolved


def _embedding_device(model: Any) -> Any:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def decoder_layers(model: Any) -> tuple[Any, str]:
    candidates = (
        "model.language_model.layers",
        "model.language_model.model.layers",
        "language_model.layers",
        "language_model.model.layers",
        "model.model.layers",
        "model.layers",
    )
    for path in candidates:
        value = model
        for part in path.split("."):
            value = getattr(value, part, None)
            if value is None:
                break
        if value is not None and hasattr(value, "__len__") and len(value) > 0:
            return value, path
    raise RuntimeError("Could not locate decoder layer ModuleList in the model wrapper")


def representation_forward_model(model: Any) -> Any:
    """Return the multimodal base model to avoid materializing full-vocabulary logits."""
    candidate = getattr(model, "model", None)
    if candidate is not None and hasattr(candidate, "language_model") and (
        hasattr(candidate, "visual") or hasattr(candidate, "vision_tower")
    ):
        return candidate
    return model


def parse_layers(value: str, count: int) -> list[int]:
    if value == "all":
        return list(range(1, count + 1))
    aliases = {
        "early": max(1, count // 4),
        "mid": max(1, count // 2),
        "late": max(1, 3 * count // 4),
        "last": count,
    }
    output: list[int] = []
    for item in value.split(","):
        item = item.strip()
        layer = aliases[item] if item in aliases else int(item)
        if layer < 0:
            layer = count + 1 + layer
        if not 1 <= layer <= count:
            raise ValueError(f"Layer {layer} is outside 1..{count}")
        output.append(layer)
    return sorted(set(output))


def _image_token_id(processor: Any, model: Any) -> int | None:
    candidates = [
        getattr(processor, "image_token_id", None),
        getattr(getattr(processor, "tokenizer", None), "image_token_id", None),
        getattr(getattr(model, "config", None), "image_token_id", None),
        getattr(getattr(model, "config", None), "image_token_index", None),
    ]
    for value in candidates:
        if value is not None:
            return int(value)
    return None


def _move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _call_processor_with_mm_types(processor: Any, kwargs: dict[str, Any]) -> Any:
    requested = {**kwargs, "return_mm_token_type_ids": True}
    try:
        return processor(**requested)
    except TypeError as exc:
        if "return_mm_token_type_ids" not in str(exc):
            raise
        return processor(**kwargs)


def _qwen_batch(processor: Any, prompt: str, image_path: Path | None) -> tuple[dict[str, Any], str, Image.Image | None]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError("Qwen2.5-VL inputs require qwen-vl-utils") from exc
    content: list[dict[str, Any]] = []
    image: Image.Image | None = None
    if image_path is not None:
        content.append({"type": "image", "image": str(image_path)})
        image = Image.open(image_path).convert("RGB")
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    kwargs = {
        "text": [rendered],
        "images": images,
        "videos": videos,
        "padding": True,
        "return_tensors": "pt",
    }
    batch = _call_processor_with_mm_types(processor, kwargs)
    return dict(batch), rendered, image


def _gemma_batch(processor: Any, prompt: str, image_path: Path | None) -> tuple[dict[str, Any], str, Image.Image | None]:
    content: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    image: Image.Image | None = None
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
        images.append(image)
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs: dict[str, Any] = {
        "text": [rendered],
        "padding": True,
        "return_tensors": "pt",
    }
    if images:
        kwargs["images"] = images
    batch = _call_processor_with_mm_types(processor, kwargs)
    return dict(batch), rendered, image


def prepare_input(
    processor: Any,
    model: Any,
    backend: str,
    prompt: str,
    image_path: Path | None,
    forward_model: Any | None = None,
) -> PreparedInput:
    import torch

    if backend == "qwen2_5_vl":
        batch, rendered, image = _qwen_batch(processor, prompt, image_path)
    elif backend == "gemma3":
        batch, rendered, image = _gemma_batch(processor, prompt, image_path)
    else:
        raise ValueError(f"Unsupported backend={backend!r}")

    input_ids = batch["input_ids"]
    mm_types = batch.get("mm_token_type_ids")
    token_types = batch.get("token_type_ids")
    forward_parameters = inspect.signature((forward_model or model).forward).parameters
    if image is not None:
        if "pixel_values" not in batch:
            raise RuntimeError(f"{backend} processor omitted pixel_values for a visual sample")
        if backend == "qwen2_5_vl":
            if "mm_token_type_ids" in forward_parameters:
                if mm_types is None or mm_types.shape != input_ids.shape or not bool(mm_types.eq(1).any().item()):
                    raise RuntimeError(
                        "Qwen forward requires non-empty mm_token_type_ids matching input_ids for a visual sample"
                    )
            else:
                batch.pop("mm_token_type_ids", None)
                if "image_grid_thw" not in batch:
                    raise RuntimeError(
                        "Legacy Qwen forward requires image_grid_thw to construct multimodal position IDs"
                    )
        elif backend == "gemma3":
            if "token_type_ids" not in forward_parameters:
                raise RuntimeError("Gemma3 forward does not expose the required token_type_ids argument")
            if token_types is None or token_types.shape != input_ids.shape or not bool(token_types.eq(1).any().item()):
                raise RuntimeError(
                    "Gemma3 processor omitted non-empty token_type_ids required for visual bidirectional attention"
                )
    elif "mm_token_type_ids" not in forward_parameters:
        batch.pop("mm_token_type_ids", None)

    attention = batch.get("attention_mask", torch.ones_like(input_ids)).bool()
    image_mask = None
    if mm_types is not None:
        image_mask = mm_types.bool()
    elif token_types is not None:
        image_mask = token_types.bool()
    if image_mask is None:
        token_id = _image_token_id(processor, model)
        image_mask = input_ids.eq(token_id) if token_id is not None else torch.zeros_like(input_ids, dtype=torch.bool)
    image_mask = image_mask & attention

    device = _embedding_device(model)
    moved = _move_batch(batch, device)
    attention = moved.get("attention_mask", attention.to(device)).bool()
    image_mask = image_mask.to(device)
    width, height = image.size if image is not None else (0, 0)
    if image is not None:
        image.close()
    return PreparedInput(
        model_inputs=moved,
        rendered_prompt=rendered,
        attention_mask=attention,
        image_mask=image_mask,
        sequence_length=int(attention.sum().item()),
        image_token_count=int(image_mask.sum().item()),
        image_width=int(width),
        image_height=int(height),
    )


def forward_kwargs(model: Any) -> dict[str, Any]:
    parameters = inspect.signature(model.forward).parameters
    kwargs: dict[str, Any] = {"use_cache": False, "return_dict": True}
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 1
    return kwargs


def _checkpoint_layout_identity(resolved_model: str) -> str:
    path = Path(resolved_model)
    if not path.is_dir():
        return ""
    entries: list[str] = []
    for pattern in ("*.json", "*.safetensors", "*.bin"):
        for item in sorted(path.glob(pattern)):
            stat = item.stat()
            if item.suffix == ".json":
                digest = hashlib.sha256(item.read_bytes()).hexdigest()
            else:
                digest = ""
            entries.append(f"{item.name}:{stat.st_size}:{stat.st_mtime_ns}:{digest}")
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest() if entries else ""


def _hardware_identity() -> dict[str, Any]:
    import torch

    devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory": int(properties.total_memory),
                }
            )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        "devices": devices,
    }


def runtime_metadata(
    model: Any,
    processor: Any,
    forward_model: Any,
    resolved_model: str,
    layer_path: str,
    args: Any,
) -> dict[str, Any]:
    device_map = getattr(model, "hf_device_map", None)
    config_json = model.config.to_json_string() if hasattr(model.config, "to_json_string") else "{}"
    chat_template = getattr(processor, "chat_template", None) or getattr(
        getattr(processor, "tokenizer", None), "chat_template", ""
    )
    return {
        "model": args.model,
        "resolved_model": resolved_model,
        "model_source": args.model_source,
        "model_revision": args.model_revision,
        "backend": args.backend,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "hf_device_map": {str(key): str(value) for key, value in (device_map or {}).items()},
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": bool(args.trust_remote_code),
        "decoder_layer_path": layer_path,
        "model_class": type(model).__name__,
        "forward_model_class": type(forward_model).__name__,
        "processor_class": type(processor).__name__,
        "checkpoint_commit": getattr(model.config, "_commit_hash", None),
        "checkpoint_layout_sha256": _checkpoint_layout_identity(resolved_model),
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "chat_template_sha256": hashlib.sha256(str(chat_template).encode("utf-8")).hexdigest(),
        "config": json.loads(config_json),
        "packages": package_versions(),
        "hardware": _hardware_identity(),
    }
