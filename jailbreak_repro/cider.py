from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import append_jsonl_row, read_jsonl, repo_root, write_json
from .models import resolve_model_reference

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable, **_: Any):
        return iterable


CIDER_OFFICIAL_THRESHOLD = -0.003936767578125
CIDER_OFFICIAL_STEPS = (0, 50, 100, 150, 200, 250, 300, 350)
CIDER_HARD_REFUSAL = "I'm sorry, but I can not assist."
CIDER_PAPER_ENCODER = "llava-hf/llava-1.5-7b-hf"
CIDER_FORMAT_VERSION = "cider_official_pipeline_v2_fixed_llava15"


def default_cider_source_dir() -> Path:
    return repo_root() / "jailbreak_repro" / "sourcecode" / "CIDER-main"


@dataclass(frozen=True)
class CiderConfig:
    source_dir: Path
    artifact_dir: Path
    threshold: float = CIDER_OFFICIAL_THRESHOLD
    denoiser: str = "diffusion"
    denoise_steps: tuple[int, ...] = CIDER_OFFICIAL_STEPS
    denoise_batch_size: int = 50
    diffusion_checkpoint: Path | None = None
    dncnn_checkpoint: Path | None = None
    encoder_mode: str = "paper_llava15"
    encoder_model: str = CIDER_PAPER_ENCODER
    encoder_source: str = "hf"
    encoder_revision: str | None = None
    encoder_cache_dir: Path | None = None
    encoder_batch_size: int = 8
    dtype: str = "float16"
    device: str = "auto"
    seed: int = 0
    calibration_image_dir: Path | None = None
    calibration_text_file: Path | None = None
    calibration_pass_rate: float = 0.95

    @property
    def effective_diffusion_checkpoint(self) -> Path:
        return self.diffusion_checkpoint or (
            self.source_dir
            / "code"
            / "models"
            / "diffusion_denoiser"
            / "imagenet"
            / "256x256_diffusion_uncond.pt"
        )

    @property
    def effective_dncnn_checkpoint(self) -> Path:
        return self.dncnn_checkpoint or (
            self.source_dir / "code" / "models" / "DnCNN" / "checkpoint.pth.tar"
        )


def parse_cider_steps(value: str | list[int] | tuple[int, ...], denoiser: str) -> tuple[int, ...]:
    if isinstance(value, str):
        steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        steps = tuple(int(item) for item in value)
    if denoiser == "dncnn":
        if steps not in {CIDER_OFFICIAL_STEPS, (0, 1)}:
            raise ValueError("CIDER DnCNN supports exactly the original and one denoised checkpoint: 0,1")
        return (0, 1)
    if not steps or steps[0] != 0 or any(step < 0 for step in steps):
        raise ValueError("CIDER denoise steps must start at 0 and contain non-negative integers")
    if tuple(sorted(set(steps))) != steps:
        raise ValueError("CIDER denoise steps must be unique and increasing")
    return steps


def cider_detect(cosine_similarities: list[float], threshold: float) -> dict[str, Any]:
    if not cosine_similarities:
        raise ValueError("CIDER detection requires at least the original-image cosine similarity")
    baseline = float(cosine_similarities[0])
    deltas = [float(value) - baseline for value in cosine_similarities]
    lowest_index = min(range(len(deltas)), key=deltas.__getitem__)
    detected = deltas[lowest_index] < threshold
    selected_index = lowest_index if detected else 0
    return {
        "detected": detected,
        "deltas": deltas,
        "lowest_index": lowest_index,
        "selected_index": selected_index,
        "minimum_delta": deltas[lowest_index],
    }


def calibrate_cider_threshold(delta_values: list[float], pass_rate: float = 0.95) -> float:
    if not 0.0 < pass_rate < 1.0:
        raise ValueError("CIDER calibration pass rate must be between 0 and 1")
    if not delta_values:
        raise ValueError("CIDER calibration produced no denoised-image cosine deltas")
    import numpy as np

    return float(np.percentile(delta_values, (1.0 - pass_rate) * 100.0))


def _config_attr(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def verify_paper_llava15_7b(model: Any) -> dict[str, Any]:
    """Verify the architecture signature used by CIDER's paper detector."""
    config = model.config
    text_config = _config_attr(config, "text_config")
    vision_config = _config_attr(config, "vision_config")
    details = {
        "model_class": type(model).__name__,
        "model_type": _config_attr(config, "model_type"),
        "architectures": list(_config_attr(config, "architectures") or []),
        "text_hidden_size": _config_attr(text_config, "hidden_size"),
        "text_num_hidden_layers": _config_attr(text_config, "num_hidden_layers"),
        "vision_hidden_size": _config_attr(vision_config, "hidden_size"),
        "vision_num_hidden_layers": _config_attr(vision_config, "num_hidden_layers"),
    }
    details["paper_llava15_7b"] = bool(
        details["model_class"] == "LlavaForConditionalGeneration"
        and details["model_type"] == "llava"
        and details["text_hidden_size"] == 4096
        and details["text_num_hidden_layers"] == 32
        and details["vision_hidden_size"] == 1024
        and details["vision_num_hidden_layers"] == 24
    )
    return details


def resolve_llava_component(model: Any, name: str) -> Any:
    """Resolve LLaVA modules across old and current Transformers layouts."""
    component = getattr(model, name, None)
    if component is not None:
        return component
    inner_model = getattr(model, "model", None)
    component = getattr(inner_model, name, None) if inner_model is not None else None
    if component is not None:
        return component
    raise AttributeError(
        f"Loaded {type(model).__name__} exposes neither .{name} nor .model.{name}"
    )


def _cleanup_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _torch_dtype(name: str):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }[name]


def _runtime_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _stable_image_key(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _config_fingerprint(config: CiderConfig) -> str:
    payload = {
        "format_version": CIDER_FORMAT_VERSION,
        "denoiser": config.denoiser,
        "denoise_steps": list(config.denoise_steps),
        "denoise_batch_size": config.denoise_batch_size,
        "diffusion_checkpoint": str(config.effective_diffusion_checkpoint.expanduser().resolve()),
        "dncnn_checkpoint": str(config.effective_dncnn_checkpoint.expanduser().resolve()),
        "encoder_model": config.encoder_model,
        "encoder_mode": config.encoder_mode,
        "encoder_source": config.encoder_source,
        "encoder_revision": config.encoder_revision,
        "encoder_dtype": config.dtype,
        "encoder_batch_size": config.encoder_batch_size,
        "device": config.device,
        "seed": config.seed,
        "threshold": config.threshold,
        "calibration_image_dir": str(config.calibration_image_dir.expanduser().resolve()) if config.calibration_image_dir else None,
        "calibration_text_file": str(config.calibration_text_file.expanduser().resolve()) if config.calibration_text_file else None,
        "calibration_pass_rate": config.calibration_pass_rate,
    }
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _save_rgb_atomic(image: Any, path: Path, image_format: str = "JPEG") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    image.save(temporary, format=image_format)
    temporary.replace(path)


class _DiffusionDenoiser:
    def __init__(self, config: CiderConfig) -> None:
        import torch

        checkpoint = config.effective_diffusion_checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "Missing CIDER diffusion checkpoint: "
                f"{checkpoint}. Download 256x256_diffusion_uncond.pt as documented in "
                "jailbreak_repro/sourcecode/CIDER-main/README.md or pass --cider-diffusion-checkpoint."
            )
        code_dir = config.source_dir / "code"
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        from models.diffusion_denoiser.imagenet.guided_diffusion.script_util import (  # type: ignore
            args_to_dict,
            create_model_and_diffusion,
            model_and_diffusion_defaults,
        )

        defaults = {
            "image_size": 256,
            "num_channels": 256,
            "num_res_blocks": 2,
            "num_heads": 4,
            "num_heads_upsample": -1,
            "num_head_channels": 64,
            "attention_resolutions": "32,16,8",
            "channel_mult": "",
            "dropout": 0.0,
            "class_cond": False,
            "use_checkpoint": False,
            "use_scale_shift_norm": True,
            "resblock_updown": True,
            "use_fp16": False,
            "use_new_attention_order": False,
            "clip_denoised": True,
            "learn_sigma": True,
            "diffusion_steps": 1000,
            "noise_schedule": "linear",
            "timestep_respacing": None,
            "use_kl": False,
            "predict_xstart": False,
            "rescale_timesteps": False,
            "rescale_learned_sigmas": False,
        }
        args = type("CiderDiffusionArgs", (), defaults)()
        self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        self.device = _runtime_device(config.device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)
        self.seed = config.seed

    def checkpoints(self, image_path: Path, output_dir: Path, steps: tuple[int, ...]) -> list[Path]:
        return self.checkpoints_many(
            [(image_path, output_dir)],
            steps,
            batch_size=1,
        )[image_path]

    def checkpoints_many(
        self,
        items: list[tuple[Path, Path]],
        steps: tuple[int, ...],
        batch_size: int,
    ) -> dict[Path, list[Path]]:
        import numpy as np
        import torch
        from PIL import Image

        records = []
        for image_path, output_dir in items:
            paths = [output_dir / f"checkpoint_{step:03d}.jpg" for step in steps]
            with Image.open(image_path) as opened:
                resized = opened.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / 255.0 * 2.0 - 1.0
            records.append(
                {
                    "image_path": image_path,
                    "paths": paths,
                    "resized": resized,
                    "original": torch.from_numpy(array).permute(2, 0, 1),
                    "seed": int(_stable_image_key(image_path)[:8], 16),
                }
            )

        progress = tqdm(
            total=len(records) * len(steps),
            desc="CIDER denoising checkpoints",
            unit="checkpoint",
            dynamic_ncols=True,
        )
        try:
            for record in records:
                path = record["paths"][0]
                if not _valid_image(path):
                    _save_rgb_atomic(record["resized"], path)
                progress.update(1)

            for step_index, step in enumerate(steps[1:], start=1):
                pending = [record for record in records if not _valid_image(record["paths"][step_index])]
                progress.update(len(records) - len(pending))
                for offset in range(0, len(pending), batch_size):
                    chunk = pending[offset : offset + batch_size]
                    originals = torch.stack([record["original"] for record in chunk]).to(self.device)
                    noises = []
                    for record in chunk:
                        generator = torch.Generator(device=self.device)
                        generator.manual_seed(self.seed + record["seed"] + step)
                        noises.append(
                            torch.randn(
                                originals[0].shape,
                                generator=generator,
                                device=self.device,
                                dtype=originals.dtype,
                            )
                        )
                    noise = torch.stack(noises)
                    timestep = torch.full((len(chunk),), step, device=self.device, dtype=torch.long)
                    noisy = self.diffusion.q_sample(x_start=originals, t=timestep, noise=noise)
                    with torch.inference_mode():
                        denoised = self.diffusion.p_sample(
                            self.model,
                            noisy,
                            timestep,
                            clip_denoised=True,
                        )["pred_xstart"]
                    outputs = denoised.detach().cpu().permute(0, 2, 3, 1).numpy()
                    outputs = ((outputs + 1.0) / 2.0 * 255.0).astype(np.uint8)
                    for record, output in zip(chunk, outputs):
                        _save_rgb_atomic(Image.fromarray(output), record["paths"][step_index])
                    progress.update(len(chunk))
        finally:
            progress.close()
        return {record["image_path"]: record["paths"] for record in records}

    def close(self) -> None:
        self.model = None
        self.diffusion = None
        _cleanup_memory()


class _DnCNNDenoiser:
    def __init__(self, config: CiderConfig) -> None:
        import torch

        checkpoint = config.effective_dncnn_checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing CIDER DnCNN checkpoint: {checkpoint}")
        code_dir = config.source_dir / "code"
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        from models.DnCNN.DnCNN import DnCNN  # type: ignore

        self.device = _runtime_device(config.device)
        self.model = DnCNN(image_channels=3, depth=17, n_channels=64)
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        state_dict = state.get("state_dict", state)
        normalized = {key.removeprefix("module."): value for key, value in state_dict.items()}
        self.model.load_state_dict(normalized)
        self.model.eval().to(self.device)

    def checkpoints(self, image_path: Path, output_dir: Path, steps: tuple[int, ...]) -> list[Path]:
        import numpy as np
        import torch
        from PIL import Image

        if steps != (0, 1):
            raise ValueError("CIDER DnCNN requires denoise steps 0,1")
        paths = [output_dir / "checkpoint_000.jpg", output_dir / "checkpoint_001.jpg"]
        if all(_valid_image(path) for path in paths):
            return paths
        with Image.open(image_path) as opened:
            resized = opened.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        if not _valid_image(paths[0]):
            _save_rgb_atomic(resized, paths[0])
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model(tensor).clamp(0.0, 1.0)[0].cpu().permute(1, 2, 0).numpy()
        _save_rgb_atomic(Image.fromarray((output * 255.0).astype(np.uint8)), paths[1])
        return paths

    def close(self) -> None:
        self.model = None
        _cleanup_memory()


class _LlavaCrossModalEncoder:
    def __init__(self, config: CiderConfig) -> None:
        # Public LLaVA shards can return 401 from Xet/CAS on tokenless servers.
        # Force the standard Hub download path without changing model contents.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            from huggingface_hub import constants as hub_constants

            hub_constants.HF_HUB_DISABLE_XET = True
        except ImportError:
            pass
        import torch
        from transformers import AutoImageProcessor, AutoModelForPreTraining, AutoTokenizer

        if config.encoder_mode not in {"paper_llava15", "custom_llava_ablation"}:
            raise ValueError(f"Unsupported CIDER encoder mode: {config.encoder_mode}")
        resolved = resolve_model_reference(
            config.encoder_model,
            model_source=config.encoder_source,
            model_revision=config.encoder_revision,
            model_cache_dir=config.encoder_cache_dir,
        )
        device_map = "auto" if config.device == "auto" else None
        self.model = AutoModelForPreTraining.from_pretrained(
            resolved,
            torch_dtype=_torch_dtype(config.dtype),
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        if config.device != "auto":
            self.model.to(config.device)
        self.model.eval()
        self.verification = verify_paper_llava15_7b(self.model)
        self.paper_encoder_verified = bool(self.verification["paper_llava15_7b"])
        if config.encoder_mode == "paper_llava15" and not self.paper_encoder_verified:
            loaded_signature = json.dumps(self.verification, ensure_ascii=True, sort_keys=True)
            self.model = None
            _cleanup_memory()
            raise ValueError(
                "CIDER paper_llava15 mode requires the paper's LLaVA-1.5-7B auxiliary encoder; "
                f"loaded signature: {loaded_signature}. "
                "Use --cider-encoder-mode custom_llava_ablation only for a clearly labeled ablation."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.image_processor = AutoImageProcessor.from_pretrained(resolved)
        self.torch = torch
        device_map_value = getattr(self.model, "hf_device_map", None)
        print(
            "CIDER auxiliary encoder loaded: "
            f"mode={config.encoder_mode} model={config.encoder_model} "
            f"paper_llava15_7b={self.paper_encoder_verified} "
            f"device_map={device_map_value or str(self.model.device)}"
        )

    def text_embedding(self, text: str):
        embedding_layer = self.model.get_input_embeddings()
        device = next(embedding_layer.parameters()).device
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(device)
        with self.torch.inference_mode():
            embedding = embedding_layer(input_ids).mean(dim=1).float().cpu()
        return embedding

    def image_embedding(self, image_path: Path):
        return self.image_embeddings([image_path], batch_size=1)[0]

    def image_embeddings(self, image_paths: list[Path], batch_size: int) -> list[Any]:
        from PIL import Image

        vision_tower = resolve_llava_component(self.model, "vision_tower")
        projector = resolve_llava_component(self.model, "multi_modal_projector")
        vision_device = next(vision_tower.parameters()).device
        vision_dtype = next(vision_tower.parameters()).dtype
        embeddings = []
        for offset in range(0, len(image_paths), batch_size):
            images = []
            for image_path in image_paths[offset : offset + batch_size]:
                with Image.open(image_path) as opened:
                    images.append(opened.convert("RGB"))
            pixels = self.image_processor(images, return_tensors="pt").pixel_values
            pixels = pixels.to(device=vision_device, dtype=vision_dtype)
            with self.torch.inference_mode():
                outputs = vision_tower(pixels, output_hidden_states=True)
                selected = outputs.hidden_states[self.model.config.vision_feature_layer][:, 1:]
                projected = projector(selected)
                batch_embeddings = projected.mean(dim=1).float().cpu()
            embeddings.extend(batch_embeddings[index : index + 1] for index in range(len(batch_embeddings)))
        return embeddings

    def cosine(self, text_embedding: Any, image_embedding: Any) -> float:
        return float(self.torch.nn.functional.cosine_similarity(text_embedding, image_embedding).item())

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        _cleanup_memory()


def _sample_prompt(sample: dict[str, Any]) -> str:
    return str(sample.get("prompt_text") or sample.get("attack_prompt_text") or sample.get("prompt") or "")


def _load_calibration_queries(path: Path) -> list[str]:
    queries = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if len(row) == 1 or row[1].strip().lower() == "standard":
                queries.append(row[0].strip())
    if not queries:
        raise ValueError(f"CIDER calibration text file has no standard harmful queries: {path}")
    return queries


def _image_files(path: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(item.resolve() for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)
    if not images:
        raise FileNotFoundError(f"CIDER image directory contains no supported images: {path}")
    return images


def _cider_metadata_valid(row: dict[str, Any]) -> bool:
    if not row.get("cider_preprocessed"):
        return False
    selected = str(row.get("cider_processed_image_path") or "")
    return not selected or _valid_image(Path(selected))


def _cached_detection_matches(sample: dict[str, Any], row: dict[str, Any], config_fingerprint: str) -> bool:
    sample_image = str(Path(str(sample.get("image_path") or "")).expanduser().resolve()) if sample.get("image_path") else ""
    cached_image = str(row.get("cider_original_image_path") or "")
    return (
        _cider_metadata_valid(row)
        and row.get("cider_config_fingerprint") == config_fingerprint
        and sample_image == cached_image
        and _sample_prompt(sample) == _sample_prompt(row)
    )


def prepare_cider_samples(
    samples: list[dict[str, Any]],
    config: CiderConfig,
    resume: bool = True,
) -> list[dict[str, Any]]:
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.artifact_dir / "detections.jsonl"
    config_fingerprint = _config_fingerprint(config)
    cached: dict[str, dict[str, Any]] = {}
    if resume and cache_path.exists():
        for row in read_jsonl(cache_path):
            if _cider_metadata_valid(row):
                cached[str(row["id"])] = row
    elif cache_path.exists():
        cache_path.unlink()

    prepared: list[dict[str, Any] | None] = [None] * len(samples)
    pending: list[tuple[int, dict[str, Any]]] = []
    for index, sample in enumerate(samples):
        cached_row = cached.get(str(sample.get("id")))
        if cached_row is not None and _cached_detection_matches(sample, cached_row, config_fingerprint):
            prepared[index] = {**sample, **cached_row}
            continue
        if not str(sample.get("image_path") or ""):
            row = {
                **sample,
                "cider_preprocessed": True,
                "cider_detected": False,
                "cider_skip_reason": "no_image",
                "cider_processed_image_path": "",
                "cider_threshold": config.threshold,
                "cider_encoder_mode": config.encoder_mode,
                "cider_encoder_role": "fixed_auxiliary_detector",
                "cider_uses_victim_encoder": False,
                "cider_seed": config.seed,
                "cider_config_fingerprint": config_fingerprint,
                "cider_format_version": CIDER_FORMAT_VERSION,
            }
            append_jsonl_row(cache_path, row)
            prepared[index] = row
            continue
        pending.append((index, sample))

    if not pending:
        return [row for row in prepared if row is not None]

    image_paths = {Path(str(sample["image_path"])).expanduser().resolve() for _, sample in pending}
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(f"CIDER input image does not exist: {path}")

    calibration_images: list[Path] = []
    calibration_queries: list[str] = []
    if config.calibration_image_dir is not None or config.calibration_text_file is not None:
        if config.calibration_image_dir is None or config.calibration_text_file is None:
            raise ValueError("CIDER calibration requires both image directory and text file")
        calibration_images = _image_files(config.calibration_image_dir.resolve())
        calibration_queries = _load_calibration_queries(config.calibration_text_file.resolve())
        image_paths.update(calibration_images)

    denoised: dict[Path, list[Path]] = {}
    denoiser = _DiffusionDenoiser(config) if config.denoiser == "diffusion" else _DnCNNDenoiser(config)
    try:
        denoise_items = [
            (image_path, config.artifact_dir / "denoised" / _stable_image_key(image_path))
            for image_path in sorted(image_paths)
        ]
        if hasattr(denoiser, "checkpoints_many"):
            denoised = denoiser.checkpoints_many(
                denoise_items,
                config.denoise_steps,
                config.denoise_batch_size,
            )
        else:
            for image_path, output_dir in tqdm(
                denoise_items,
                desc="CIDER denoising",
                unit="image",
                dynamic_ncols=True,
            ):
                denoised[image_path] = denoiser.checkpoints(image_path, output_dir, config.denoise_steps)
    finally:
        denoiser.close()

    encoder = _LlavaCrossModalEncoder(config)
    text_cache: dict[str, Any] = {}

    def text_embedding(text: str):
        if text not in text_cache:
            text_cache[text] = encoder.text_embedding(text)
        return text_cache[text]

    def image_embeddings(paths: list[Path]):
        if hasattr(encoder, "image_embeddings"):
            return encoder.image_embeddings(paths, config.encoder_batch_size)
        return [encoder.image_embedding(path) for path in paths]

    effective_threshold = config.threshold
    paper_encoder_verified = bool(getattr(encoder, "paper_encoder_verified", False))
    encoder_verification = dict(getattr(encoder, "verification", {}))
    paper_configuration = (
        config.denoiser == "diffusion"
        and config.denoise_steps == CIDER_OFFICIAL_STEPS
        and config.encoder_mode == "paper_llava15"
        and paper_encoder_verified
        and config.dtype == "float16"
        and (
            abs(config.threshold - CIDER_OFFICIAL_THRESHOLD) < 1e-12
            or (
                bool(calibration_images)
                and abs(config.calibration_pass_rate - 0.95) < 1e-12
            )
        )
    )
    try:
        if calibration_images:
            calibration_deltas = []
            calibration_image_embeddings = {
                image_path: image_embeddings(denoised[image_path])
                for image_path in tqdm(
                    calibration_images,
                    desc="CIDER calibration image encoding",
                    unit="image",
                    dynamic_ncols=True,
                )
            }
            total = len(calibration_queries) * len(calibration_images)
            combinations = (
                (query, image_path)
                for query in calibration_queries
                for image_path in calibration_images
            )
            for query, image_path in tqdm(
                combinations,
                total=total,
                desc="CIDER threshold calibration",
                unit="pair",
                dynamic_ncols=True,
            ):
                text_vector = text_embedding(query)
                similarities = [
                    encoder.cosine(text_vector, image_vector)
                    for image_vector in calibration_image_embeddings[image_path]
                ]
                baseline = similarities[0]
                calibration_deltas.extend(value - baseline for value in similarities[1:])
            effective_threshold = calibrate_cider_threshold(
                calibration_deltas,
                pass_rate=config.calibration_pass_rate,
            )
            write_json(
                config.artifact_dir / "calibration.json",
                {
                    "threshold": effective_threshold,
                    "pass_rate": config.calibration_pass_rate,
                    "image_count": len(calibration_images),
                    "query_count": len(calibration_queries),
                    "delta_count": len(calibration_deltas),
                    "encoder_mode": config.encoder_mode,
                    "encoder_role": "fixed_auxiliary_detector",
                    "uses_victim_encoder": False,
                    "encoder_model": config.encoder_model,
                    "encoder_verification": encoder_verification,
                    "encoder_source": config.encoder_source,
                    "encoder_dtype": config.dtype,
                    "encoder_batch_size": config.encoder_batch_size,
                    "denoiser": config.denoiser,
                    "denoise_steps": list(config.denoise_steps),
                    "denoise_batch_size": config.denoise_batch_size,
                    "seed": config.seed,
                },
            )

        for index, sample in tqdm(pending, desc="CIDER cross-modal detection", unit="sample", dynamic_ncols=True):
            image_path = Path(str(sample["image_path"])).expanduser().resolve()
            prompt = _sample_prompt(sample)
            text_vector = text_embedding(prompt)
            checkpoint_paths = denoised[image_path]
            checkpoint_embeddings = image_embeddings(checkpoint_paths)
            similarities = [encoder.cosine(text_vector, image_vector) for image_vector in checkpoint_embeddings]
            detection = cider_detect(similarities, effective_threshold)
            selected_index = int(detection["selected_index"])
            row = {
                **sample,
                "cider_preprocessed": True,
                "cider_detected": bool(detection["detected"]),
                "cider_threshold": effective_threshold,
                "cider_minimum_delta": detection["minimum_delta"],
                "cider_cosine_similarities": similarities,
                "cider_similarity_deltas": detection["deltas"],
                "cider_lowest_checkpoint_index": detection["lowest_index"],
                "cider_selected_checkpoint_index": selected_index,
                "cider_selected_denoise_step": config.denoise_steps[selected_index],
                "cider_original_image_path": str(image_path),
                "cider_processed_image_path": str(checkpoint_paths[selected_index].resolve()),
                "cider_checkpoint_paths": [str(path.resolve()) for path in checkpoint_paths],
                "cider_detection_prompt": prompt,
                "cider_encoder_mode": config.encoder_mode,
                "cider_encoder_role": "fixed_auxiliary_detector",
                "cider_uses_victim_encoder": False,
                "cider_encoder_model": config.encoder_model,
                "cider_encoder_verification": encoder_verification,
                "cider_paper_encoder_verified": paper_encoder_verified,
                "cider_encoder_source": config.encoder_source,
                "cider_encoder_dtype": config.dtype,
                "cider_encoder_batch_size": config.encoder_batch_size,
                "cider_denoiser": config.denoiser,
                "cider_denoise_steps": list(config.denoise_steps),
                "cider_denoise_batch_size": config.denoise_batch_size,
                "cider_seed": config.seed,
                "cider_config_fingerprint": config_fingerprint,
                "cider_format_version": CIDER_FORMAT_VERSION,
                "cider_calibrated": bool(calibration_images),
                "cider_paper_configuration": paper_configuration,
            }
            append_jsonl_row(cache_path, row)
            prepared[index] = row
    finally:
        encoder.close()

    if any(row is None for row in prepared):
        raise RuntimeError("CIDER preprocessing did not produce one result per sample")
    return [row for row in prepared if row is not None]
