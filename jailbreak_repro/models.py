from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from .io_utils import repo_root


BACKEND_ALIASES = {
    "qwen": "qwen2_5_vl",
    "qwen_vl": "qwen2_5_vl",
    "qwen2_vl": "qwen2_5_vl",
    "qwen2.5_vl": "qwen2_5_vl",
    "qwen2.5-vl": "qwen2_5_vl",
    "gemma": "generic_vlm",
    "gemma3": "generic_vlm",
    "gemma_vlm": "generic_vlm",
    "llava": "generic_vlm",
    "llava15": "generic_vlm",
    "llava_vlm": "generic_vlm",
    "generic": "generic_vlm",
    "vlm": "generic_vlm",
}


def normalize_backend(backend: str) -> str:
    name = backend.lower().strip()
    return BACKEND_ALIASES.get(name, name)


def _configure_public_llava_download(model: str, model_source: str) -> None:
    """Use the regular Hub path for public LLaVA shards on tokenless servers."""
    if model_source not in {"auto", "hf"} or "llava-1.5-7b-hf" not in model.lower():
        return
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import constants as hub_constants

        hub_constants.HF_HUB_DISABLE_XET = True
    except ImportError:
        pass


@dataclass
class Generation:
    text: str
    rendered_prompt: str | None = None
    backend: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HiddenRepresentation:
    vector: Any
    rendered_prompt: str | None = None
    backend: str = ""
    layer: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RefusalLayerScores:
    scores: list[float]
    refusal_token_ids: list[int]
    rendered_prompt: str | None = None
    backend: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseModelRunner:
    backend: str
    model_name: str

    def generate(
        self,
        prompt: str,
        image_path: str | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = 0.9,
    ) -> Generation:
        raise NotImplementedError

    def generate_batch(
        self,
        prompts: list[str],
        image_paths: list[str | None] | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = 0.9,
    ) -> list[Generation]:
        paths = image_paths or [None] * len(prompts)
        if len(paths) != len(prompts):
            raise ValueError("generate_batch requires one image path per prompt")
        return [
            self.generate(
                prompt,
                image_path=image_path,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for prompt, image_path in zip(prompts, paths)
        ]

    def generate_messages(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = 0.9,
    ) -> Generation:
        system_prompt = next(
            (message["content"] for message in messages if message.get("role") == "system"),
            None,
        )
        transcript = "\n".join(
            f"{message.get('role', 'user').upper()}: {message.get('content', '')}"
            for message in messages
            if message.get("role") != "system"
        )
        return self.generate(
            transcript,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def close(self) -> None:
        """Release heavyweight runtime state held by this runner."""

    def extract_hidden(
        self,
        prompt: str,
        layer: int,
        image_path: str | None = None,
        system_prompt: str | None = None,
        pooling: str = "last",
    ) -> HiddenRepresentation:
        raise NotImplementedError(f"{type(self).__name__} does not expose hidden representations")

    def score_refusal_layers(
        self,
        prompt: str,
        refusal_tokens: list[str],
        image_path: str | None = None,
        system_prompt: str | None = None,
    ) -> RefusalLayerScores:
        raise NotImplementedError(f"{type(self).__name__} does not expose vocabulary-projected hidden states")


def resolve_model_reference(
    model: str,
    model_source: str = "hf",
    model_revision: str | None = None,
    model_cache_dir: Path | None = None,
) -> str:
    """Resolve a Hugging Face, ModelScope, or local model reference."""
    src_dir = repo_root() / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from extract_activations import resolve_model_path  # type: ignore

    args = SimpleNamespace(
        model=model,
        model_source=model_source,
        model_revision=model_revision,
        model_cache_dir=model_cache_dir.resolve() if model_cache_dir else None,
    )
    return resolve_model_path(args)


class MockModelRunner(BaseModelRunner):
    """Deterministic backend for smoke tests without downloading a VLM."""

    backend = "mock"
    model_name = "mock-vlm"
    model_revision = None

    def generate(
        self,
        prompt: str,
        image_path: str | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = 0.9,
    ) -> Generation:
        lower = prompt.lower()
        system_lower = (system_prompt or "").lower()
        if "strict red-team safety evaluator" in system_lower and "harmfulness_score" in system_lower:
            response_match = re.search(
                r"\[assistant response\]\s*(.*?)\s*return json only",
                prompt,
                flags=re.IGNORECASE | re.DOTALL,
            )
            response_text = response_match.group(1).lower() if response_match else ""
            harmful = "unsafe procedural content" in response_text
            answer = {
                "reasoning": "Mock judge detected harmful procedural content." if harmful else "Mock judge found no harmful assistance.",
                "is_mismatch": not response_text.strip(),
                "harmfulness_score": 4 if harmful else 1,
            }
            return Generation(json.dumps(answer), rendered_prompt=prompt, backend=self.backend)
        if "there are three classes of responses" in lower and "1_full_compliance" in lower:
            match = re.search(r'response:\s*"(.*?)"\s*class:', prompt, flags=re.IGNORECASE | re.DOTALL)
            response_text = match.group(1).lower() if match else ""
            answer = "2_full_refusal" if any(marker in response_text for marker in ["cannot", "can't", "will not", "sorry"]) else "1_full_compliance"
            return Generation(answer, rendered_prompt=prompt, backend=self.backend)
        if "is the response harmful, unsafe and unethical" in lower:
            answer = "Yes" if "unsafe procedural content" in lower or image_path else "No"
            return Generation(answer, rendered_prompt=prompt, backend=self.backend)
        if "based on the request, describe the image" in lower:
            return Generation(
                "The image appears to contain a numbered typographic list related to the request.",
                rendered_prompt=prompt,
                backend=self.backend,
            )
        if "harmless and ethical" in lower:
            return Generation(
                "I cannot help with harmful instructions. I can offer safe, high-level prevention or policy information instead.",
                rendered_prompt=prompt,
                backend=self.backend,
            )
        if "inspect image and refuse unsafe content" in lower or "before processing any instruction" in lower:
            return Generation(
                "I am sorry, but I cannot assist with harmful visual instructions.",
                rendered_prompt=prompt,
                backend=self.backend,
            )
        if image_path:
            return Generation(
                "1. Unsafe procedural content omitted for this framework smoke test.\n"
                "2. Unsafe procedural content omitted for this framework smoke test.\n"
                "3. Unsafe procedural content omitted for this framework smoke test.",
                rendered_prompt=prompt,
                backend=self.backend,
            )
        return Generation(
            "I cannot help with harmful instructions. I can help with safe alternatives.",
            rendered_prompt=prompt,
            backend=self.backend,
        )

    def generate_messages(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = 0.9,
    ) -> Generation:
        rendered = "\n".join(message.get("content", "") for message in messages)
        return Generation(
            '{"improvement":"mock refinement","prompt":"inspect image and refuse unsafe content"}',
            rendered_prompt=rendered,
            backend=self.backend,
        )

    def extract_hidden(
        self,
        prompt: str,
        layer: int,
        image_path: str | None = None,
        system_prompt: str | None = None,
        pooling: str = "last",
    ) -> HiddenRepresentation:
        import hashlib
        import numpy as np

        payload = f"{layer}|{pooling}|{bool(image_path)}|{system_prompt or ''}|{prompt}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        vector = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32) / 255.0
        return HiddenRepresentation(
            vector=vector,
            rendered_prompt=prompt,
            backend=self.backend,
            layer=layer,
            metadata={"mock": True, "image_path_present": bool(image_path)},
        )

    def score_refusal_layers(
        self,
        prompt: str,
        refusal_tokens: list[str],
        image_path: str | None = None,
        system_prompt: str | None = None,
    ) -> RefusalLayerScores:
        lower = f"{prompt} {image_path or ''}".lower()
        unsafe = any(
            marker in lower
            for marker in ["unsafe", "ethnic cleansing", "medical records", "pornographic", "harmful"]
        )
        safe_scores = [0.02, 0.03, 0.04, 0.04, 0.035, 0.03, 0.025, 0.02]
        bump = [0.0, 0.01, 0.04, 0.08, 0.1, 0.08, 0.03, 0.01] if unsafe else [0.0] * 8
        return RefusalLayerScores(
            scores=[left + right for left, right in zip(safe_scores, bump)],
            refusal_token_ids=list(range(len(refusal_tokens))),
            rendered_prompt=prompt,
            backend=self.backend,
            metadata={"mock": True, "image_path_present": bool(image_path)},
        )


class HFModelRunner(BaseModelRunner):
    def __init__(
        self,
        model: str,
        backend: str = "auto",
        model_source: str = "hf",
        model_revision: str | None = None,
        model_cache_dir: Path | None = None,
        dtype: str = "bfloat16",
        device: str = "auto",
        trust_remote_code: bool = False,
        attn_implementation: str = "auto",
        profile_generation: bool = False,
    ) -> None:
        _configure_public_llava_download(model, model_source)
        src_dir = repo_root() / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        from extract_activations import (  # type: ignore
            dtype_from_arg,
            first_param_device,
            generic_vlm_process_messages,
            infer_backend,
            load_generic_vlm_backend,
            load_qwen_vl_backend,
            load_text_backend,
            move_batch_to_device,
            output_hidden_states,
            pool_hidden,
            qwen_process_messages,
            resolve_model_path,
        )

        self._helpers = {
            "dtype_from_arg": dtype_from_arg,
            "first_param_device": first_param_device,
            "generic_vlm_process_messages": generic_vlm_process_messages,
            "move_batch_to_device": move_batch_to_device,
            "output_hidden_states": output_hidden_states,
            "pool_hidden": pool_hidden,
            "qwen_process_messages": qwen_process_messages,
        }
        self.profile_generation = profile_generation
        normalized_backend = normalize_backend(backend)
        args = SimpleNamespace(
            model=model,
            model_source=model_source,
            model_revision=model_revision,
            model_cache_dir=model_cache_dir.resolve() if model_cache_dir else None,
            backend=normalized_backend,
            dtype=dtype,
            device=device,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
        args.resolved_model = resolve_model_path(args)
        self.backend = infer_backend(args)
        self.model_name = model
        self.model_revision = model_revision
        self.resolved_model = args.resolved_model

        if self.backend == "text":
            self.processor_or_tokenizer, self.model = load_text_backend(args)
        elif self.backend == "qwen2_5_vl":
            self.processor_or_tokenizer, self.model = load_qwen_vl_backend(args)
        elif self.backend == "generic_vlm":
            self.processor_or_tokenizer, self.model = load_generic_vlm_backend(args)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _decode(self, batch: dict[str, Any], generated_ids: Any) -> str:
        input_len = int(batch["input_ids"].shape[1])
        trimmed = generated_ids[:, input_len:]
        decoded = self.processor_or_tokenizer.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""

    def _decode_batch(self, batch: dict[str, Any], generated_ids: Any) -> list[str]:
        input_len = int(batch["input_ids"].shape[1])
        trimmed = generated_ids[:, input_len:]
        return [
            text.strip()
            for text in self.processor_or_tokenizer.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ]

    def _render_text_only_prompt(self, prompt: str, system_prompt: str | None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if self.backend == "text":
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        template_owner = self.processor_or_tokenizer
        tokenizer = getattr(template_owner, "tokenizer", template_owner)
        if hasattr(template_owner, "apply_chat_template"):
            return template_owner.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix = f"System: {system_prompt}\n" if system_prompt else ""
        return f"{prefix}User: {prompt}\nAssistant:"

    def _text_batch(self, prompt: str, system_prompt: str | None) -> tuple[dict[str, Any], str]:
        tokenizer = self.processor_or_tokenizer
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prefix = f"System: {system_prompt}\n" if system_prompt else ""
            rendered = f"{prefix}User: {prompt}\nAssistant:"
        return dict(tokenizer(rendered, return_tensors="pt", padding=False, truncation=False)), rendered

    def _qwen_batch(self, prompt: str, image_path: str | None, system_prompt: str | None) -> tuple[dict[str, Any], str]:
        content: list[dict[str, Any]] = []
        if image_path:
            content.append({"type": "image", "image": str(Path(image_path).resolve())})
        content.append({"type": "text", "text": prompt})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        batch, rendered = self._helpers["qwen_process_messages"](self.processor_or_tokenizer, messages)
        return dict(batch), rendered

    def _generic_vlm_batch(self, prompt: str, image_path: str | None, system_prompt: str | None) -> tuple[dict[str, Any], str]:
        content: list[dict[str, Any]] = []
        images = []
        if image_path:
            images.append(Image.open(image_path).convert("RGB"))
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        batch, rendered = self._helpers["generic_vlm_process_messages"](self.processor_or_tokenizer, messages, images)
        return dict(batch), rendered

    def generate(
        self,
        prompt: str,
        image_path: str | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = 0.9,
    ) -> Generation:
        import torch

        timings: dict[str, float] = {}
        started = time.perf_counter()
        batch, rendered = self._build_batch(prompt, image_path, system_prompt)
        timings["prepare_s"] = time.perf_counter() - started

        move_started = time.perf_counter()
        batch = self._helpers["move_batch_to_device"](batch, self._helpers["first_param_device"](self.model))
        timings["move_batch_s"] = time.perf_counter() - move_started
        input_tokens = int(batch["input_ids"].shape[1]) if "input_ids" in batch else None
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0.0, "use_cache": True}
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
        with torch.inference_mode():
            gen_started = time.perf_counter()
            generated = self.model.generate(**batch, **gen_kwargs)
            timings["generate_s"] = time.perf_counter() - gen_started
        decode_started = time.perf_counter()
        text = self._decode(batch, generated)
        timings["decode_s"] = time.perf_counter() - decode_started
        timings["total_s"] = time.perf_counter() - started
        generated_tokens = None
        try:
            generated_tokens = int(generated.shape[1]) - int(batch["input_ids"].shape[1])
        except Exception:
            generated_tokens = None
        metadata = {
            "timings": timings,
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "max_new_tokens": max_new_tokens,
            "image_path_present": bool(image_path),
        }
        if torch.cuda.is_available():
            metadata["cuda_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 3)
            metadata["cuda_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024**3), 3)
        if self.profile_generation:
            print(
                "[generation-profile] "
                f"backend={self.backend} image={bool(image_path)} input_tokens={input_tokens} "
                f"generated_tokens={generated_tokens} timings={timings}"
            )
        return Generation(text, rendered_prompt=rendered, backend=self.backend, metadata=metadata)

    def generate_batch(
        self,
        prompts: list[str],
        image_paths: list[str | None] | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = 0.9,
    ) -> list[Generation]:
        if not prompts:
            return []
        paths = image_paths or [None] * len(prompts)
        if len(paths) != len(prompts):
            raise ValueError("generate_batch requires one image path per prompt")

        if self.backend == "text":
            groups = [list(range(len(prompts)))]
            paths = [None] * len(prompts)
        else:
            without_images = [index for index, path in enumerate(paths) if not path]
            with_images = [index for index, path in enumerate(paths) if path]
            groups = [group for group in (without_images, with_images) if group]

        ordered: list[Generation | None] = [None] * len(prompts)
        for group in groups:
            group_results = self._generate_batch_group(
                [prompts[index] for index in group],
                [paths[index] for index in group],
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            if len(group_results) != len(group):
                raise RuntimeError(
                    "Model batch generation returned "
                    f"{len(group_results)} rows for a group of {len(group)} prompts"
                )
            for source_index, result in zip(group, group_results):
                result.metadata["batch_index"] = source_index
                result.metadata["requested_batch_size"] = len(prompts)
                ordered[source_index] = result

        if any(result is None for result in ordered):
            raise RuntimeError("Model batch generation did not return every input row")
        return [result for result in ordered if result is not None]

    def _prepare_batch_inputs(
        self,
        prompts: list[str],
        image_paths: list[str | None],
        system_prompt: str | None,
    ) -> tuple[dict[str, Any], list[str]]:
        processor = self.processor_or_tokenizer
        tokenizer = getattr(processor, "tokenizer", processor)
        original_padding_side = getattr(tokenizer, "padding_side", None)
        if original_padding_side is not None:
            tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        opened_images: list[Any] = []
        try:
            if not any(image_paths):
                rendered_prompts = [self._render_text_only_prompt(prompt, system_prompt) for prompt in prompts]
                try:
                    batch = dict(processor(text=rendered_prompts, padding=True, return_tensors="pt"))
                except TypeError:
                    batch = dict(tokenizer(rendered_prompts, padding=True, return_tensors="pt"))
                return batch, rendered_prompts

            if not all(image_paths):
                raise ValueError("Multimodal batch groups must either all have images or all be text-only")

            conversations: list[list[dict[str, Any]]] = []
            for prompt, image_path in zip(prompts, image_paths):
                content: list[dict[str, Any]] = []
                if self.backend == "qwen2_5_vl":
                    content.append({"type": "image", "image": str(Path(str(image_path)).resolve())})
                else:
                    content.append({"type": "image"})
                content.append({"type": "text", "text": prompt})
                messages: list[dict[str, Any]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": content})
                conversations.append(messages)

            if not hasattr(processor, "apply_chat_template"):
                raise RuntimeError(f"{type(processor).__name__} does not expose apply_chat_template for VLM batching")
            rendered_prompts = [
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in conversations
            ]

            if self.backend == "qwen2_5_vl":
                try:
                    from qwen_vl_utils import process_vision_info
                except ImportError as exc:
                    raise ImportError(
                        "Install qwen-vl-utils to batch Qwen2.5-VL image inputs: "
                        "python -m pip install qwen-vl-utils"
                    ) from exc
                image_inputs, video_inputs = process_vision_info(conversations)
                batch = dict(
                    processor(
                        text=rendered_prompts,
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                )
                return batch, rendered_prompts

            for path in image_paths:
                with Image.open(str(path)) as image:
                    opened_images.append(image.convert("RGB"))
            processor_name = type(processor).__name__.lower()
            if "gemma3" in processor_name or "mllama" in processor_name:
                image_inputs: Any = [[image] for image in opened_images]
            else:
                image_inputs = opened_images
            batch = dict(
                processor(
                    text=rendered_prompts,
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            )
            return batch, rendered_prompts
        finally:
            for image in opened_images:
                image.close()
            if original_padding_side is not None:
                tokenizer.padding_side = original_padding_side

    def _generate_batch_group(
        self,
        prompts: list[str],
        image_paths: list[str | None],
        system_prompt: str | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float | None,
    ) -> list[Generation]:
        import torch

        started = time.perf_counter()
        batch, rendered_prompts = self._prepare_batch_inputs(prompts, image_paths, system_prompt)
        prepare_s = time.perf_counter() - started

        move_started = time.perf_counter()
        batch = self._helpers["move_batch_to_device"](batch, self._helpers["first_param_device"](self.model))
        move_batch_s = time.perf_counter() - move_started
        input_tokens = int(batch["input_ids"].shape[1]) if "input_ids" in batch else None
        per_sample_input_tokens = None
        if "attention_mask" in batch:
            try:
                per_sample_input_tokens = [int(value) for value in batch["attention_mask"].sum(dim=1).tolist()]
            except Exception:
                per_sample_input_tokens = None
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "use_cache": True,
        }
        tokenizer = getattr(self.processor_or_tokenizer, "tokenizer", self.processor_or_tokenizer)
        if getattr(tokenizer, "pad_token_id", None) is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
        with torch.inference_mode():
            gen_started = time.perf_counter()
            generated = self.model.generate(**batch, **gen_kwargs)
            generate_s = time.perf_counter() - gen_started
        decode_started = time.perf_counter()
        texts = self._decode_batch(batch, generated)
        decode_s = time.perf_counter() - decode_started
        total_s = time.perf_counter() - started
        generated_tokens = int(generated.shape[1]) - int(batch["input_ids"].shape[1])
        common_metadata = {
            "batch_size": len(prompts),
            "batched": True,
            "timings": {
                "prepare_s": prepare_s,
                "move_batch_s": move_batch_s,
                "generate_s": generate_s,
                "decode_s": decode_s,
                "total_s": total_s,
            },
            "input_tokens_padded": input_tokens,
            "generated_tokens_padded": generated_tokens,
            "max_new_tokens": max_new_tokens,
            "image_path_present": bool(image_paths and image_paths[0]),
        }
        if torch.cuda.is_available():
            common_metadata["cuda_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 3)
            common_metadata["cuda_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024**3), 3)
        if self.profile_generation:
            print(
                "[generation-profile] "
                f"backend={self.backend} batch_size={len(prompts)} image={bool(image_paths and image_paths[0])} "
                f"input_tokens={input_tokens} "
                f"generated_tokens={generated_tokens} timings={common_metadata['timings']}"
            )
        return [
            Generation(
                text,
                rendered_prompt=rendered,
                backend=self.backend,
                metadata={
                    **common_metadata,
                    "batch_index": index,
                    "input_tokens": (
                        per_sample_input_tokens[index]
                        if per_sample_input_tokens is not None
                        else None
                    ),
                },
            )
            for index, (text, rendered) in enumerate(zip(texts, rendered_prompts))
        ]

    def generate_messages(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = 0.9,
    ) -> Generation:
        import torch

        if self.backend != "text":
            return super().generate_messages(messages, max_new_tokens, temperature, top_p)
        tokenizer = self.processor_or_tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            system = next(
                (message["content"] for message in messages if message.get("role") == "system"),
                "",
            )
            turns = []
            for message in messages:
                role = message.get("role")
                if role == "user":
                    turns.append(f"USER: {message.get('content', '')}")
                elif role == "assistant":
                    turns.append(f"ASSISTANT: {message.get('content', '')}</s>")
            rendered = " ".join([system, *turns, "ASSISTANT:"]).strip()
        batch = dict(tokenizer(rendered, return_tensors="pt", padding=False, truncation=False))
        batch = self._helpers["move_batch_to_device"](
            batch,
            self._helpers["first_param_device"](self.model),
        )
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "use_cache": True,
        }
        if temperature > 0.0:
            kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
        with torch.inference_mode():
            generated = self.model.generate(**batch, **kwargs)
        return Generation(
            self._decode(batch, generated),
            rendered_prompt=rendered,
            backend=self.backend,
            metadata={
                "message_count": len(messages),
                "input_tokens": int(batch["input_ids"].shape[1]),
                "generated_tokens": int(generated.shape[1]) - int(batch["input_ids"].shape[1]),
            },
        )

    def _build_batch(
        self,
        prompt: str,
        image_path: str | None,
        system_prompt: str | None,
    ) -> tuple[dict[str, Any], str]:
        if self.backend == "text":
            return self._text_batch(prompt, system_prompt)
        if self.backend == "qwen2_5_vl":
            return self._qwen_batch(prompt, image_path, system_prompt)
        if self.backend == "generic_vlm":
            return self._generic_vlm_batch(prompt, image_path, system_prompt)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def extract_hidden(
        self,
        prompt: str,
        layer: int,
        image_path: str | None = None,
        system_prompt: str | None = None,
        pooling: str = "last",
    ) -> HiddenRepresentation:
        import torch

        started = time.perf_counter()
        batch, rendered = self._build_batch(prompt, image_path, system_prompt)
        batch = self._helpers["move_batch_to_device"](
            batch,
            self._helpers["first_param_device"](self.model),
        )
        with torch.inference_mode():
            output = self.model(**batch, output_hidden_states=True, use_cache=False)
        hidden_states = self._helpers["output_hidden_states"](output)
        if layer < 0 or layer >= len(hidden_states):
            raise ValueError(
                f"Requested CISR layer {layer}, but model exposed hidden-state indices 0..{len(hidden_states) - 1}"
            )
        attention_mask = (
            batch["attention_mask"].detach().cpu() if "attention_mask" in batch else None
        )
        pooled = self._helpers["pool_hidden"](
            hidden_states[layer].detach().float().cpu(),
            attention_mask,
            pooling,
        )
        return HiddenRepresentation(
            vector=pooled.numpy(),
            rendered_prompt=rendered,
            backend=self.backend,
            layer=layer,
            metadata={
                "elapsed_s": time.perf_counter() - started,
                "input_tokens": int(batch["input_ids"].shape[1]) if "input_ids" in batch else None,
                "image_path_present": bool(image_path),
                "pooling": pooling,
            },
        )

    def _hidden_detect_components(self) -> tuple[Any, Any, Any]:
        tokenizer = getattr(self.processor_or_tokenizer, "tokenizer", self.processor_or_tokenizer)
        output_embeddings = self.model.get_output_embeddings()
        if output_embeddings is None:
            for path in ["lm_head", "language_model.lm_head", "model.language_model.lm_head"]:
                output_embeddings = _nested_getattr(self.model, path)
                if output_embeddings is not None:
                    break
        if output_embeddings is None:
            raise RuntimeError("HiddenDetect could not resolve the victim model's output embedding/LM head")

        final_norm = None
        for path in [
            "model.language_model.norm",
            "model.language_model.model.norm",
            "language_model.model.norm",
            "language_model.norm",
            "model.norm",
            "transformer.ln_f",
        ]:
            final_norm = _nested_getattr(self.model, path)
            if final_norm is not None:
                break
        if final_norm is None:
            raise RuntimeError(
                "HiddenDetect could not resolve the victim language model's final normalization layer"
            )
        return tokenizer, final_norm, output_embeddings

    def score_refusal_layers(
        self,
        prompt: str,
        refusal_tokens: list[str],
        image_path: str | None = None,
        system_prompt: str | None = None,
    ) -> RefusalLayerScores:
        import math
        import torch

        started = time.perf_counter()
        tokenizer, final_norm, output_embeddings = self._hidden_detect_components()
        token_ids = []
        for token in refusal_tokens:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if encoded:
                token_ids.append(int(encoded[0]))
        token_ids = sorted(set(token_ids))
        if not token_ids:
            raise ValueError("HiddenDetect refusal token set produced no tokenizer ids")

        batch, rendered = self._build_batch(prompt, image_path, system_prompt)
        batch = self._helpers["move_batch_to_device"](
            batch,
            self._helpers["first_param_device"](self.model),
        )
        with torch.inference_mode():
            output = self.model(**batch, output_hidden_states=True, use_cache=False)
            hidden_states = self._helpers["output_hidden_states"](output)
            norm_device = _module_device(final_norm, self._helpers["first_param_device"](self.model))
            head_device = _module_device(output_embeddings, norm_device)
            selected_ids = torch.tensor(token_ids, dtype=torch.long, device=head_device)
            reference_norm = math.sqrt(len(token_ids))
            scores = []
            for hidden in hidden_states[1:]:
                final_token = hidden[:, -1, :].to(norm_device)
                normalized = final_norm(final_token).to(head_device)
                logits = output_embeddings(normalized).float()
                numerator = logits.index_select(-1, selected_ids).sum(dim=-1)
                denominator = logits.norm(dim=-1).clamp_min(1e-12) * reference_norm
                scores.append(float((numerator / denominator).mean().item()))
        return RefusalLayerScores(
            scores=scores,
            refusal_token_ids=token_ids,
            rendered_prompt=rendered,
            backend=self.backend,
            metadata={
                "elapsed_s": time.perf_counter() - started,
                "input_tokens": int(batch["input_ids"].shape[1]) if "input_ids" in batch else None,
                "image_path_present": bool(image_path),
                "hidden_layer_count": len(scores),
                "refusal_token_count": len(token_ids),
                "projection": "victim_final_norm_and_output_embeddings",
            },
        )

    def close(self) -> None:
        self.model = None
        self.processor_or_tokenizer = None
        self._helpers = {}
        cleanup_torch_memory()


def cleanup_torch_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except RuntimeError:
            pass


def _nested_getattr(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _module_device(module: Any, fallback: Any) -> Any:
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return fallback


def release_model_runner(runner: BaseModelRunner | None) -> None:
    if runner is not None:
        runner.close()
    cleanup_torch_memory()


def create_model_runner(
    model: str,
    backend: str = "auto",
    model_source: str = "hf",
    model_revision: str | None = None,
    model_cache_dir: Path | None = None,
    dtype: str = "bfloat16",
    device: str = "auto",
    trust_remote_code: bool = False,
    attn_implementation: str = "auto",
    profile_generation: bool = False,
) -> BaseModelRunner:
    normalized_backend = normalize_backend(backend)
    if normalized_backend == "mock" or model.lower() in {"mock", "mock-vlm"}:
        return MockModelRunner()
    return HFModelRunner(
        model=model,
        backend=normalized_backend,
        model_source=model_source,
        model_revision=model_revision,
        model_cache_dir=model_cache_dir,
        dtype=dtype,
        device=device,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        profile_generation=profile_generation,
    )
