from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Any

from counterfactual_risk_field.cnrf.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CNRF counterfactual candidates by loading a local model directly."
    )
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--backend", default="generic_vlm")
    parser.add_argument("--model-source", choices=["hf", "modelscope", "auto"], default="modelscope")
    parser.add_argument("--model-revision")
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sampling-seed", type=int, default=20260721)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile-generation", action="store_true")
    return parser.parse_args()


def _message_parts(request: dict[str, Any]) -> tuple[str, str]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Request {request.get('request_id')} has no messages list")
    systems = [str(row.get("content") or "") for row in messages if row.get("role") == "system"]
    users = [str(row.get("content") or "") for row in messages if row.get("role") == "user"]
    if len(systems) != 1 or len(users) != 1 or not systems[0] or not users[0]:
        raise ValueError(
            f"Request {request.get('request_id')} must contain exactly one non-empty system and user message"
        )
    return systems[0], users[0]


def _seed_sampling(seed: int, request_ids: list[str]) -> int:
    digest = hashlib.sha256("\n".join(request_ids).encode("utf-8")).digest()
    batch_seed = int(seed) ^ int.from_bytes(digest[:4], "big")
    random.seed(batch_seed)
    try:
        import numpy as np

        np.random.seed(batch_seed % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(batch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(batch_seed)
    except ImportError:
        pass
    return batch_seed


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    requests = list(read_jsonl(args.requests))
    if args.max_requests and args.max_requests > 0:
        requests = requests[: args.max_requests]
    existing = list(read_jsonl(args.out)) if args.resume and args.out.exists() else []
    completed_ids = {str(row.get("request_id") or "") for row in existing}
    pending = [row for row in requests if str(row["request_id"]) not in completed_ids]
    responses = list(existing)
    failures: list[dict[str, Any]] = []
    fatal_error: dict[str, str] | None = None

    from jailbreak_repro.models import create_model_runner, release_model_runner

    runner = None
    try:
        runner = create_model_runner(
            model=args.model,
            backend=args.backend,
            model_source=args.model_source,
            model_revision=args.model_revision,
            model_cache_dir=args.model_cache_dir,
            dtype=args.dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
            profile_generation=args.profile_generation,
        )
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            batch_seed = _seed_sampling(
                args.sampling_seed, [str(row["request_id"]) for row in batch]
            )
            parts = [_message_parts(row) for row in batch]
            systems = {system for system, _ in parts}
            if len(systems) != 1:
                raise ValueError("Every request in a local-generation batch must share one system prompt")
            try:
                generated = runner.generate_batch(
                    [user for _, user in parts],
                    system_prompt=next(iter(systems)),
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            except Exception as exc:
                failures.extend(
                    {
                        "request_id": str(row["request_id"]),
                        "error": f"{type(exc).__name__}: {exc}",
                        "batch_offset": offset,
                    }
                    for row in batch
                )
                break
            for request, result in zip(batch, generated):
                responses.append(
                    {
                        "request_id": str(request["request_id"]),
                        "content": result.text,
                        "model": args.model,
                        "backend": result.backend,
                        "generation_metadata": {
                            **result.metadata,
                            "sampling_seed": batch_seed,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                        },
                    }
                )
            write_jsonl(args.out, responses)
            print(f"completed={len(responses)}/{len(requests)}", flush=True)
    except Exception as exc:
        fatal_error = {
            "stage": "model_load_or_runtime",
            "error": f"{type(exc).__name__}: {exc}",
        }
        failures.append(fatal_error)
    finally:
        release_model_runner(runner)

    write_jsonl(args.out, responses)
    write_json(
        args.out.with_suffix(".run.json"),
        {
            "format_version": "cnrf_local_generation_run_v1",
            "model": args.model,
            "backend": args.backend,
            "model_source": args.model_source,
            "model_cache_dir": str(args.model_cache_dir) if args.model_cache_dir else None,
            "requested": len(requests),
            "completed": len(responses),
            "failures": failures,
            "fatal_error": fatal_error,
            "resume": bool(args.resume),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "sampling_seed": args.sampling_seed,
            "external_data_transfer": False,
            "automatic_retry": False,
        },
    )
    if fatal_error:
        print(f"Local generation failed: {fatal_error['error']}", flush=True)
    print(f"Completed {len(responses)}/{len(requests)} local generation requests")
    return 0 if not failures and len(responses) == len(requests) else 2


if __name__ == "__main__":
    raise SystemExit(main())
