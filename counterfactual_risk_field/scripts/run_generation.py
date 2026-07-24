from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

from counterfactual_risk_field.cnrf.io import read_jsonl, write_json, write_jsonl


def _is_local(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run counterfactual requests on an OpenAI-compatible chat endpoint.")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", required=True, help="For example http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument(
        "--allow-external-data-transfer",
        action="store_true",
        help="Required for non-loopback endpoints because harmful research prompts leave this machine.",
    )
    args = parser.parse_args()
    if not _is_local(args.base_url) and not args.allow_external_data_transfer:
        raise PermissionError(
            "Refusing to send the research corpus to a remote provider without "
            "--allow-external-data-transfer. Confirm provider policy and data authorization first."
        )
    key = os.environ.get(args.api_key_env, "")
    if not key and not _is_local(args.base_url):
        raise EnvironmentError(f"Environment variable {args.api_key_env!r} is empty")
    requests = list(read_jsonl(args.requests))
    if args.max_requests and args.max_requests > 0:
        requests = requests[: args.max_requests]
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    responses = []
    failures = []
    for row in requests:
        payload = {
            "model": args.model,
            "messages": row["messages"],
            "temperature": args.temperature,
            "response_format": row.get("response_format", {"type": "json_object"}),
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {key}"} if key else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
            responses.append({"request_id": row["request_id"], **value})
        except Exception as exc:
            failures.append(
                {
                    "request_id": row["request_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    write_jsonl(args.out, responses)
    write_json(
        args.out.with_suffix(".run.json"),
        {
            "format_version": "cnrf_generation_run_v1",
            "endpoint": endpoint,
            "model": args.model,
            "requested": len(requests),
            "completed": len(responses),
            "failures": failures,
            "automatic_retry": False,
        },
    )
    print(f"Completed {len(responses)}/{len(requests)} generation requests")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
