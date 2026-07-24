from __future__ import annotations

import argparse
from pathlib import Path

from counterfactual_risk_field.cnrf.generation import ingest_generation_responses
from counterfactual_risk_field.cnrf.io import read_jsonl, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse LLM responses into human-review candidates.")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidates, failures = ingest_generation_responses(
        list(read_jsonl(args.requests)), list(read_jsonl(args.responses))
    )
    write_jsonl(args.out, candidates)
    write_json(
        args.out.with_suffix(".audit.json"),
        {
            "format_version": "cnrf_counterfactual_candidate_audit_v1",
            "candidates": len(candidates),
            "failures": failures,
            "next_gate": "A human reviewer must set audit_status=approved only after all five review fields pass.",
        },
    )
    print(f"Wrote {len(candidates)} candidates; failures={len(failures)}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
