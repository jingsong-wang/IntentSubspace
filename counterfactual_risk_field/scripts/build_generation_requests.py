from __future__ import annotations

import argparse
import json
from pathlib import Path

from counterfactual_risk_field.cnrf.generation import build_generation_requests
from counterfactual_risk_field.cnrf.io import read_jsonl, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Build LLM requests for controlled counterfactual pairs.")
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("counterfactual_risk_field/configs/protocol_v1.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    generation = config["counterfactual_generation"]
    rows = list(read_jsonl(args.seeds))
    requests = build_generation_requests(
        rows,
        seed=int(config["seed"]),
        split_fractions=config["pair_split_fractions"],
        include_native_image=bool(generation["include_native_image"]),
        counterfactual_axes=generation.get("counterfactual_axes"),
        variants_per_carrier=int(generation.get("variants_per_carrier", 1)),
        skip_image_dependent_non_native=bool(
            generation.get("skip_image_dependent_non_native", False)
        ),
    )
    write_jsonl(args.out, requests)
    write_json(
        args.out.with_suffix(".manifest.json"),
        {
            "format_version": "cnrf_generation_requests_v1",
            "requests": len(requests),
            "seed_manifest": str(args.seeds),
            "config": str(args.config),
            "counterfactual_axes": generation.get("counterfactual_axes"),
            "variants_per_carrier": int(generation.get("variants_per_carrier", 1)),
            "recommended_temperature": generation.get("recommended_temperature"),
            "recommended_top_p": generation.get("recommended_top_p"),
            "formal_warning": "Generated candidates must pass human review before fitting.",
        },
    )
    print(f"Wrote {len(requests)} generation requests to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
