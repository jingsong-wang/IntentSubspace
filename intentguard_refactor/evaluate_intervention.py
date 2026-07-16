from __future__ import annotations

import argparse
from pathlib import Path

from intentguard.intervention_eval import evaluate_oracle_bypass
from intentguard.io import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate forced CSRL on every ground-truth risk sample."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-judge", type=Path, required=True)
    parser.add_argument("--post-judge", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate_oracle_bypass(
        read_jsonl(args.manifest),
        read_jsonl(args.base_judge),
        read_jsonl(args.post_judge),
    )
    write_json(args.out, result)
    print(f"Wrote oracle intervention evaluation to {args.out}")


if __name__ == "__main__":
    main()

