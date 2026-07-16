from __future__ import annotations

import argparse
from pathlib import Path

from intentguard.intervention_data import build_manifest, summarize_manifest
from intentguard.io import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join detector and response-judge outputs into CSRL train/eval roles."
    )
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()

    rows = build_manifest(read_jsonl(args.detections), read_jsonl(args.judge))
    write_jsonl(args.out, rows)
    summary_path = args.summary_out or args.out.with_name(args.out.stem + "_summary.json")
    write_json(summary_path, summarize_manifest(rows))
    print(f"Wrote {len(rows)} intervention manifest rows to {args.out}")
    print(f"Wrote intervention manifest summary to {summary_path}")


if __name__ == "__main__":
    main()

