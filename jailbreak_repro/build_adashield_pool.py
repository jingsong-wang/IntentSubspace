from __future__ import annotations

import argparse
from pathlib import Path

from .adashield import build_pool_from_official_tables, default_adashield_source_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert AdaShield official auto-refinement final_table.csv outputs to a standard prompt pool."
    )
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--victim-model", required=True, help="Exact victim model id/path used during AdaShield training.")
    parser.add_argument("--victim-revision")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=default_adashield_source_dir())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_pool_from_official_tables(
        table_dir=args.table_dir,
        output_path=args.out,
        victim_model=args.victim_model,
        source_dir=args.source_dir,
        victim_revision=args.victim_revision,
    )
    print(f"Wrote {payload['entry_count']} AdaShield-A prompt-pool entries to {args.out.resolve()}")


if __name__ == "__main__":
    main()
