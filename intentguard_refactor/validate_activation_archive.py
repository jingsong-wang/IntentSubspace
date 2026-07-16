from __future__ import annotations

import argparse
import sys
from pathlib import Path

from intentguard.artifacts import activation_archive_errors, format_activation_archive_error
from intentguard.io import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a resumable activation NPZ cache.")
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--require-multimodal-anchor", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--backend")
    parser.add_argument("--quiet", action="store_true", help="Suppress the success message.")
    args = parser.parse_args()

    errors = activation_archive_errors(
        args.activations,
        expected_rows=read_jsonl(args.data),
        require_multimodal_anchor=args.require_multimodal_anchor,
        expected_model=args.model,
        expected_backend=args.backend,
    )
    if errors:
        print(format_activation_archive_error(args.activations, errors), file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"Activation archive is compatible: {args.activations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
