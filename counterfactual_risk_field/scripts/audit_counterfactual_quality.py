from __future__ import annotations

import argparse
from pathlib import Path

from counterfactual_risk_field.cnrf.io import read_jsonl, write_json, write_jsonl
from counterfactual_risk_field.cnrf.quality import audit_candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit and optionally filter generated CNRF counterfactual candidates."
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--accepted-out", type=Path, required=True)
    parser.add_argument("--rejected-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-topic-coverage", type=float, default=0.15)
    parser.add_argument("--min-length-ratio", type=float, default=0.4)
    parser.add_argument("--max-length-ratio", type=float, default=2.5)
    parser.add_argument("--allow-image-semantic-mismatch", action="store_true")
    parser.add_argument("--allow-legacy-axis", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true")
    args = parser.parse_args()

    accepted, rejected, report = audit_candidates(
        read_jsonl(args.candidates),
        min_topic_coverage=args.min_topic_coverage,
        min_length_ratio=args.min_length_ratio,
        max_length_ratio=args.max_length_ratio,
        reject_image_semantic_mismatch=not args.allow_image_semantic_mismatch,
        require_known_axis=not args.allow_legacy_axis,
        deduplicate=not args.keep_duplicates,
    )
    write_jsonl(args.accepted_out, accepted)
    if args.rejected_out:
        write_jsonl(args.rejected_out, rejected)
    write_json(args.report, report)
    print(
        f"Audited {report['candidates']} candidates: "
        f"accepted={report['accepted']}, rejected={report['rejected']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
