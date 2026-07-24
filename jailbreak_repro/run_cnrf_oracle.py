from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .cnrf_oracle import summarize_oracle_run
from .io_utils import repo_root, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and summarize the label-leaking CNRF arrow-bank Oracle ceiling."
    )
    parser.add_argument(
        "--work",
        type=Path,
        default=Path("counterfactual_risk_field/work/v2_axes_temp07"),
        help="CNRF work directory containing manifest, activations, and fitted results.",
    )
    parser.add_argument("--model-tag", default="qwen25vl7b")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("counterfactual_risk_field/configs/protocol_v2_diverse_axes.json"),
    )
    parser.add_argument(
        "--oracle-config",
        type=Path,
        default=Path("counterfactual_risk_field/configs/oracle_v2.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Defaults to jailbreak_repro/runs/cnrf_oracle/<model>/<work-name>.",
    )
    parser.add_argument("--pack-budgets")
    parser.add_argument("--max-fprs")
    parser.add_argument("--random-subsets", type=int)
    parser.add_argument("--pack-policy", choices=["abstain_safe", "abstain_risk"])
    parser.add_argument("--pack-max-fpr", type=float)
    parser.add_argument("--skip-axis-subsets", action="store_true")
    parser.add_argument("--skip-pack-search", action="store_true")
    parser.add_argument("--max-axis-subsets", type=int)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Regenerate jailbreak_repro summaries from an existing raw/ directory.",
    )
    return parser.parse_args(argv)


def _resolve(path: Path, root: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def _oracle_command(oracle_argv: list[str]) -> list[str]:
    """Launch through the module CLI so both main() and main(argv) are supported."""
    return [
        sys.executable,
        "-m",
        "counterfactual_risk_field.scripts.run_oracle_ceiling",
        *oracle_argv,
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    work = _resolve(args.work, root)
    config = _resolve(args.config, root)
    oracle_config = _resolve(args.oracle_config, root)
    out_dir = (
        _resolve(args.out_dir, root)
        if args.out_dir
        else root / "jailbreak_repro" / "runs" / "cnrf_oracle" / args.model_tag / work.name
    )
    raw_dir = out_dir / "raw"
    inputs = {
        "manifest": work / "experiment.jsonl",
        "activations": work / "activations" / args.model_tag,
        "source_summary": work / "results" / args.model_tag / "summary.json",
        "config": config,
        "oracle_config": oracle_config,
    }
    missing = [f"{name}: {path}" for name, path in inputs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing CNRF Oracle inputs:\n" + "\n".join(missing))

    launch = {
        "format_version": "jailbreak_repro_cnrf_oracle_launch_v1",
        "method": "cnrf",
        "oracle_only": True,
        "paper_claim_compatible": False,
        "model_tag": args.model_tag,
        "work": str(work),
        "out_dir": str(out_dir),
        "raw_dir": str(raw_dir),
        "inputs": {name: str(path) for name, path in inputs.items()},
        "overrides": {
            "pack_budgets": args.pack_budgets,
            "max_fprs": args.max_fprs,
            "random_subsets": args.random_subsets,
            "pack_policy": args.pack_policy,
            "pack_max_fpr": args.pack_max_fpr,
            "skip_axis_subsets": args.skip_axis_subsets,
            "skip_pack_search": args.skip_pack_search,
            "max_axis_subsets": args.max_axis_subsets,
            "summary_only": args.summary_only,
        },
    }
    write_json(out_dir / "launch.json", launch)

    if not args.summary_only:
        oracle_argv = [
            "--manifest",
            str(inputs["manifest"]),
            "--activations",
            str(inputs["activations"]),
            "--summary",
            str(inputs["source_summary"]),
            "--config",
            str(config),
            "--oracle-config",
            str(oracle_config),
            "--out-dir",
            str(raw_dir),
        ]
        optional_values = {
            "--pack-budgets": args.pack_budgets,
            "--max-fprs": args.max_fprs,
            "--random-subsets": args.random_subsets,
            "--pack-policy": args.pack_policy,
            "--pack-max-fpr": args.pack_max_fpr,
            "--max-axis-subsets": args.max_axis_subsets,
        }
        for flag, value in optional_values.items():
            if value is not None:
                oracle_argv.extend([flag, str(value)])
        if args.skip_axis_subsets:
            oracle_argv.append("--skip-axis-subsets")
        if args.skip_pack_search:
            oracle_argv.append("--skip-pack-search")
        completed = subprocess.run(_oracle_command(oracle_argv), check=False)
        if completed.returncode != 0:
            return int(completed.returncode)

    payload = summarize_oracle_run(
        raw_dir,
        out_dir,
        model_tag=args.model_tag,
        source_work=work,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "oracle_only": True,
                "out_dir": str(out_dir),
                "result_count": payload["result_count"],
                "primary_result_count": len(payload["primary_results"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
