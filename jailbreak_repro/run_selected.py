from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from jailbreak_repro.run_experiment import BACKEND_CHOICES, MODEL_PRESETS, MODEL_SOURCE_CHOICES, main as run_experiment_main
from jailbreak_repro.representation_detectors import REPRESENTATION_DEFENSE_CHOICES


ATTACK_CHOICES = ["figstep", "csdj", "cs-dj", "jood", "umk"]
DEFENSE_CHOICES = [
    "none",
    "ecso",
    "cider",
    "cisr",
    "cisr2",
    "cisr3",
    "adashield",
    "hiddendetect",
    *REPRESENTATION_DEFENSE_CHOICES,
]


def _model_args(prefix: str, value: str, backend: str, source: str) -> list[str]:
    args: list[str] = []
    if value in MODEL_PRESETS:
        args.extend([f"--{prefix}-preset", value])
    else:
        model_flag = "--model" if prefix == "model" else "--judge-model"
        args.extend([model_flag, value])
    if backend != "auto":
        args.extend([f"--{prefix}-backend", backend])
    if source != "auto":
        source_flag = f"--{prefix}-source" if prefix == "model" else "--judge-model-source"
        args.extend([source_flag, source])
    return args


def build_run_args(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    run_args = _model_args("model", args.victim_model, args.victim_backend, args.victim_source)
    if args.benchmark:
        run_args.extend(["--benchmark", args.benchmark])
    else:
        run_args.extend(["--attack", args.attack])
    run_args.extend(["--defense", args.defense])

    if args.judge_model == "none":
        run_args.extend(["--judge-mode", "none"])
    elif args.judge_model == "heuristic":
        run_args.extend(["--judge-mode", "heuristic"])
    else:
        run_args.extend(["--judge-mode", "model"])
        run_args.extend(_model_args("judge", args.judge_model, args.judge_backend, args.judge_source))
    run_args.extend(["--judge-task", args.judge_task, "--judge-batch-size", str(args.judge_batch_size)])

    if args.max_samples is not None:
        run_args.extend(["--max-samples", str(args.max_samples)])
    if args.out_dir is not None:
        run_args.extend(["--out-dir", str(args.out_dir)])
    run_args.extend(["--dtype", args.dtype, "--device", args.device])
    if args.attn_implementation != "auto":
        run_args.extend(["--attn-implementation", args.attn_implementation])
    if args.profile_generation:
        run_args.append("--profile-generation")
    if args.no_resume:
        run_args.append("--no-resume")
    if args.force_responses:
        run_args.append("--force-responses")
    if args.force_judge:
        run_args.append("--force-judge")
    return run_args + passthrough


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Convenience launcher for jailbreak_repro.run_experiment. "
            "Unknown arguments are forwarded to run_experiment, so attack-specific options still work."
        )
    )
    parser.add_argument("--victim-model", default="mock", help="Victim preset name or explicit model id/path.")
    parser.add_argument("--victim-backend", choices=BACKEND_CHOICES, default="auto")
    parser.add_argument("--victim-source", choices=MODEL_SOURCE_CHOICES, default="auto")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--attack", choices=ATTACK_CHOICES)
    source.add_argument("--benchmark", help="Benchmark name/path, e.g. HADES, jailbreakV, jailbreakV-mini.")
    parser.add_argument("--defense", choices=DEFENSE_CHOICES, default="ecso")
    parser.add_argument(
        "--judge-model",
        default="heuristic",
        help="none, heuristic, a judge preset name, or an explicit judge model id/path.",
    )
    parser.add_argument("--judge-task", choices=["auto", "asr", "xstest"], default="auto")
    parser.add_argument("--judge-batch-size", type=int, default=8)
    parser.add_argument("--judge-backend", choices=BACKEND_CHOICES, default="auto")
    parser.add_argument("--judge-source", choices=MODEL_SOURCE_CHOICES, default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument("--profile-generation", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force-responses", action="store_true")
    parser.add_argument("--force-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the delegated run_experiment command and exit.")
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    run_args = build_run_args(args, passthrough)
    command = "python -m jailbreak_repro.run_experiment " + " ".join(shlex.quote(part) for part in run_args)
    print(command)
    if args.dry_run:
        return
    run_experiment_main(run_args)


if __name__ == "__main__":
    main()
