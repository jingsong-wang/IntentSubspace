from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .io_utils import load_config, repo_root
from .panels import MODALITY_PANELS


STAGES = ("manifest", "extract", "analyze", "visualize", "compare")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the layer-wise OOD study with resumable stages.")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "default.json")
    parser.add_argument("--work-dir", type=Path, default=Path("runs/ood_intent_study"))
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional manifest path override; defaults to <work-dir>/samples.jsonl.",
    )
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--models", help="Comma-separated model names from the config.")
    parser.add_argument("--max-per-label-source", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--modality-panels",
        default="all",
        help="Comma-separated analysis panels: all,text_only,multimodal_only.",
    )
    parser.add_argument("--skip-loso", action="store_true")
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _parse_modality_panels(value: str) -> list[str]:
    panels = list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not panels:
        raise ValueError("--modality-panels must contain at least one panel")
    invalid = [panel for panel in panels if panel not in MODALITY_PANELS]
    if invalid:
        raise ValueError(
            f"Unknown modality panels {invalid}; expected values from {list(MODALITY_PANELS)}"
        )
    return panels


def _panel_output_dir(work: Path, kind: str, panel: str, model: str | None, nested: bool) -> Path:
    if not nested:
        base = work / kind
    elif kind == "analysis":
        base = work / "analysis_panels" / panel
    elif kind == "figures":
        base = work / "figures_panels" / panel
    elif kind == "comparison":
        base = work / "comparison_panels" / panel
    elif kind == "comparison_common_multimodal":
        base = work / "comparison_panels_common_multimodal" / panel
    else:
        raise ValueError(f"Unknown panel output kind {kind!r}")
    return base / model if model else base


def _run(command: list[str], root: Path, dry_run: bool) -> None:
    print("$ " + " ".join(shlex.quote(value) for value in command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=root, check=True)


def _model_command(python: str, manifest: Path, work: Path, model: dict[str, Any], args: argparse.Namespace) -> list[str]:
    command = [
        python,
        "-m",
        "ood_intent_study.extract",
        "--manifest",
        str(manifest),
        "--out-dir",
        str(work / "activations" / model["name"]),
        "--model-name",
        str(model["name"]),
        "--model",
        str(model["model"]),
        "--model-source",
        str(model["model_source"]),
        "--backend",
        str(model["backend"]),
        "--dtype",
        str(model.get("dtype", "bfloat16")),
        "--storage-dtype",
        str(model.get("storage_dtype", "float32")),
        "--attn-implementation",
        str(model.get("attn_implementation", "sdpa")),
        "--layers",
        "all",
        "--readouts",
        "last,non_image_mean,image_mean",
        "--shard-size",
        str(args.shard_size),
        "--resume",
    ]
    if model.get("model_revision"):
        command.extend(["--model-revision", str(model["model_revision"])])
    if args.max_samples:
        command.extend(["--max-samples", str(args.max_samples)])
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    work = args.work_dir if args.work_dir.is_absolute() else root / args.work_dir
    work = work.resolve()
    if args.manifest is None:
        manifest = work / "samples.jsonl"
    else:
        manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
        manifest = manifest.resolve()
    wanted = {value.strip() for value in (args.models or "").split(",") if value.strip()}
    models = [model for model in config["models"] if not wanted or model["name"] in wanted]
    if not models:
        raise ValueError("No configured models matched --models")
    python = sys.executable
    stages = STAGES if args.stage == "all" else (args.stage,)
    modality_panels = _parse_modality_panels(args.modality_panels)
    nested_panel_layout = modality_panels != ["all"]

    if "manifest" in stages:
        command = [python, "-m", "ood_intent_study.build_manifest", "--config", str(config_path), "--out", str(manifest)]
        if args.max_per_label_source:
            command.extend(["--max-per-label-source", str(args.max_per_label_source)])
        if args.allow_missing_images:
            command.append("--allow-missing-images")
        _run(command, root, args.dry_run)
        if not args.allow_missing_images:
            _run(
                [
                    python,
                    "-m",
                    "ood_intent_study.audit_manifest",
                    "--manifest",
                    str(manifest),
                    "--require-images",
                    "--require-sidecar",
                ],
                root,
                args.dry_run,
            )

    if "extract" in stages:
        for model in models:
            _run(_model_command(python, manifest, work, model, args), root, args.dry_run)

    if "analyze" in stages:
        for panel in modality_panels:
            for model in models:
                command = [
                    python,
                    "-m",
                    "ood_intent_study.analyze",
                    "--activations",
                    str(work / "activations" / model["name"]),
                    "--manifest",
                    str(manifest),
                    "--out-dir",
                    str(_panel_output_dir(work, "analysis", panel, model["name"], nested_panel_layout)),
                    "--modality-panel",
                    panel,
                    "--bootstrap",
                    str(args.bootstrap),
                ]
                if args.skip_loso:
                    command.append("--skip-loso")
                _run(command, root, args.dry_run)

    if "visualize" in stages:
        for panel in modality_panels:
            for model in models:
                _run(
                    [
                        python,
                        "-m",
                        "ood_intent_study.visualize",
                        "--analysis-dir",
                        str(_panel_output_dir(work, "analysis", panel, model["name"], nested_panel_layout)),
                        "--activations",
                        str(work / "activations" / model["name"]),
                        "--manifest",
                        str(manifest),
                        "--out-dir",
                        str(_panel_output_dir(work, "figures", panel, model["name"], nested_panel_layout)),
                    ],
                    root,
                    args.dry_run,
                )

    if "compare" in stages:
        for panel in modality_panels:
            command = [python, "-m", "ood_intent_study.compare_models"]
            for model in models:
                analysis_dir = _panel_output_dir(
                    work, "analysis", panel, model["name"], nested_panel_layout
                )
                command.extend(["--analysis", f"{model['name']}={analysis_dir}"])
            command.extend(
                [
                    "--out-dir",
                    str(_panel_output_dir(work, "comparison", panel, None, nested_panel_layout)),
                ]
            )
            _run(command, root, args.dry_run)
            if panel != "text_only":
                common_command = command[:-2] + [
                    "--common-panel",
                    "--out-dir",
                    str(
                        _panel_output_dir(
                            work,
                            "comparison_common_multimodal",
                            panel,
                            None,
                            nested_panel_layout,
                        )
                    ),
                ]
                _run(common_command, root, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
