from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .panels import PANEL_TITLES
from .visualize import (
    COLORS,
    SOURCE_STYLES,
    _analysis_panel_metadata,
    _pyplot,
    _read_csv_if_nonempty,
)
from .io_utils import write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Qwen and Gemma on normalized decoder depth.")
    parser.add_argument(
        "--analysis",
        action="append",
        required=True,
        help="NAME=analysis_directory; repeat once per model.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--readout", default="last")
    parser.add_argument("--common-panel", action="store_true")
    return parser.parse_args(argv)


def _parse_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    probes: list[pd.DataFrame] = []
    attacks: list[pd.DataFrame] = []
    protocols: list[dict[str, object]] = []
    for name, directory in map(_parse_spec, args.analysis):
        analysis = json.loads((directory / "analysis.json").read_text(encoding="utf-8"))
        run = analysis.get("extraction_run") or {}
        if run.get("status") != "complete":
            raise RuntimeError(f"{name} extraction is not complete")
        coverage = analysis.get("coverage") or {}
        modality_panel, strong_labels_only = _analysis_panel_metadata(analysis)
        protocol = {
            "model": name,
            "manifest_sha256": analysis.get("manifest_sha256"),
            "analysis_sample_ids_sha256": coverage.get("analysis_sample_ids_sha256"),
            "strong_label_sensitivity": strong_labels_only,
            "modality_panel": modality_panel,
            "by_modality": coverage.get("by_modality") or {},
            "common_panel_sample_ids_sha256": (analysis.get("common_multimodal_panel") or {}).get("sample_ids_sha256"),
            "storage_dtype": run.get("storage_dtype"),
        }
        protocols.append(protocol)
        probe_name = "common_panel_layer_metrics.csv" if args.common_panel else "layer_probe_metrics.csv"
        attack_name = "common_panel_attack_metrics.csv" if args.common_panel else "attack_shift_metrics.csv"
        probe = _read_csv_if_nonempty(directory / probe_name)
        if probe.empty:
            raise RuntimeError(f"{name} has no rows in {probe_name}")
        probe = probe[probe["readout"] == args.readout].copy()
        if probe.empty:
            raise RuntimeError(f"{name} has no {args.readout!r} rows in {probe_name}")
        probe["model"] = name
        probes.append(probe)
        attack_path = directory / attack_name
        if attack_path.is_file() and attack_path.stat().st_size:
            attack = _read_csv_if_nonempty(attack_path)
            if attack.empty:
                continue
            attack = attack[attack["readout"] == args.readout].copy()
            depth = probe[["layer", "normalized_depth"]].drop_duplicates()
            attack = attack.merge(depth, on="layer", how="left", validate="many_to_one")
            if attack["normalized_depth"].isna().any():
                raise ValueError(f"{name} attack rows contain layers absent from the probe table")
            attack["model"] = name
            attacks.append(attack)
    invariant_keys = [
        "manifest_sha256",
        "analysis_sample_ids_sha256",
        "strong_label_sensitivity",
        "modality_panel",
        "storage_dtype",
    ]
    if args.common_panel:
        invariant_keys.append("common_panel_sample_ids_sha256")
    for key in invariant_keys:
        if len({str(value.get(key)) for value in protocols}) != 1:
            raise ValueError(f"Cross-model protocol mismatch for {key}: {protocols}")
    probe_all = pd.concat(probes, ignore_index=True)
    probe_all.to_csv(out_dir / "cross_model_layer_metrics.csv", index=False)
    attack_all = pd.concat(attacks, ignore_index=True) if attacks else pd.DataFrame()
    attack_all.to_csv(out_dir / "cross_model_attack_metrics.csv", index=False)
    write_json_atomic(
        out_dir / "comparison.json",
        {
            "schema_version": "cross_model_comparison_v1",
            "panel": "common_multimodal" if args.common_panel else "standard",
            "modality_panel": protocols[0]["modality_panel"],
            "readout": args.readout,
            "protocols": protocols,
        },
    )

    plt = _pyplot()
    panel_title = PANEL_TITLES[str(protocols[0]["modality_panel"])]
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    for index, (name, subset) in enumerate(probe_all.groupby("model")):
        subset = subset.sort_values("normalized_depth")
        axes[0].plot(subset["normalized_depth"], subset["test_auroc"], label=name, color=COLORS[index % len(COLORS)], linewidth=2.0)
        selected = subset[subset["validation_selected"].astype(bool)]
        axes[0].scatter(selected["normalized_depth"], selected["test_auroc"], color=COLORS[index % len(COLORS)], edgecolor="black", linewidth=0.6)
    axes[0].set(
        title=f"Standard holdout across models ({panel_title})",
        xlabel="Normalized decoder depth",
        ylabel="AUROC",
        ylim=(0.0, 1.02),
    )
    axes[0].legend(frameon=False)
    if not attack_all.empty:
        attack_names = sorted(attack_all["attack"].astype(str).unique().tolist())
        missing_styles = [attack for attack in attack_names if attack not in SOURCE_STYLES]
        if missing_styles:
            raise ValueError(
                f"Cross-model attack curves lack stable source styles for {missing_styles}"
            )
        model_names = sorted(attack_all["model"].astype(str).unique().tolist())
        line_styles = {
            model: ("-", "--", ":", "-.")[index % 4]
            for index, model in enumerate(model_names)
        }
        for (model, attack), subset in attack_all.groupby(["model", "attack"]):
            subset = subset.sort_values("normalized_depth")
            axes[1].plot(
                subset["normalized_depth"],
                subset["tpr"],
                label=f"{model}: {attack}",
                color=SOURCE_STYLES[str(attack)].color,
                linestyle=line_styles[str(model)],
                linewidth=1.6,
            )
        axes[1].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].set(
        title=f"Frozen attack recall across models ({panel_title})",
        xlabel="Normalized decoder depth",
        ylabel="TPR",
        ylim=(0.0, 1.02),
    )
    figure.savefig(out_dir / "cross_model_layer_curves.png", bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote cross-model comparison to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
