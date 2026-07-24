from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import load_activation_table
from .io_utils import canonical_json, read_jsonl, sha256_text
from .panels import MODALITY_PANELS, PANEL_TITLES, modality_panel_mask, source_display_name


COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#7f7f7f"]


@dataclass(frozen=True)
class SourceStyle:
    color: str
    marker: str


# The order is also the stable legend order. Colors are deliberately unique;
# marker shape supplies a second channel for grayscale and color-vision access.
SOURCE_STYLES: dict[str, SourceStyle] = {
    "AdvBench": SourceStyle("#0072B2", "^"),
    "Alpaca": SourceStyle("#E69F00", "o"),
    "CS-DJ": SourceStyle("#009E73", "x"),
    "DAN-Prompts": SourceStyle("#CC79A7", "s"),
    "FigStep": SourceStyle("#D55E00", "x"),
    "HADES": SourceStyle("#56B4E9", "D"),
    "JOOD": SourceStyle("#6F6F6F", "x"),
    "JailBreakV-28K-FigStep": SourceStyle("#882255", "x"),
    "JailBreakV-28K-LLM-Transfer": SourceStyle("#44AA99", "x"),
    "JailBreakV-28K-Query-Related": SourceStyle("#DDCC77", "x"),
    "MM-SafetyBench": SourceStyle("#332288", "v"),
    "MM-Vet": SourceStyle("#AA4499", "P"),
    "OpenAssistant": SourceStyle("#117733", "h"),
    "VizWiz-VQA": SourceStyle("#999933", "*"),
    "XSTest-safe": SourceStyle("#88CCEE", "<"),
    "XSTest-unsafe": SourceStyle("#CC3311", ">"),
}

def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 9,
        }
    )
    return plt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render layer, shift, and PCA diagnostics.")
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--primary-readout", default="last")
    parser.add_argument("--max-pca-samples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args(argv)


def _read_csv_if_nonempty(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _analysis_panel_metadata(analysis: dict[str, Any]) -> tuple[str, bool]:
    coverage = analysis.get("coverage", {})
    top_panel = analysis.get("modality_panel")
    coverage_panel = coverage.get("modality_panel")
    if top_panel is not None and coverage_panel is not None and top_panel != coverage_panel:
        raise ValueError(
            "analysis.json has inconsistent modality_panel values: "
            f"top-level={top_panel!r}, coverage={coverage_panel!r}"
        )
    panel = str(top_panel or coverage_panel or "all")
    if panel not in MODALITY_PANELS:
        raise ValueError(
            f"Unknown analysis modality_panel {panel!r}; expected one of {MODALITY_PANELS}"
        )

    top_strong = analysis.get("strong_label_sensitivity")
    coverage_strong = coverage.get("strong_label_sensitivity")
    if (
        top_strong is not None
        and coverage_strong is not None
        and bool(top_strong) != bool(coverage_strong)
    ):
        raise ValueError(
            "analysis.json has inconsistent strong-label flags: "
            f"top-level={top_strong!r}, coverage={coverage_strong!r}"
        )
    strong = bool(coverage_strong if coverage_strong is not None else top_strong)
    return panel, strong


def _analysis_panel_selection(
    frame: pd.DataFrame,
    sample_ids: np.ndarray,
    analysis: dict[str, Any],
) -> tuple[np.ndarray, str, bool]:
    """Recreate and fingerprint the exact row panel used by analysis."""
    coverage = analysis.get("coverage", {})
    panel, strong = _analysis_panel_metadata(analysis)

    if len(frame) != len(sample_ids):
        raise ValueError(
            f"Manifest/activation row mismatch: frame={len(frame)}, activations={len(sample_ids)}"
        )
    keep = modality_panel_mask(frame["modality"].tolist(), panel)
    if strong:
        if "label_confidence" not in frame:
            raise ValueError("Strong-label analysis requires manifest column 'label_confidence'")
        keep &= frame["label_confidence"].eq("strong").to_numpy()

    expected_ids_sha = coverage.get("analysis_sample_ids_sha256")
    if not isinstance(expected_ids_sha, str) or len(expected_ids_sha) != 64:
        raise ValueError(
            "analysis.json is missing a valid coverage.analysis_sample_ids_sha256; "
            "PCA will not run without an exact panel fingerprint"
        )
    selected_ids = np.asarray(sample_ids).astype(str)[keep]
    actual_ids_sha = sha256_text("\n".join(sorted(selected_ids.tolist())))
    if expected_ids_sha != actual_ids_sha:
        raise ValueError(
            "Visualization sample panel differs from the analysis sample fingerprint: "
            f"panel={panel!r}, strong={strong}, expected={expected_ids_sha}, actual={actual_ids_sha}"
        )
    return keep, panel, strong


def _add_source_display(
    frame: pd.DataFrame,
    validate_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    output = frame.copy()
    output["source_display"] = [
        source_display_name(source, label)
        for source, label in zip(output["source"], output["label"], strict=True)
    ]
    if validate_mask is None:
        style_values = output["source_display"]
    else:
        validate_mask = np.asarray(validate_mask, dtype=bool)
        if validate_mask.shape != (len(output),):
            raise ValueError(
                f"Source-style validation mask has shape {validate_mask.shape}; "
                f"expected {(len(output),)}"
            )
        style_values = output.loc[validate_mask, "source_display"]
    missing = sorted(set(style_values.astype(str)) - set(SOURCE_STYLES))
    if missing:
        raise ValueError(
            "PCA has sources without an explicit stable style: "
            f"{missing}. Add each source to SOURCE_STYLES before plotting."
        )
    return output


def _save_layer_curves(
    probe: pd.DataFrame,
    attack: pd.DataFrame,
    path: Path,
    primary_readout: str,
    panel_title: str,
) -> None:
    plt = _pyplot()
    readouts = sorted(probe["readout"].unique().tolist())
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    for index, readout in enumerate(readouts):
        subset = probe[probe["readout"] == readout].sort_values("normalized_depth")
        axes[0].plot(
            subset["normalized_depth"],
            subset["test_auroc"],
            label=readout,
            color=COLORS[index % len(COLORS)],
            linewidth=1.8,
        )
        chosen = subset[subset["validation_selected"].astype(bool)]
        if not chosen.empty:
            axes[0].scatter(
                chosen["normalized_depth"],
                chosen["test_auroc"],
                color=COLORS[index % len(COLORS)],
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
    axes[0].set(
        title=f"Standard holdout separability ({panel_title})",
        xlabel="Normalized decoder depth",
        ylabel="AUROC",
        ylim=(0.0, 1.02),
    )
    axes[0].legend(frameon=False)

    axes[1].set_title(f"Frozen-threshold attack recall ({panel_title})")
    if attack.empty:
        axes[1].text(0.5, 0.5, "No attack rows", ha="center", va="center")
    else:
        primary = attack[attack["readout"] == primary_readout].copy()
        if primary.empty:
            primary = attack[attack["readout"] == attack["readout"].iloc[0]].copy()
        depth = probe[probe["readout"] == primary["readout"].iloc[0]][
            ["layer", "normalized_depth"]
        ].drop_duplicates()
        primary = primary.merge(depth, on="layer", how="left", validate="many_to_one")
        for index, (name, subset) in enumerate(primary.groupby("attack")):
            subset = subset.sort_values("normalized_depth")
            axes[1].plot(
                subset["normalized_depth"],
                subset["tpr"],
                label=name,
                color=COLORS[index % len(COLORS)],
                linewidth=1.8,
            )
        axes[1].legend(frameon=False)
    axes[1].set(xlabel="Normalized decoder depth", ylabel="TPR", ylim=(0.0, 1.02))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _heatmap(matrix: pd.DataFrame, path: Path, title: str, colorbar: str, vmin: float | None = None, vmax: float | None = None) -> None:
    plt = _pyplot()
    width = max(7.0, 0.22 * matrix.shape[1] + 2.4)
    height = max(2.8, 0.52 * matrix.shape[0] + 1.7)
    figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = axis.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_xlabel("Decoder layer")
    axis.set_yticks(np.arange(len(matrix.index)), labels=matrix.index.tolist())
    step = max(1, int(np.ceil(matrix.shape[1] / 14)))
    ticks = np.arange(0, matrix.shape[1], step)
    axis.set_xticks(ticks, labels=[str(matrix.columns[index]) for index in ticks])
    axis.grid(False)
    figure.colorbar(image, ax=axis, label=colorbar, shrink=0.88)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _save_panel_composition(
    analysis: dict[str, Any],
    path: Path,
    panel_title: str,
) -> None:
    counts = (analysis.get("coverage") or {}).get("standard_by_source_label") or {}
    if not isinstance(counts, dict) or not counts:
        return
    sources = sorted(str(source) for source in counts)
    benign = np.array(
        [int((counts.get(source) or {}).get("0", 0)) for source in sources],
        dtype=int,
    )
    harmful = np.array(
        [int((counts.get(source) or {}).get("1", 0)) for source in sources],
        dtype=int,
    )
    positions = np.arange(len(sources), dtype=float)
    plt = _pyplot()
    height = max(3.2, 0.42 * len(sources) + 1.5)
    figure, axis = plt.subplots(figsize=(8.4, height), constrained_layout=True)
    axis.barh(positions - 0.18, benign, height=0.34, color="#1f77b4", label="benign (0)")
    axis.barh(positions + 0.18, harmful, height=0.34, color="#d62728", label="harmful (1)")
    axis.set_yticks(positions, labels=sources)
    axis.invert_yaxis()
    axis.set(
        title=f"Standard-panel source/label composition ({panel_title})",
        xlabel="Samples across train, validation, and test",
    )
    axis.legend(frameon=False)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _source_accuracy(source: pd.DataFrame) -> pd.Series:
    output = pd.Series(np.nan, index=source.index, dtype=float)
    positive = source["positive_n"] > 0
    negative = source["negative_n"] > 0
    output.loc[positive & ~negative] = source.loc[positive & ~negative, "tpr"]
    output.loc[negative & ~positive] = source.loc[negative & ~positive, "tnr"]
    output.loc[positive & negative] = source.loc[positive & negative, "balanced_accuracy"]
    return output


def _balanced_indices(frame: pd.DataFrame, valid: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    eligible = np.where(valid)[0]
    if len(eligible) <= maximum:
        return eligible
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[int]] = {}
    for index in eligible:
        display = (
            frame.iloc[index]["source_display"]
            if "source_display" in frame
            else f"{frame.iloc[index]['source']}:{frame.iloc[index]['label']}"
        )
        key = str(display)
        buckets.setdefault(key, []).append(int(index))
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[int] = []
    keys = sorted(buckets)
    while keys and len(selected) < maximum:
        for key in list(keys):
            if len(selected) >= maximum:
                break
            if buckets[key]:
                selected.append(buckets[key].pop())
            if not buckets[key]:
                keys.remove(key)
    return np.array(sorted(selected), dtype=int)


def _save_pca(
    activations: Path,
    manifest: Path,
    analysis: dict[str, Any],
    readout: str,
    maximum: int,
    seed: int,
    out_dir: Path,
) -> None:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    table = load_activation_table(activations)
    manifest_records = list(read_jsonl(manifest))
    manifest_sha256 = sha256_text(
        "\n".join(canonical_json(row) for row in manifest_records)
    )
    for owner, expected_sha256 in (
        ("analysis", analysis.get("manifest_sha256")),
        ("activation", table.metadata.get("manifest_sha256")),
    ):
        if expected_sha256 and str(expected_sha256) != manifest_sha256:
            raise ValueError(
                f"{owner.capitalize()}/manifest fingerprint mismatch: "
                f"expected={expected_sha256}, actual={manifest_sha256}"
            )
    manifest_rows = {str(row["sample_id"]): row for row in manifest_records}
    if len(manifest_rows) != len(manifest_records):
        raise ValueError("Manifest contains duplicate sample IDs")
    missing_ids = [value for value in table.sample_ids.tolist() if value not in manifest_rows]
    if missing_ids:
        raise ValueError(f"Activation IDs are missing from the manifest: {missing_ids[:5]}")
    frame = pd.DataFrame([manifest_rows[value] for value in table.sample_ids.tolist()])
    frame["label"] = frame["label"].astype(int)
    analysis_keep, panel, strong = _analysis_panel_selection(
        frame, table.sample_ids, analysis
    )
    frame = _add_source_display(frame, analysis_keep)
    panel_title = PANEL_TITLES[panel]
    readout_matches = np.where(table.readouts.astype(str) == readout)[0]
    if len(readout_matches) != 1:
        raise ValueError(f"Readout {readout!r} not found in activation archive")
    selected_layer = analysis["selected_layers"][readout]["layer"]
    layer_matches = np.where(table.layers.astype(int) == int(selected_layer))[0]
    if len(layer_matches) != 1:
        raise ValueError(f"Selected layer {selected_layer} not found")
    readout_index = int(readout_matches[0])
    layer_index = int(layer_matches[0])
    valid = table.readout_valid[:, readout_index] & analysis_keep
    indices = _balanced_indices(frame, valid, maximum, seed)
    fit_mask = (
        valid
        & ~frame["is_attack"].astype(bool).to_numpy()
        & frame["split"].eq("train").to_numpy()
    )
    if int(fit_mask.sum()) < 2:
        raise RuntimeError("PCA requires at least two valid standard-train samples")
    fit_X = table.activations[fit_mask, readout_index, layer_index].astype(np.float32)
    point_X = table.activations[indices, readout_index, layer_index].astype(np.float32)
    scaler = StandardScaler().fit(fit_X)
    pca = PCA(n_components=2, random_state=seed).fit(scaler.transform(fit_X))
    embedding = pca.transform(scaler.transform(point_X))
    points = frame.iloc[indices].copy()
    points["pca_1"] = embedding[:, 0]
    points["pca_2"] = embedding[:, 1]
    points["analysis_modality_panel"] = panel
    points["analysis_strong_labels_only"] = strong
    points.to_csv(out_dir / "pca_points.csv", index=False)

    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 4.6), constrained_layout=True)
    for label, name, color in ((0, "benign", "#1f77b4"), (1, "harmful", "#d62728")):
        mask = points["label"].to_numpy(dtype=int) == label
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1], s=16, alpha=0.62, label=name, c=color, edgecolors="none")
    axes[0].set(
        title=f"Train-fitted PCA by intent label ({panel_title}), layer {selected_layer}",
        xlabel="PC1",
        ylabel="PC2",
    )
    axes[0].legend(frameon=False)

    present_sources = set(points["source_display"].astype(str))
    for source in (name for name in SOURCE_STYLES if name in present_sources):
        subset = points[points["source_display"].eq(source)]
        positions = subset.index.to_numpy()
        local = np.array([points.index.get_loc(position) for position in positions])
        style = SOURCE_STYLES[source]
        attack = bool(subset["is_attack"].astype(bool).all())
        if bool(subset["is_attack"].astype(bool).any()) != attack:
            raise ValueError(f"Source display group {source!r} mixes attack and non-attack rows")
        scatter_kwargs: dict[str, Any] = {
            "s": 27 if attack else 20,
            "alpha": 0.82 if attack else 0.66,
            "label": source,
            "c": style.color,
            "marker": style.marker,
        }
        if attack:
            scatter_kwargs["linewidths"] = 1.25
        else:
            scatter_kwargs["edgecolors"] = "none"
        axes[1].scatter(
            embedding[local, 0],
            embedding[local, 1],
            **scatter_kwargs,
        )
    axes[1].set(
        title=f"PCA by source ({panel_title}; x = external attack)",
        xlabel="PC1",
        ylabel="PC2",
    )
    axes[1].legend(
        frameon=False,
        fontsize=7,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    figure.savefig(out_dir / "selected_layer_pca.png", bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    analysis_dir = args.analysis_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = pd.read_csv(analysis_dir / "layer_probe_metrics.csv")
    attack_path = analysis_dir / "attack_shift_metrics.csv"
    attack = _read_csv_if_nonempty(attack_path)
    source_path = analysis_dir / "source_metrics.csv"
    source = _read_csv_if_nonempty(source_path)
    source_label = _read_csv_if_nonempty(analysis_dir / "source_label_metrics.csv")
    loso = _read_csv_if_nonempty(analysis_dir / "leave_one_source_out.csv")
    loso_label = _read_csv_if_nonempty(
        analysis_dir / "leave_one_source_out_label_metrics.csv"
    )
    analysis = json.loads((analysis_dir / "analysis.json").read_text(encoding="utf-8"))
    panel, _ = _analysis_panel_metadata(analysis)
    panel_title = PANEL_TITLES[panel]

    _save_layer_curves(
        probe,
        attack,
        out_dir / "layer_curves.png",
        args.primary_readout,
        panel_title,
    )
    _save_panel_composition(
        analysis,
        out_dir / "panel_composition.png",
        panel_title,
    )
    common_probe = _read_csv_if_nonempty(analysis_dir / "common_panel_layer_metrics.csv")
    common_attack = _read_csv_if_nonempty(analysis_dir / "common_panel_attack_metrics.csv")
    if not common_probe.empty:
        _save_layer_curves(
            common_probe,
            common_attack,
            out_dir / "common_multimodal_panel_layer_curves.png",
            args.primary_readout,
            f"{panel_title}; common multimodal panel",
        )
    if not attack.empty:
        subset = attack[attack["readout"] == args.primary_readout]
        matrix = subset.pivot(index="attack", columns="layer", values="standardized_score_shift")
        _heatmap(
            matrix,
            out_dir / "attack_score_shift_heatmap.png",
            f"Attack score-location shift ({panel_title})",
            "Shift / standard harmful SD",
        )
        domain = subset.pivot(index="attack", columns="layer", values="group_cv_harmful_domain_auroc")
        _heatmap(
            domain,
            out_dir / "attack_domain_auroc_heatmap.png",
            f"Attack vs standard-harmful domain separability ({panel_title})",
            "Domain AUROC",
            0.0,
            1.0,
        )
    source_heatmap_written = False
    if not loso_label.empty and {"eligible", "label_recall"}.issubset(loso_label.columns):
        subset = loso_label[
            (loso_label["readout"] == args.primary_readout)
            & loso_label["eligible"].astype(str).str.lower().isin({"true", "1"})
        ].copy()
        if not subset.empty:
            matrix = subset.pivot(
                index="held_out_source_label", columns="layer", values="label_recall"
            )
            _heatmap(
                matrix,
                out_dir / "source_generalization_heatmap.png",
                f"Leave-one-source-out generalization ({panel_title})",
                "TPR for unsafe/harmful; TNR for safe/benign",
                0.0,
                1.0,
            )
            source_heatmap_written = True
    elif not loso.empty and "eligible" in loso:
        subset = loso[
            (loso["readout"] == args.primary_readout)
            & loso["eligible"].astype(str).str.lower().isin({"true", "1"})
        ].copy()
        if not subset.empty:
            subset["class_accuracy"] = _source_accuracy(subset)
            matrix = subset.pivot(index="held_out_source", columns="layer", values="class_accuracy")
            _heatmap(
                matrix,
                out_dir / "source_generalization_heatmap.png",
                f"Leave-one-source-out generalization ({panel_title})",
                "TPR / TNR / balanced accuracy",
                0.0,
                1.0,
            )
            source_heatmap_written = True
    if (
        not source_heatmap_written
        and not source_label.empty
        and "label_recall" in source_label
    ):
        subset = source_label[source_label["readout"] == args.primary_readout].copy()
        if not subset.empty:
            matrix = subset.pivot(index="source_label", columns="layer", values="label_recall")
            _heatmap(
                matrix,
                out_dir / "source_generalization_heatmap.png",
                f"Pooled probe by source-label test slice ({panel_title})",
                "TPR for unsafe/harmful; TNR for safe/benign",
                0.0,
                1.0,
            )
            source_heatmap_written = True
    if not source_heatmap_written and not source.empty:
        subset = source[source["readout"] == args.primary_readout].copy()
        subset["class_accuracy"] = _source_accuracy(subset)
        matrix = subset.pivot(index="source", columns="layer", values="class_accuracy")
        _heatmap(
            matrix,
            out_dir / "source_generalization_heatmap.png",
            f"Pooled probe by source test slice ({panel_title})",
            "TPR / TNR / balanced accuracy",
            0.0,
            1.0,
        )
    _save_pca(
        args.activations.expanduser().resolve(),
        args.manifest.expanduser().resolve(),
        analysis,
        args.primary_readout,
        args.max_pca_samples,
        args.seed,
        out_dir,
    )
    print(f"Wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
