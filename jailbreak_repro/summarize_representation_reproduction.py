from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jailbreak_repro.io_utils import read_json, repo_root, write_json


DEFAULT_ROOT = repo_root() / "runs" / "representation_repository_repro"


def _metric(summary: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    representation = summary.get("representation_detector")
    if isinstance(representation, dict):
        return str(representation.get("method") or ""), representation
    hiddendetect = summary.get("hiddendetect")
    if isinstance(hiddendetect, dict):
        return "hiddendetect", hiddendetect
    return None, None


def collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*/external/*/summary.json")):
        relative = path.relative_to(root).parts
        if len(relative) < 5:
            continue
        model, path_method, _, source = relative[:4]
        summary = read_json(path)
        method, metrics = _metric(summary)
        if metrics is None:
            continue
        tpr = metrics.get("tpr")
        if source == "csdj" and tpr is None:
            # Older platform summaries did not map attack `label=1` into the
            # safety label helper. CS-DJ contains only harmful attack samples.
            tpr = metrics.get("detection_rate")
        fpr = metrics.get("fpr")
        scored_count = metrics.get("scored_count")
        rows.append(
            {
                "model": model,
                "method": method or path_method,
                "source": source,
                "protocol": metrics.get("protocol"),
                "artifact_fingerprint": metrics.get("artifact_fingerprint")
                or metrics.get("profile_fingerprint"),
                "layer": metrics.get("layer"),
                "threshold": metrics.get("threshold"),
                "n": metrics.get("labeled_count") or scored_count,
                "safe_count": metrics.get("safe_count"),
                "unsafe_count": (
                    scored_count if source == "csdj" else metrics.get("unsafe_count")
                ),
                "auprc": metrics.get("auprc"),
                "auroc": metrics.get("auroc"),
                "tpr": tpr,
                "fpr": fpr,
                "false_negative_rate": 1.0 - float(tpr) if tpr is not None else None,
                "detection_rate": metrics.get("detection_rate"),
                "paper_training_protocol_count": metrics.get("paper_training_protocol_count"),
                "summary_path": str(path.resolve()),
            }
        )
    return rows


def _display(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "model",
        "method",
        "source",
        "protocol",
        "n",
        "auroc",
        "tpr",
        "fpr",
        "false_negative_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Repository Representation Detector Results",
        "",
        "Detection-only results with frozen training artifacts. XSTest FPR measures safe-prompt "
        "over-detection; CS-DJ false-negative rate measures attack misses.",
        "",
        "| Model | Method | Test | Protocol | N | AUPRC | AUROC | TPR | FPR | FNR |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {method} | {source} | {protocol} | {n} | {auprc} | {auroc} | {tpr} | {fpr} | {fnr} |".format(
                model=row["model"],
                method=row["method"],
                source=row["source"],
                protocol=row.get("protocol") or "-",
                n=_display(row.get("n")),
                auprc=_display(row.get("auprc")),
                auroc=_display(row.get("auroc")),
                tpr=_display(row.get("tpr")),
                fpr=_display(row.get("fpr")),
                fnr=_display(row.get("false_negative_rate")),
            )
        )
    if not rows:
        lines.extend(["", "No completed external summaries were found."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate XSTest and CS-DJ detector summaries.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-prefix", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    prefix = (
        args.out_prefix.expanduser().resolve()
        if args.out_prefix
        else root / "external_detection_summary"
    )
    rows = collect(root)
    write_json(prefix.with_suffix(".json"), {"root": str(root), "results": rows})
    write_csv(prefix.with_suffix(".csv"), rows)
    write_markdown(prefix.with_suffix(".md"), rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Wrote {len(rows)} result rows under {prefix.parent}")


if __name__ == "__main__":
    main()
