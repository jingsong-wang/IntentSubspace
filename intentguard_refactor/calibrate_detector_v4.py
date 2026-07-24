from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from intentguard.detector import CISRDetector
from intentguard.io import read_jsonl, write_json, write_jsonl
from intentguard.selective import (
    SelectiveRoute,
    SelectiveThresholds,
    choose_selective_thresholds,
    decide_route,
    selective_metrics,
)


SPLITS = ("train", "validation", "calibration", "test")


def _labels_and_scores(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    scores = np.asarray([float(row["cisr_probability"]) for row in rows], dtype=float)
    return labels, scores


def _split_report(
    rows: list[dict[str, Any]],
    thresholds: SelectiveThresholds,
    confidence: float,
) -> dict[str, Any]:
    labels, scores = _labels_and_scores(rows)
    return selective_metrics(labels, scores, thresholds, confidence=confidence)


def _group_report(
    rows: list[dict[str, Any]],
    thresholds: SelectiveThresholds,
    confidence: float,
    key: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    report: dict[str, Any] = {}
    for name, members in sorted(grouped.items()):
        labels = {int(row["label"]) for row in members}
        if labels != {0, 1}:
            continue
        report[name] = _split_report(members, thresholds, confidence)
    return report


def _copy_v3_artifact(
    source_path: Path,
    destination_path: Path,
    thresholds: SelectiveThresholds,
    selection: dict[str, Any],
) -> None:
    with np.load(source_path, allow_pickle=True) as source:
        payload = {name: np.asarray(source[name]).copy() for name in source.files}
    source_format = str(np.asarray(payload.get("format_version", [""])).reshape(-1)[0])
    if not source_format.lower().startswith("cisr_v3_detector"):
        raise ValueError(
            f"CISR_v4 calibration requires a frozen CISR_v3 detector, got {source_format!r}"
        )

    safe_selection = selection["confident_safe_selection"]
    danger_selection = selection["confident_dangerous_selection"]
    targets = selection["targets"]
    payload.update(
        {
            "format_version": np.array(["CISR_v4_detector_v1"]),
            "threshold": np.array([thresholds.danger_min], dtype=np.float64),
            "safe_threshold": np.array([thresholds.safe_max], dtype=np.float64),
            "danger_threshold": np.array([thresholds.danger_min], dtype=np.float64),
            "safe_route_enabled": np.array([thresholds.safe_enabled], dtype=bool),
            "danger_route_enabled": np.array([thresholds.danger_enabled], dtype=bool),
            "confident_safe_error_upper_bound": np.array(
                [safe_selection.get("error_upper_bound") or 1.0], dtype=np.float64
            ),
            "confident_dangerous_error_upper_bound": np.array(
                [danger_selection.get("error_upper_bound") or 1.0], dtype=np.float64
            ),
            "maximum_confident_safe_error": np.array(
                [targets["maximum_confident_safe_error"]], dtype=np.float64
            ),
            "maximum_confident_dangerous_error": np.array(
                [targets["maximum_confident_dangerous_error"]], dtype=np.float64
            ),
            "maximum_harmful_unsafe_escape": np.array(
                [targets["maximum_harmful_unsafe_escape"]], dtype=np.float64
            ),
            "maximum_benign_hard_refusal": np.array(
                [targets["maximum_benign_hard_refusal"]], dtype=np.float64
            ),
            "coverage_confidence": np.array(
                [selection["coverage_confidence"]], dtype=np.float64
            ),
            "deployment_constraints_met": np.array(
                [selection["deployment_constraints_met"]], dtype=bool
            ),
            "source_detector_format": np.array([source_format]),
        }
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination_path, **payload)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    thresholds = summary["thresholds"]
    lines = [
        "# CISR_v4 Selective Detection Report",
        "",
        f"Source detector: `{summary['source_detector']}`",
        f"Safe threshold: `{thresholds['safe_max']:.6f}` (enabled: `{thresholds['safe_enabled']}`)",
        f"Danger threshold: `{thresholds['danger_min']:.6f}` (enabled: `{thresholds['danger_enabled']}`)",
        f"Deployment constraints met: `{'yes' if summary['deployment_constraints_met'] else 'no'}`",
        "",
        "| split | safe coverage | safe error | harmful escape | review | danger coverage | danger error | benign hard-refusal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        item = summary["splits"][split]

        def fmt(value: Any) -> str:
            return "n/a" if value is None else f"{float(value):.4f}"

        lines.append(
            f"| {split} | {fmt(item['confident_safe_rate'])} | "
            f"{fmt(item['confident_safe_error_rate'])} | "
            f"{fmt(item['harmful_unsafe_escape_rate'])} | {fmt(item['review_rate'])} | "
            f"{fmt(item['confident_dangerous_rate'])} | "
            f"{fmt(item['confident_dangerous_error_rate'])} | "
            f"{fmt(item['benign_hard_refusal_rate'])} |"
        )
    lines.extend(
        [
            "",
            "`confident_safe_error_rate` is harmful / confident-safe. "
            "`confident_dangerous_error_rate` is benign / confident-dangerous.",
            "",
            "`harmful_unsafe_escape_rate` and `benign_hard_refusal_rate` use the full harmful "
            "and benign classes as denominators and must be reported alongside route-conditional errors.",
            "",
            "This calibration stage reuses its supplied fitted representation; only thresholds and routing change here.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upgrade a frozen CISR_v3 detector to CISR_v4 selective routing."
    )
    parser.add_argument("--source-detector", type=Path, required=True)
    parser.add_argument("--detection-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration-split", default="calibration")
    parser.add_argument("--constraint-group", default="carrier_type")
    parser.add_argument("--minimum-group-selected", type=int, default=10)
    parser.add_argument("--maximum-confident-safe-error", type=float, default=0.02)
    parser.add_argument("--maximum-confident-dangerous-error", type=float, default=0.02)
    parser.add_argument("--maximum-harmful-unsafe-escape", type=float, default=0.05)
    parser.add_argument("--maximum-benign-hard-refusal", type=float, default=0.05)
    parser.add_argument("--coverage-confidence", type=float, default=0.95)
    parser.add_argument("--require-deployable", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.detection_results)
    if not rows:
        raise ValueError("detection results are empty")
    calibration_rows = [
        row for row in rows if str(row.get("evaluation_split")) == args.calibration_split
    ]
    if not calibration_rows:
        raise ValueError(f"no rows found for calibration split {args.calibration_split!r}")
    labels, scores = _labels_and_scores(calibration_rows)
    groups = [str(row.get(args.constraint_group, "")) for row in calibration_rows]
    thresholds, selection = choose_selective_thresholds(
        labels,
        scores,
        maximum_safe_error=args.maximum_confident_safe_error,
        maximum_danger_error=args.maximum_confident_dangerous_error,
        maximum_harmful_escape=args.maximum_harmful_unsafe_escape,
        maximum_benign_hard_refusal=args.maximum_benign_hard_refusal,
        confidence=args.coverage_confidence,
        groups=groups,
        minimum_group_selected=args.minimum_group_selected,
    )

    detector_path = args.out_dir / "detector.npz"
    _copy_v3_artifact(args.source_detector, detector_path, thresholds, selection)
    split_reports = {}
    for split in SPLITS:
        split_rows = [row for row in rows if str(row.get("evaluation_split")) == split]
        if not split_rows:
            raise ValueError(f"missing required split {split!r}")
        split_reports[split] = _split_report(
            split_rows, thresholds, args.coverage_confidence
        )
    test_rows = [row for row in rows if str(row.get("evaluation_split")) == "test"]
    summary = {
        "format_version": "CISR_v4_detection_protocol_v1",
        "source_detector": str(args.source_detector),
        "source_detector_sha1": hashlib.sha1(args.source_detector.read_bytes()).hexdigest(),
        "source_detection_results": str(args.detection_results),
        "detector": str(detector_path),
        "thresholds": selection["thresholds"],
        "targets": selection["targets"],
        "coverage_confidence": args.coverage_confidence,
        "constraint_group": args.constraint_group,
        "minimum_group_selected": args.minimum_group_selected,
        "deployment_constraints_met": selection["deployment_constraints_met"],
        "calibration_selection": selection,
        "splits": split_reports,
        "test_by_condition": _group_report(
            test_rows, thresholds, args.coverage_confidence, "condition"
        ),
        "test_by_carrier_type": _group_report(
            test_rows, thresholds, args.coverage_confidence, "carrier_type"
        ),
        "test_by_intent_family": _group_report(
            test_rows, thresholds, args.coverage_confidence, "intent_family"
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "detection_summary.json", summary)
    _write_report(args.out_dir / "detection_report.md", summary)

    safe_bound = selection["confident_safe_selection"].get("error_upper_bound")
    danger_bound = selection["confident_dangerous_selection"].get("error_upper_bound")
    output_rows = []
    for row in rows:
        decision = decide_route(
            float(row["cisr_probability"]),
            thresholds,
            safe_error_upper_bound=safe_bound,
            danger_error_upper_bound=danger_bound,
        )
        route = decision.route
        action = {
            SelectiveRoute.CONFIDENT_SAFE: "normal_generation",
            SelectiveRoute.REVIEW: "safe_layer_route",
            SelectiveRoute.CONFIDENT_DANGEROUS: "hard_refusal",
        }[route]
        output_rows.append(
            {
                **row,
                "cisr_version": "cisr4",
                "cisr_detector_format": "CISR_v4_detector_v1",
                "cisr_safe_threshold": thresholds.safe_max,
                "cisr_danger_threshold": thresholds.danger_min,
                "cisr_route": route.value,
                "cisr_route_margin": decision.route_margin,
                "cisr_route_error_upper_bound": decision.route_error_upper_bound,
                "cisr_requires_intervention": decision.requires_intervention,
                "cisr_planned_action": action,
                "cisr_deployment_constraints_met": selection[
                    "deployment_constraints_met"
                ],
            }
        )
    write_jsonl(args.out_dir / "detection_results.jsonl", output_rows)

    loaded = CISRDetector.load(detector_path)
    if (
        loaded.format_version != "CISR_v4_detector_v1"
        or loaded.safe_threshold != thresholds.safe_max
        or loaded.danger_threshold != thresholds.danger_min
    ):
        raise RuntimeError("serialized CISR_v4 detector failed round-trip validation")
    print(
        f"Wrote CISR_v4 detector to {detector_path}; "
        f"safe<={thresholds.safe_max:.6f}, danger>={thresholds.danger_min:.6f}, "
        f"deployable={selection['deployment_constraints_met']}"
    )
    if args.require_deployable and not selection["deployment_constraints_met"]:
        raise RuntimeError("CISR_v4 calibration could not certify both direct-decision regions")


if __name__ == "__main__":
    main()
