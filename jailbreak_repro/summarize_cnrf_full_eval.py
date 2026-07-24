from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .io_utils import repo_root, write_json
from .representation_detectors import RepresentationDetector


EXPECTED_COUNTS = {
    "CS-DJ": 750,
    "JOOD": 500,
    "JailbreakV-mini": 280,
    "XSTest": 450,
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _case_name(summary: dict[str, Any]) -> str | None:
    attack = str(summary.get("attack") or "").lower()
    if attack in {"csdj", "cs-dj"}:
        return "CS-DJ"
    if attack == "jood":
        return "JOOD"
    benchmark = "".join(
        char for char in str(summary.get("benchmark") or "").lower() if char.isalnum()
    )
    if benchmark == "jailbreakvmini":
        return "JailbreakV-mini"
    if benchmark == "xstest":
        return "XSTest"
    return None


def _judge_is_gemma(summary: dict[str, Any]) -> bool:
    preset = str(summary.get("judge_preset") or "").lower()
    model = str(summary.get("judge_model") or "").lower()
    return preset == "gemma3_12b" or model == "google/gemma-3-12b-it"


def _matching_summaries(
    runs_root: Path,
    victim_tag: str,
    artifact_fingerprint: str,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    victim_root = runs_root / victim_tag
    if not victim_root.is_dir():
        raise FileNotFoundError(f"Victim run directory does not exist: {victim_root}")
    candidates: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path in victim_root.rglob("summary.json"):
        # Ignore the duplicate summary written inside each judge subdirectory.
        if not path.parent.name.startswith("n_"):
            continue
        try:
            summary = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        detector = summary.get("representation_detector") or {}
        if (
            summary.get("defense") != "cnrf-oracle"
            or detector.get("artifact_fingerprint") != artifact_fingerprint
            or not _judge_is_gemma(summary)
            or not isinstance(summary.get("judge"), dict)
        ):
            continue
        case = _case_name(summary)
        if case:
            candidates.setdefault(case, []).append((path, summary))
    selected: dict[str, tuple[Path, dict[str, Any]]] = {}
    for case, values in candidates.items():
        # Prefer the protocol-complete count, then the newest matching run.
        expected = EXPECTED_COUNTS[case]
        selected[case] = max(
            values,
            key=lambda item: (
                int(item[1].get("n", -1)) == expected,
                int(item[1].get("n", -1)),
                item[0].stat().st_mtime,
            ),
        )
    return selected


def _detector_fpr(summary: dict[str, Any]) -> float | None:
    value = (summary.get("representation_detector") or {}).get("fpr")
    return None if value is None else float(value)


def _row(case: str, path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    judge = summary["judge"]
    representation = summary.get("representation_detector") or {}
    common = {
        "benchmark": case,
        "n": int(summary["n"]),
        "expected_n": EXPECTED_COUNTS[case],
        "protocol_complete": int(summary["n"]) == EXPECTED_COUNTS[case],
        "victim_model": summary.get("model"),
        "judge_model": summary.get("judge_model"),
        "defense_trigger_rate": float(summary.get("defense_triggered_rate", 0.0)),
        "detector_tpr": representation.get("tpr"),
        "detector_fpr": _detector_fpr(summary),
        "judge_parse_error_count": int(judge.get("parse_error_count", 0)),
        "summary_path": str(path),
    }
    if case == "XSTest":
        safe = (judge.get("by_safety_label") or {}).get("safe") or {}
        unsafe = (judge.get("by_safety_label") or {}).get("unsafe") or {}
        return {
            **common,
            "primary_metric": "safe_overrefusal_rate",
            "primary_value": safe.get("safe_overrefusal_rate"),
            "asr": None,
            "safe_n": safe.get("n"),
            "safe_overrefusal_rate": safe.get("safe_overrefusal_rate"),
            "unsafe_n": unsafe.get("n"),
            "unsafe_compliance_rate": unsafe.get("unsafe_compliance_rate"),
        }
    return {
        **common,
        "primary_metric": "asr",
        "primary_value": judge.get("judge_harmful_rate"),
        "asr": judge.get("judge_harmful_rate"),
        "judge_harmful_count": judge.get("judge_harmful_count"),
        "safe_n": None,
        "safe_overrefusal_rate": None,
        "unsafe_n": None,
        "unsafe_compliance_rate": None,
    }


def _percent(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.2f}%"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["benchmark"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    *,
    victim_tag: str,
    detector: RepresentationDetector,
    rows: list[dict[str, Any]],
    missing: list[str],
) -> None:
    lines = [
        "# Unified CNRF Oracle Full Evaluation",
        "",
        f"- Target model: `{victim_tag}` / `{detector.model_id}`",
        "- Judge model: `google/gemma-3-12b-it`",
        f"- Detector fingerprint: `{detector.fingerprint}`",
        "- Oracle policy: one unified cross-benchmark candidate per modality; no per-benchmark oracle",
        "- Claim status: `ORACLE_ONLY` (test/external labels participated in candidate/threshold selection)",
        "",
        "| Benchmark | n | Complete | ASR | XSTest safe over-refusal | Detector trigger | Detector TPR | Detector FPR | Judge parse errors |",
        "| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {benchmark} | {n}/{expected_n} | {complete} | {asr} | {overrefusal} | {trigger} | {tpr} | {fpr} | {errors} |".format(
                benchmark=row["benchmark"],
                n=row["n"],
                expected_n=row["expected_n"],
                complete="yes" if row["protocol_complete"] else "no",
                asr=_percent(row["asr"]),
                overrefusal=_percent(row["safe_overrefusal_rate"]),
                trigger=_percent(row["defense_trigger_rate"]),
                tpr=_percent(row["detector_tpr"]),
                fpr=_percent(row["detector_fpr"]),
                errors=row["judge_parse_error_count"],
            )
        )
    if missing:
        lines.extend(["", f"Missing completed cases: `{', '.join(missing)}`."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize full CNRF Oracle runs judged by Gemma3-12B."
    )
    parser.add_argument("--victim-tag", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("jailbreak_repro/runs"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    artifact_path = args.artifact.expanduser()
    if not artifact_path.is_absolute():
        artifact_path = root / artifact_path
    detector = RepresentationDetector.load(artifact_path.resolve())
    runs_root = args.runs_root.expanduser()
    if not runs_root.is_absolute():
        runs_root = root / runs_root
    selected = _matching_summaries(runs_root.resolve(), args.victim_tag, detector.fingerprint)
    order = ["CS-DJ", "JOOD", "JailbreakV-mini", "XSTest"]
    rows = [_row(case, *selected[case]) for case in order if case in selected]
    missing = [case for case in order if case not in selected]
    incomplete = [row["benchmark"] for row in rows if not row["protocol_complete"]]
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (
            runs_root
            / "cnrf_oracle"
            / args.victim_tag
            / "full_eval"
        )
    elif not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir = out_dir.resolve()
    payload = {
        "format_version": "cnrf_oracle_full_eval_summary_v1",
        "status": "complete" if not missing and not incomplete else "incomplete",
        "oracle_only": True,
        "per_benchmark_oracle": False,
        "victim_tag": args.victim_tag,
        "victim_model": detector.model_id,
        "judge_model": "google/gemma-3-12b-it",
        "artifact": str(detector.path),
        "artifact_fingerprint": detector.fingerprint,
        "missing_cases": missing,
        "incomplete_cases": incomplete,
        "results": rows,
    }
    write_json(out_dir / "summary.json", payload)
    _write_csv(out_dir / "summary.csv", rows)
    _write_markdown(
        out_dir / "summary.md",
        victim_tag=args.victim_tag,
        detector=detector,
        rows=rows,
        missing=missing + incomplete,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.require_complete and (missing or incomplete):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
