from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intentguard.detector import CISRDetector, CISRDetectorBundle
from intentguard.io import read_json, write_json


@dataclass(frozen=True)
class Candidate:
    branch: str
    pooling: str
    summary_path: Path
    detector_path: Path
    summary: dict[str, Any]

    @property
    def validation(self) -> dict[str, Any]:
        return dict(self.summary.get("splits", {}).get("validation", {}))

    def selection_key(self) -> tuple[float, ...]:
        metrics = self.validation
        stability = metrics.get("view_stability", {}).get("mean_probability_std")
        return (
            float(metrics.get("auc") if metrics.get("auc") is not None else -1.0),
            float(
                metrics.get("average_precision")
                if metrics.get("average_precision") is not None
                else -1.0
            ),
            float(
                metrics.get("balanced_accuracy")
                if metrics.get("balanced_accuracy") is not None
                else -1.0
            ),
            -float(stability if stability is not None else 0.0),
        )


def _candidate(values: list[str]) -> Candidate:
    branch, pooling, summary_value, detector_value = values
    if branch not in {"text", "multimodal"}:
        raise ValueError(f"Unsupported CISR_v4 branch: {branch!r}")
    summary_path = Path(summary_value).expanduser().resolve()
    detector_path = Path(detector_value).expanduser().resolve()
    summary = read_json(summary_path)
    detector = CISRDetector.load(detector_path)
    if detector.pooling != pooling:
        raise ValueError(
            f"Candidate pooling mismatch for {branch}: argument={pooling!r}, "
            f"artifact={detector.pooling!r}."
        )
    if not detector.format_version.lower().startswith("cisr_v4_detector"):
        raise ValueError(
            f"Candidate {branch}/{pooling} is not selectively calibrated v4: "
            f"{detector.format_version!r}."
        )
    return Candidate(branch, pooling, summary_path, detector_path, summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select validation-best pooling per modality and package CISR_v4."
    )
    parser.add_argument(
        "--candidate",
        nargs=4,
        action="append",
        metavar=("BRANCH", "POOLING", "V3_SUMMARY", "V4_DETECTOR"),
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-alias", default="")
    args = parser.parse_args()

    candidates = [_candidate(values) for values in args.candidate]
    grouped = {
        branch: [candidate for candidate in candidates if candidate.branch == branch]
        for branch in ("text", "multimodal")
    }
    if any(not values for values in grouped.values()):
        raise ValueError("CISR_v4 bundle requires text and multimodal candidates.")
    selected = {branch: max(values, key=Candidate.selection_key) for branch, values in grouped.items()}

    loaded = [CISRDetector.load(candidate.detector_path) for candidate in selected.values()]
    model_ids = {detector.model_id for detector in loaded if detector.model_id}
    if len(model_ids) > 1:
        raise ValueError(f"Selected CISR_v4 branches use different models: {sorted(model_ids)}")
    model_id = next(iter(model_ids)) if model_ids else ""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    branches: dict[str, Any] = {}
    for branch in ("text", "multimodal"):
        candidate = selected[branch]
        destination = args.out_dir / f"{branch}_detector.npz"
        shutil.copy2(candidate.detector_path, destination)
        branches[branch] = {
            "detector": destination.name,
            "pooling": candidate.pooling,
            "selected_layer": int(CISRDetector.load(destination).layer),
            "source_detector": str(candidate.detector_path),
            "source_summary": str(candidate.summary_path),
            "detector_sha1": hashlib.sha1(destination.read_bytes()).hexdigest(),
            "validation_selection_key": list(candidate.selection_key()),
        }

    manifest = {
        "format_version": "CISR_v4_detector_bundle_v1",
        "model_id": model_id,
        "model_alias": args.model_alias,
        "routing_rule": "multimodal iff image_path is present; otherwise text",
        "selection_split": "validation",
        "selection_metric_order": [
            "auc",
            "average_precision",
            "balanced_accuracy",
            "negative_mean_view_probability_std",
        ],
        "branches": branches,
        "candidates": [
            {
                "branch": candidate.branch,
                "pooling": candidate.pooling,
                "summary": str(candidate.summary_path),
                "detector": str(candidate.detector_path),
                "validation": candidate.validation,
                "selection_key": list(candidate.selection_key()),
                "selected": selected[candidate.branch] == candidate,
            }
            for candidate in candidates
        ],
    }
    manifest_path = args.out_dir / "detector_bundle.json"
    write_json(manifest_path, manifest)
    bundle = CISRDetectorBundle.load(manifest_path)
    if bundle.model_id != model_id:
        raise RuntimeError("CISR_v4 bundle failed round-trip validation.")
    print(
        f"Wrote {manifest_path}; text={selected['text'].pooling}, "
        f"multimodal={selected['multimodal'].pooling}"
    )


if __name__ == "__main__":
    main()
