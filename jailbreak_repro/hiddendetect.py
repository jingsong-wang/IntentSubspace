from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .io_utils import read_json, repo_root, slugify, write_json
from .models import BaseModelRunner, RefusalLayerScores

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable, **_: Any):
        return iterable


HIDDENDETECT_PROFILE_FORMAT = "hiddendetect_profile_v2"
HIDDENDETECT_SCORE_FORMAT = "hiddendetect_refusal_trapz_or_point_v2"
HIDDENDETECT_PROTOCOL = "official_12shot_fdv_cross_model_adaptation_v2"
HIDDENDETECT_SELECTION_RULE = "fdv_above_final_contiguous_hull_v2"
HIDDENDETECT_AGGREGATION_RULE = "unit_trapezoid_or_singleton_point_v2"
HIDDENDETECT_REFUSAL_TOKENS = [
    "Sorry",
    "sorry",
    "unfortunately",
    "unfortunate",
    "sadly",
    "explicit",
    "deadly",
    "crim",
    "criminal",
    "illegal",
    "dangerous",
    "harmful",
    "warning",
    "alarm",
    "caution",
    "shame",
    "conspiracy",
    "Subject",
    "contrary",
    "shouldn",
]

HIDDENDETECT_HARD_REFUSAL = (
    "I cannot assist with harmful, illegal, or dangerous instructions. "
    "I can help with safety, prevention, or other non-operational information."
)


def default_hiddendetect_source_dir() -> Path:
    return repo_root() / "jailbreak_repro" / "sourcecode" / "HiddenDetect-main"


def default_hiddendetect_profile_path(model_name: str) -> Path:
    return repo_root() / "runs" / "HiddenDetect" / slugify(model_name) / "profile_v2.json"


def trapezoid_score(values: list[float]) -> float:
    """Match the released ``np.trapz`` aggregation exactly."""
    if not values:
        raise ValueError("HiddenDetect requires at least one safety-aware layer")
    if len(values) == 1:
        return 0.0
    return float(sum((left + right) * 0.5 for left, right in zip(values, values[1:])))


def hiddendetect_score(values: list[float]) -> float:
    """Aggregate a valid interval without collapsing a singleton to zero."""
    if not values:
        raise ValueError("HiddenDetect requires at least one safety-aware layer")
    if len(values) == 1:
        return float(values[0])
    return trapezoid_score(values)


def safety_aware_layer_candidates(fdv: list[float]) -> list[int]:
    if not fdv:
        raise ValueError("HiddenDetect FDV is empty")
    baseline = fdv[-1]
    return [index for index, value in enumerate(fdv) if value > baseline]


def select_safety_aware_layers(fdv: list[float]) -> list[int]:
    candidates = safety_aware_layer_candidates(fdv)
    if not candidates:
        raise ValueError(
            "HiddenDetect found no layers with FDV greater than the final-layer baseline; "
            "the supplied few-shot set does not identify a safety-aware range for this victim."
        )
    # The released scorer slices F[s:e+1], so its integration domain is a
    # contiguous interval even when the diagnostic FDV hits are sparse.
    return list(range(candidates[0], candidates[-1] + 1))


def _balanced_threshold(labels: list[int], scores: list[float]) -> tuple[float, dict[str, float]]:
    if len(labels) != len(scores) or not labels:
        raise ValueError("HiddenDetect threshold calibration requires aligned labels and scores")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("HiddenDetect threshold calibration requires both safe and unsafe examples")
    unique = sorted(set(float(score) for score in scores))
    epsilon = max(1e-12, (unique[-1] - unique[0]) * 1e-9)
    candidates = [unique[0] - epsilon]
    candidates.extend((left + right) * 0.5 for left, right in zip(unique, unique[1:]))
    candidates.append(unique[-1] + epsilon)

    best = None
    for threshold in candidates:
        tp = sum(1 for label, score in zip(labels, scores) if label == 1 and score > threshold)
        fp = sum(1 for label, score in zip(labels, scores) if label == 0 and score > threshold)
        tpr = tp / positives
        fpr = fp / negatives
        balanced_accuracy = 0.5 * (tpr + (1.0 - fpr))
        key = (balanced_accuracy, -fpr, tpr, threshold)
        if best is None or key > best[0]:
            best = (key, threshold, {"balanced_accuracy": balanced_accuracy, "tpr": tpr, "fpr": fpr})
    assert best is not None
    return float(best[1]), best[2]


def _fewshot_path(source_dir: Path, fewshot_file: Path | None) -> Path:
    path = fewshot_file or source_dir / "data" / "few_shot" / "few_shot.json"
    path = path.expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HiddenDetect official few-shot file does not exist: {path}")
    return path


def load_hiddendetect_fewshot(source_dir: Path, fewshot_file: Path | None = None) -> tuple[list[dict[str, Any]], str]:
    path = _fewshot_path(source_dir, fewshot_file)
    raw = read_json(path)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"HiddenDetect few-shot file must contain a non-empty list: {path}")
    samples = []
    digest = hashlib.sha1(path.read_bytes())
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"HiddenDetect few-shot entry {index} is not an object")
        prompt = str(item.get("txt") or "").strip()
        label = int(item.get("toxicity", -1))
        if not prompt or label not in {0, 1}:
            raise ValueError(f"HiddenDetect few-shot entry {index} has invalid txt/toxicity")
        image_value = item.get("img")
        image_path = None
        if image_value not in {None, "null", ""}:
            candidate = Path(str(image_value)).expanduser()
            if not candidate.is_absolute():
                candidate = source_dir / candidate
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"HiddenDetect few-shot image does not exist: {candidate}")
            digest.update(candidate.read_bytes())
            image_path = str(candidate)
        samples.append({"id": f"fewshot-{index}", "prompt": prompt, "image_path": image_path, "label": label})
    if {sample["label"] for sample in samples} != {0, 1}:
        raise ValueError("HiddenDetect few-shot set must contain both safe and unsafe examples")
    return samples, digest.hexdigest()


@dataclass
class HiddenDetectProfile:
    format_version: str
    score_format: str
    protocol: str
    model_id: str
    backend: str
    refusal_tokens: list[str]
    refusal_token_ids: list[int]
    layer_count: int
    selection_rule: str
    selection_candidates: list[int]
    safety_aware_layers: list[int]
    aggregation_rule: str
    paper_score_compatible: bool
    safe_mean: list[float]
    unsafe_mean: list[float]
    fdv: list[float]
    threshold: float
    threshold_method: str
    threshold_metrics: dict[str, float]
    calibration_safe_scores: list[float]
    calibration_unsafe_scores: list[float]
    fewshot_sha1: str
    profile_fingerprint: str
    model_revision: str | None = None

    @classmethod
    def load(cls, path: Path) -> "HiddenDetectProfile":
        payload = read_json(path)
        if payload.get("format_version") != HIDDENDETECT_PROFILE_FORMAT:
            raise ValueError(
                f"Unsupported HiddenDetect profile format {payload.get('format_version')!r}; "
                f"expected {HIDDENDETECT_PROFILE_FORMAT!r}."
            )
        profile = cls(**payload)
        profile.validate()
        return profile

    def save(self, path: Path) -> None:
        self.validate()
        write_json(path, asdict(self))

    def validate(self) -> None:
        if self.score_format != HIDDENDETECT_SCORE_FORMAT:
            raise ValueError(f"Unsupported HiddenDetect score format: {self.score_format!r}")
        if self.protocol != HIDDENDETECT_PROTOCOL:
            raise ValueError(f"Unsupported HiddenDetect protocol: {self.protocol!r}")
        if self.selection_rule != HIDDENDETECT_SELECTION_RULE:
            raise ValueError(f"Unsupported HiddenDetect selection rule: {self.selection_rule!r}")
        if self.aggregation_rule != HIDDENDETECT_AGGREGATION_RULE:
            raise ValueError(f"Unsupported HiddenDetect aggregation rule: {self.aggregation_rule!r}")
        expected = list(range(self.safety_aware_layers[0], self.safety_aware_layers[-1] + 1)) \
            if self.safety_aware_layers else []
        if self.safety_aware_layers != expected:
            raise ValueError("HiddenDetect safety-aware layers must form a non-empty contiguous interval")
        if any(index < 0 or index >= self.layer_count for index in self.safety_aware_layers):
            raise ValueError("HiddenDetect safety-aware layer is outside the victim layer range")
        if not self.selection_candidates or any(
            index not in self.safety_aware_layers for index in self.selection_candidates
        ):
            raise ValueError("HiddenDetect FDV candidates must be contained in the selected interval")
        if len(self.safe_mean) != self.layer_count or len(self.unsafe_mean) != self.layer_count:
            raise ValueError("HiddenDetect calibration means do not match the victim layer count")
        if len(self.fdv) != self.layer_count:
            raise ValueError("HiddenDetect FDV does not match the victim layer count")
        calibration_scores = self.calibration_safe_scores + self.calibration_unsafe_scores
        if not calibration_scores or any(not math.isfinite(float(score)) for score in calibration_scores):
            raise ValueError("HiddenDetect calibration scores must be non-empty and finite")
        if not math.isfinite(float(self.threshold)):
            raise ValueError("HiddenDetect threshold must be finite")


def _mean_rows(rows: list[list[float]]) -> list[float]:
    if not rows:
        raise ValueError("Cannot average an empty HiddenDetect calibration group")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("HiddenDetect calibration rows expose inconsistent layer counts")
    return [sum(row[index] for row in rows) / len(rows) for index in range(width)]


def build_hiddendetect_profile(
    runner: BaseModelRunner,
    output_path: Path,
    source_dir: Path | None = None,
    fewshot_file: Path | None = None,
) -> HiddenDetectProfile:
    source = (source_dir or default_hiddendetect_source_dir()).expanduser().resolve()
    fewshot, fewshot_sha1 = load_hiddendetect_fewshot(source, fewshot_file)
    safe_rows: list[list[float]] = []
    unsafe_rows: list[list[float]] = []
    token_ids: list[int] | None = None
    backend = ""
    for sample in tqdm(fewshot, desc="HiddenDetect victim calibration"):
        result = runner.score_refusal_layers(
            sample["prompt"],
            refusal_tokens=HIDDENDETECT_REFUSAL_TOKENS,
            image_path=sample["image_path"],
        )
        if token_ids is None:
            token_ids = result.refusal_token_ids
            backend = result.backend
        elif token_ids != result.refusal_token_ids:
            raise RuntimeError("HiddenDetect tokenizer ids changed during calibration")
        (unsafe_rows if sample["label"] == 1 else safe_rows).append(result.scores)

    safe_mean = _mean_rows(safe_rows)
    unsafe_mean = _mean_rows(unsafe_rows)
    fdv = [unsafe - safe for unsafe, safe in zip(unsafe_mean, safe_mean)]
    candidates = safety_aware_layer_candidates(fdv)
    layers = select_safety_aware_layers(fdv)
    safe_scores = [hiddendetect_score([row[index] for index in layers]) for row in safe_rows]
    unsafe_scores = [hiddendetect_score([row[index] for index in layers]) for row in unsafe_rows]
    labels = [0] * len(safe_scores) + [1] * len(unsafe_scores)
    calibration_scores = safe_scores + unsafe_scores
    if len(set(calibration_scores)) < 2:
        raise RuntimeError(
            "HiddenDetect calibration produced a constant score. Refusing to write a degenerate detector profile."
        )
    threshold, threshold_metrics = _balanced_threshold(labels, calibration_scores)
    threshold_metrics.update(
        {
            "auprc": float(binary_auprc(labels, calibration_scores) or 0.0),
            "auroc": float(binary_auroc(labels, calibration_scores) or 0.0),
            "unique_score_count": float(len(set(calibration_scores))),
        }
    )
    core = {
        "format_version": HIDDENDETECT_PROFILE_FORMAT,
        "score_format": HIDDENDETECT_SCORE_FORMAT,
        "protocol": HIDDENDETECT_PROTOCOL,
        "model_id": runner.model_name,
        "model_revision": getattr(runner, "model_revision", None),
        "backend": backend or runner.backend,
        "refusal_tokens": HIDDENDETECT_REFUSAL_TOKENS,
        "refusal_token_ids": token_ids or [],
        "layer_count": len(fdv),
        "selection_rule": HIDDENDETECT_SELECTION_RULE,
        "selection_candidates": candidates,
        "safety_aware_layers": layers,
        "aggregation_rule": HIDDENDETECT_AGGREGATION_RULE,
        "paper_score_compatible": len(layers) >= 2,
        "safe_mean": safe_mean,
        "unsafe_mean": unsafe_mean,
        "fdv": fdv,
        "threshold": threshold,
        "threshold_method": "balanced_accuracy_on_official_12shot_platform_calibration_v2",
        "threshold_metrics": threshold_metrics,
        "calibration_safe_scores": safe_scores,
        "calibration_unsafe_scores": unsafe_scores,
        "fewshot_sha1": fewshot_sha1,
    }
    fingerprint = hashlib.sha1(
        json.dumps(core, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    profile = HiddenDetectProfile(**core, profile_fingerprint=fingerprint)
    output = output_path.expanduser()
    if not output.is_absolute():
        output = repo_root() / output
    profile.save(output.resolve())
    return profile


def ensure_hiddendetect_profile(
    runner: BaseModelRunner,
    profile_path: Path,
    source_dir: Path | None = None,
    fewshot_file: Path | None = None,
    allow_model_mismatch: bool = False,
    model_revision: str | None = None,
) -> HiddenDetectProfile:
    path = profile_path.expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    path = path.resolve()
    if path.is_file():
        profile = HiddenDetectProfile.load(path)
        print(f"Reusing HiddenDetect victim profile: {path}")
    else:
        print(f"Building HiddenDetect victim profile at {path}")
        profile = build_hiddendetect_profile(runner, path, source_dir=source_dir, fewshot_file=fewshot_file)
    if profile.model_id != runner.model_name and not allow_model_mismatch:
        raise ValueError(
            f"HiddenDetect profile was built for {profile.model_id!r}, but victim is {runner.model_name!r}. "
            "Build a victim-specific profile or pass --hiddendetect-allow-model-mismatch for a transfer ablation."
        )
    if profile.model_revision != model_revision and not allow_model_mismatch:
        raise ValueError(
            f"HiddenDetect profile revision is {profile.model_revision!r}, but victim revision is {model_revision!r}. "
            "Build a profile for the selected revision."
        )
    source = (source_dir or default_hiddendetect_source_dir()).expanduser().resolve()
    _, current_fewshot_sha1 = load_hiddendetect_fewshot(source, fewshot_file)
    if profile.fewshot_sha1 != current_fewshot_sha1:
        if profile.model_id != runner.model_name or profile.model_revision != model_revision:
            raise ValueError(
                "HiddenDetect profile uses a different few-shot calibration set and cannot be rebuilt "
                "while model-mismatch transfer mode is active."
            )
        print("HiddenDetect few-shot fingerprint changed; rebuilding the victim profile.")
        profile = build_hiddendetect_profile(
            runner,
            path,
            source_dir=source,
            fewshot_file=fewshot_file,
        )
    if profile.refusal_tokens != HIDDENDETECT_REFUSAL_TOKENS:
        raise ValueError("HiddenDetect profile does not use the paper's refusal token set")
    return profile


def score_hiddendetect_sample(
    runner: BaseModelRunner,
    profile: HiddenDetectProfile,
    prompt: str,
    image_path: str | None,
    threshold_override: float | None = None,
) -> dict[str, Any]:
    result: RefusalLayerScores = runner.score_refusal_layers(
        prompt,
        refusal_tokens=profile.refusal_tokens,
        image_path=image_path,
    )
    if len(result.scores) != profile.layer_count:
        raise ValueError(
            f"HiddenDetect profile expects {profile.layer_count} language layers, "
            f"but victim exposed {len(result.scores)}"
        )
    if result.refusal_token_ids != profile.refusal_token_ids:
        raise ValueError("HiddenDetect profile refusal token ids do not match the current victim tokenizer")
    selected = [result.scores[index] for index in profile.safety_aware_layers]
    score = hiddendetect_score(selected)
    threshold = profile.threshold if threshold_override is None else float(threshold_override)
    return {
        "score": score,
        "threshold": threshold,
        "detected": score > threshold,
        "layer_scores": result.scores,
        "selected_scores": selected,
        "selection_rule": profile.selection_rule,
        "aggregation_rule": profile.aggregation_rule,
        "singleton_point_fallback": len(selected) == 1,
        "rendered_prompt": result.rendered_prompt,
        "backend": result.backend,
        "metadata": result.metadata,
    }


def binary_auroc(labels: list[int], scores: list[float]) -> float | None:
    if len(labels) != len(scores) or not labels or set(labels) != {0, 1}:
        return None
    indexed = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = ((start + 1) + end) * 0.5
        for offset in range(start, end):
            ranks[indexed[offset][0]] = average_rank
        start = end
    positives = sum(labels)
    negatives = len(labels) - positives
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) * 0.5) / (positives * negatives)


def binary_auprc(labels: list[int], scores: list[float]) -> float | None:
    """Match precision_recall_curve followed by trapezoidal AUC."""
    if len(labels) != len(scores) or not labels:
        return None
    positives = sum(int(label) == 1 for label in labels)
    if positives == 0 or positives == len(labels):
        return None

    grouped: dict[float, list[int]] = {}
    for label, score in zip(labels, scores):
        grouped.setdefault(float(score), []).append(int(label))

    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    previous_precision = 1.0
    area = 0.0
    for score in sorted(grouped, reverse=True):
        group = grouped[score]
        true_positives += sum(label == 1 for label in group)
        false_positives += sum(label == 0 for label in group)
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * (precision + previous_precision) * 0.5
        previous_recall = recall
        previous_precision = precision
    return float(area)


def sample_safety_label(sample: dict[str, Any]) -> int | None:
    xstest = str(sample.get("xstest_label") or "").lower()
    if xstest in {"safe", "unsafe"}:
        return 1 if xstest == "unsafe" else 0
    for key in ("toxicity", "label"):
        value = sample.get(key)
        if value in {0, 1, "0", "1"}:
            return int(value)
    return None
