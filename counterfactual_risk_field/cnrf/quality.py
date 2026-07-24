from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from .generation import COUNTERFACTUAL_AXES, is_image_dependent_request


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)
_STOPWORDS = set(
    "a an the to of and or in on for with into from by is are be being been that this these "
    "those it its as at about how what why when where who provide give write create describe "
    "explain tell me you your can could would should request tutorial guide instructions methods "
    "ways use using used".split()
)
_SAFE_PIVOTS = set(
    "safe safely safety prevent preventing prevention detect detecting detection report reporting "
    "recover recovery protect protection protective secure security ethical ethically authorized "
    "authorization legal legally defensive defense defend mitigate mitigation respond response "
    "responsible responsibly legitimate permission consent consensual compliance educational "
    "education high level awareness identify identifying avoid avoiding harmless non harmful "
    "training simulation recognize recognition monitor monitoring remediation support".split()
)
_GENERIC_TASK_WORDS = set(
    "script system systems method methods tutorial guide instructions information data process "
    "steps step technique techniques way ways".split()
)
_REFUSAL_PATTERN = re.compile(
    r"\b(?:cannot|can't|won't|refuse|unable)\b|\bi am sorry\b|\bi'm sorry\b",
    flags=re.IGNORECASE,
)
_ANSWERLIKE_PATTERN = re.compile(
    r"\b(?:first|second|finally)\b|\bstep\s*\d|^here (?:are|is)\b",
    flags=re.IGNORECASE,
)
_TEMPLATE_MARKER_PATTERN = re.compile(
    r"prevent|protect|avoid|mitigat|secure|detect|identify|recogniz|report|monitor|"
    r"warning sign|recover|remediat|respond|response|authoriz|permission|consent|ethical|"
    r"legal|legitimate|lawful|education|educational|awareness|high[- ]level|training|"
    r"simulation|theoretical|historical|\bsafe\b|\bsafely\b|\bsafety\b|harmless|"
    r"defensive|defense",
    flags=re.IGNORECASE,
)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(str(text).lower()))


def _topic_tokens(text: str) -> set[str]:
    return _tokens(text) - _STOPWORDS - _SAFE_PIVOTS - _GENERIC_TASK_WORDS


def topic_token_coverage(harmful: str, benign: str) -> float:
    """Lexical proxy for how much harmful-side topic content remains in the benign side."""

    source = _topic_tokens(harmful)
    if not source:
        return 1.0
    return len(source & _topic_tokens(benign)) / len(source)


def token_jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def _word_count(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(str(text)))


def candidate_quality(
    candidate: dict[str, Any],
    *,
    min_topic_coverage: float = 0.15,
    min_length_ratio: float = 0.4,
    max_length_ratio: float = 2.5,
    reject_image_semantic_mismatch: bool = True,
    require_known_axis: bool = True,
) -> dict[str, Any]:
    harmful = str(candidate.get("harmful_request") or "").strip()
    benign = str(candidate.get("benign_counterfactual") or "").strip()
    semantic = candidate.get("semantic_seed")
    semantic = semantic if isinstance(semantic, dict) else {}
    seed_request = str(semantic.get("prompt_text") or semantic.get("semantic_text") or "").strip()
    carrier_kind = str(candidate.get("carrier_kind") or "")
    axis = str(candidate.get("counterfactual_axis") or "legacy_unspecified")
    image_dependent = bool(
        candidate.get("image_dependent_seed", is_image_dependent_request(seed_request))
    )
    image_semantics_valid = not image_dependent or carrier_kind == "native_image"

    harmful_words = max(1, _word_count(harmful))
    length_ratio = _word_count(benign) / harmful_words
    coverage = topic_token_coverage(harmful, benign)
    jaccard = token_jaccard(harmful, benign)
    reject_reasons: list[str] = []
    warnings: list[str] = []

    if not harmful or not benign:
        reject_reasons.append("empty_endpoint")
    if harmful == benign:
        reject_reasons.append("identical_endpoints")
    if reject_image_semantic_mismatch and not image_semantics_valid:
        reject_reasons.append("image_dependent_seed_without_native_image")
    if coverage < float(min_topic_coverage):
        reject_reasons.append("topic_coverage_below_threshold")
    if not (float(min_length_ratio) <= length_ratio <= float(max_length_ratio)):
        reject_reasons.append("length_ratio_out_of_range")
    if require_known_axis and axis not in COUNTERFACTUAL_AXES:
        reject_reasons.append("unknown_or_legacy_counterfactual_axis")
    if _REFUSAL_PATTERN.search(benign):
        reject_reasons.append("generic_refusal_like_benign")
    if _ANSWERLIKE_PATTERN.search(benign):
        reject_reasons.append("answerlike_benign")

    if seed_request and harmful != seed_request:
        warnings.append("materialized_harmful_differs_from_seed")
    if candidate.get("harmful_seed_preserved_by_model") is False:
        warnings.append("model_did_not_copy_harmful_seed")
    if candidate.get("axis_matches_preference") is False:
        warnings.append("model_used_fallback_axis")
    if _TEMPLATE_MARKER_PATTERN.search(benign):
        warnings.append("explicit_safety_template_marker")

    return {
        "status": "reject" if reject_reasons else ("warn" if warnings else "pass"),
        "reject_reasons": reject_reasons,
        "warnings": warnings,
        "metrics": {
            "topic_token_coverage": coverage,
            "token_jaccard": jaccard,
            "length_ratio": length_ratio,
            "harmful_word_count": harmful_words,
            "benign_word_count": _word_count(benign),
            "image_dependent_seed": image_dependent,
            "image_semantics_valid": image_semantics_valid,
            "counterfactual_axis": axis,
        },
    }


def audit_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    min_topic_coverage: float = 0.15,
    min_length_ratio: float = 0.4,
    max_length_ratio: float = 2.5,
    reject_image_semantic_mismatch: bool = True,
    require_known_axis: bool = True,
    deduplicate: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    reason_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    axis_counts: Counter[str] = Counter()

    for raw in candidates:
        candidate = dict(raw)
        audit = candidate_quality(
            candidate,
            min_topic_coverage=min_topic_coverage,
            min_length_ratio=min_length_ratio,
            max_length_ratio=max_length_ratio,
            reject_image_semantic_mismatch=reject_image_semantic_mismatch,
            require_known_axis=require_known_axis,
        )
        signature = (
            str(candidate.get("pack_id") or ""),
            str(candidate.get("carrier_kind") or ""),
            str(candidate.get("carrier_id") or ""),
            " ".join(str(candidate.get("benign_counterfactual") or "").lower().split()),
        )
        if deduplicate and signature in seen:
            audit["reject_reasons"].append("duplicate_benign_within_pack_carrier")
            audit["status"] = "reject"
        seen.add(signature)
        candidate["automatic_quality_audit"] = audit
        axis_counts[str(audit["metrics"]["counterfactual_axis"])] += 1
        warning_counts.update(audit["warnings"])
        if audit["reject_reasons"]:
            reason_counts.update(audit["reject_reasons"])
            rejected.append(candidate)
        else:
            accepted.append(candidate)

    total = len(accepted) + len(rejected)
    report = {
        "format_version": "cnrf_counterfactual_quality_audit_v1",
        "candidates": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / total if total else None,
        "parameters": {
            "min_topic_coverage": min_topic_coverage,
            "min_length_ratio": min_length_ratio,
            "max_length_ratio": max_length_ratio,
            "reject_image_semantic_mismatch": reject_image_semantic_mismatch,
            "require_known_axis": require_known_axis,
            "deduplicate": deduplicate,
        },
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "warnings": dict(sorted(warning_counts.items())),
        "axes": dict(sorted(axis_counts.items())),
        "formal_warning": (
            "Automatic quality checks do not replace the five-field human approval gate."
        ),
    }
    return accepted, rejected, report
