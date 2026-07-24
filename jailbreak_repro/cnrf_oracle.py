from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from counterfactual_risk_field.cnrf.oracle import ORACLE_WARNING

from .io_utils import read_json, write_json


FORMAT_VERSION = "jailbreak_repro_cnrf_oracle_summary_v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    parsed = _optional_float(value)
    return None if parsed is None else int(parsed)


def _key(row: dict[str, Any]) -> tuple[str, str, float, str]:
    return (
        str(row["branch"]),
        str(row["policy"]),
        float(row["max_fpr"]),
        str(row["target"]),
    )


def _quality(row: dict[str, Any]) -> tuple[float, ...]:
    budget = _optional_int(row.get("budget"))
    if budget is None:
        budget = _optional_int(row.get("pack_count")) or 10**9
    return (
        float(row["objective_tpr"]),
        float(row["worst_target_tpr"]),
        float(row["macro_target_tpr"]),
        -float(row["worst_empirical_fpr"]),
        -float(budget),
    )


def _best(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    values = list(rows)
    return max(values, key=_quality) if values else None


def _metric(row: dict[str, Any] | None, key: str) -> float | None:
    return None if row is None else _optional_float(row.get(key))


def _compact_result(
    key: tuple[str, str, float, str],
    *,
    full: dict[str, Any] | None,
    best_axis: dict[str, Any] | None,
    best_pack: dict[str, Any] | None,
) -> dict[str, Any]:
    branch, policy, max_fpr, target = key
    full_tpr = _metric(full, "objective_tpr")
    axis_tpr = _metric(best_axis, "objective_tpr")
    pack_tpr = _metric(best_pack, "objective_tpr")
    return {
        "branch": branch,
        "policy": policy,
        "max_fpr": max_fpr,
        "target": target,
        "full_bank_candidate_id": None if full is None else full.get("candidate_id"),
        "full_bank_budget": None if full is None else _optional_int(full.get("budget")),
        "full_bank_tpr": full_tpr,
        "full_bank_macro_tpr": _metric(full, "macro_target_tpr"),
        "full_bank_worst_tpr": _metric(full, "worst_target_tpr"),
        "full_bank_empirical_fpr": _metric(full, "worst_empirical_fpr"),
        "full_bank_test_support": _metric(full, "test_support_coverage"),
        "full_bank_external_support": _metric(full, "external_support_coverage"),
        "best_axis_candidate_id": None if best_axis is None else best_axis.get("candidate_id"),
        "best_axis_axes": None if best_axis is None else best_axis.get("axes"),
        "best_axis_count": None if best_axis is None else _optional_int(best_axis.get("axis_count")),
        "best_axis_tpr": axis_tpr,
        "best_axis_empirical_fpr": _metric(best_axis, "worst_empirical_fpr"),
        "best_axis_test_support": _metric(best_axis, "test_support_coverage"),
        "best_axis_external_support": _metric(best_axis, "external_support_coverage"),
        "best_axis_delta_vs_full": (
            None if full_tpr is None or axis_tpr is None else axis_tpr - full_tpr
        ),
        "best_pack_candidate_id": None if best_pack is None else best_pack.get("candidate_id"),
        "best_pack_budget": None if best_pack is None else _optional_int(best_pack.get("budget")),
        "best_pack_tpr": pack_tpr,
        "best_pack_macro_tpr": _metric(best_pack, "macro_target_tpr"),
        "best_pack_worst_tpr": _metric(best_pack, "worst_target_tpr"),
        "best_pack_empirical_fpr": _metric(best_pack, "worst_empirical_fpr"),
        "best_pack_fpr_ci95_upper": _metric(best_pack, "worst_fpr_ci95_upper"),
        "best_pack_test_support": _metric(best_pack, "test_support_coverage"),
        "best_pack_external_support": _metric(best_pack, "external_support_coverage"),
        "best_pack_delta_vs_full": (
            None if full_tpr is None or pack_tpr is None else pack_tpr - full_tpr
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["branch", "policy", "max_fpr", "target"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _percent(value: Any) -> str:
    parsed = _optional_float(value)
    return "-" if parsed is None else f"{100.0 * parsed:.1f}%"


def _write_markdown(
    path: Path,
    *,
    source: dict[str, Any],
    rows: list[dict[str, Any]],
    primary_policy: str,
    primary_max_fpr: float,
) -> None:
    primary = [
        row
        for row in rows
        if row["policy"] == primary_policy
        and abs(float(row["max_fpr"]) - primary_max_fpr) < 1e-12
    ]
    lines = [
        "# CNRF Oracle Ceiling Results",
        "",
        "## Material Passport",
        "",
        "- Origin: jailbreak_repro CNRF Oracle adapter",
        "- Verification Status: ORACLE_ONLY",
        "- Paper Claim Compatible: false",
        "",
        f"> {ORACLE_WARNING}",
        "",
        f"Primary view: `{primary_policy}` with empirical FPR <= {primary_max_fpr:.3g}.",
        "",
        "| Branch | Target | Full bank TPR | Best axis TPR | Best pack TPR | Pack budget | Empirical FPR | FPR 95% upper | External support |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in primary:
        lines.append(
            "| {branch} | {target} | {full} | {axis} | {pack} | {budget} | {fpr} | {upper} | {support} |".format(
                branch=row["branch"],
                target=row["target"],
                full=_percent(row["full_bank_tpr"]),
                axis=_percent(row["best_axis_tpr"]),
                pack=_percent(row["best_pack_tpr"]),
                budget=row["best_pack_budget"] if row["best_pack_budget"] is not None else "-",
                fpr=_percent(row["best_pack_empirical_fpr"]),
                upper=_percent(row["best_pack_fpr_ci95_upper"]),
                support=_percent(row["best_pack_external_support"]),
            )
        )
    lines.extend(
        [
            "",
            "The axis sweep is exhaustive over the available non-empty axis subsets. The pack search is empirical over LOO-ranked and random candidates, not a proof of the global optimum.",
            "",
            "Support radii are recalibrated for each candidate bank. Improvements therefore combine arrow selection and support-routing changes and require a fixed-support ablation before causal interpretation.",
            "",
            f"Raw CNRF format: `{source.get('format_version')}`; completed axis candidates: {source.get('axis_candidates_completed')}; pack candidates: {source.get('pack_candidates_evaluated')}.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_oracle_run(
    raw_dir: Path,
    output_dir: Path,
    *,
    model_tag: str,
    source_work: Path,
    primary_policy: str = "abstain_safe",
    primary_max_fpr: float = 0.05,
) -> dict[str, Any]:
    raw_dir = raw_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_summary = read_json(raw_dir / "summary.json")
    if not source_summary.get("oracle_only"):
        raise ValueError("CNRF source summary is not marked oracle_only=true")

    axis_rows = _read_csv(raw_dir / "axis_subset_results.csv")
    pack_rows = _read_csv(raw_dir / "pack_budget_oracle.csv")
    keys = sorted({_key(row) for row in axis_rows + pack_rows})
    results: list[dict[str, Any]] = []
    for key in keys:
        matching_axes = [row for row in axis_rows if _key(row) == key]
        matching_packs = [row for row in pack_rows if _key(row) == key]
        pack_budgets = [
            budget
            for row in matching_packs
            if (budget := _optional_int(row.get("budget"))) is not None
        ]
        full_budget = max(pack_budgets) if pack_budgets else None
        full_rows = [
            row
            for row in matching_packs
            if _optional_int(row.get("budget")) == full_budget
        ]
        results.append(
            _compact_result(
                key,
                full=_best(full_rows),
                best_axis=_best(matching_axes),
                best_pack=_best(matching_packs),
            )
        )

    primary_results = [
        row
        for row in results
        if row["policy"] == primary_policy
        and abs(float(row["max_fpr"]) - primary_max_fpr) < 1e-12
    ]
    payload = {
        "format_version": FORMAT_VERSION,
        "method": "cnrf",
        "protocol": "label_leaking_arrow_bank_oracle_ceiling_v1",
        "oracle_only": True,
        "paper_claim_compatible": False,
        "warning": ORACLE_WARNING,
        "selection_leakage": [
            "test/external labels select operating thresholds",
            "test/external labels select the reported best axis subset",
            "test/external labels select the reported best pack subset",
        ],
        "search_scope": {
            "axis_subsets": "exhaustive_non_empty_subsets",
            "pack_subsets": "loo_ranked_plus_random_candidates_not_exhaustive",
            "support_radius": "recalibrated_per_candidate_bank",
        },
        "model_tag": model_tag,
        "source_work": str(source_work.expanduser().resolve()),
        "raw_dir": str(raw_dir),
        "primary_policy": primary_policy,
        "primary_max_fpr": primary_max_fpr,
        "source_run": source_summary,
        "result_count": len(results),
        "primary_results": primary_results,
        "results": results,
    }
    write_json(output_dir / "summary.json", payload)
    _write_csv(output_dir / "summary.csv", results)
    _write_markdown(
        output_dir / "summary.md",
        source=source_summary,
        rows=results,
        primary_policy=primary_policy,
        primary_max_fpr=primary_max_fpr,
    )
    return payload
