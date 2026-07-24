from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from counterfactual_risk_field.cnrf.activations import load_activation_data
from counterfactual_risk_field.cnrf.io import (
    canonical_json,
    read_jsonl,
    stable_id,
    write_json,
)
from counterfactual_risk_field.cnrf.oracle import (
    ORACLE_WARNING,
    OracleAnalyzer,
    enumerate_axis_subsets,
    fixed_views_from_summary,
)


def _comma_ints(value: str | None, default: Iterable[int]) -> list[int]:
    if not value:
        return [int(item) for item in default]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _comma_floats(value: str | None, default: Iterable[float]) -> list[float]:
    if not value:
        return [float(item) for item in default]
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary = path.with_name(path.name + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.temporary.open("w", encoding="utf-8", newline="\n")

    def write(self, value: dict[str, Any]) -> None:
        self.handle.write(canonical_json(value) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()
        os.replace(self.temporary, self.path)

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.handle.close()
        if exc_type is None:
            os.replace(self.temporary, self.path)


def _oracle_rows(
    result: dict[str, Any],
    *,
    candidate_id: str,
    candidate_kind: str,
    budget: int | str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, policy_report in result["oracle"].items():
        for max_fpr, targets in policy_report.items():
            for target, point in targets.items():
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_kind": candidate_kind,
                        "branch": result["branch"],
                        "budget": budget,
                        "axes": ",".join(result["axes"]),
                        "axis_count": len(result["axes"]),
                        "pack_count": result["selected_pack_count"],
                        "policy": policy,
                        "max_fpr": float(max_fpr),
                        "target": target,
                        "objective_tpr": point["objective_tpr"],
                        "macro_target_tpr": point["macro_target_tpr"],
                        "worst_target_tpr": point["worst_target_tpr"],
                        "worst_empirical_fpr": point["worst_empirical_fpr"],
                        "worst_fpr_ci95_upper": point["worst_fpr_ci95_upper"],
                        "oracle_threshold": point["threshold"],
                        "calibration_threshold": result["threshold"],
                        "test_support_coverage": result["splits"]
                        .get("test", {})
                        .get("support_coverage"),
                        "external_support_coverage": result["splits"]
                        .get("external", {})
                        .get("support_coverage"),
                    }
                )
    return rows


def _point_key(point: dict[str, Any], pack_count: int) -> tuple[float, ...]:
    return (
        float(point["objective_tpr"]),
        float(point["worst_target_tpr"]),
        float(point["macro_target_tpr"]),
        -float(point["worst_empirical_fpr"]),
        -float(pack_count),
    )


def _candidate_metrics(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for policy, policy_report in result["oracle"].items():
        for max_fpr, targets in policy_report.items():
            for target, point in targets.items():
                output[f"{policy}|{max_fpr}|{target}"] = {
                    "objective_tpr": point["objective_tpr"],
                    "worst_target_tpr": point["worst_target_tpr"],
                    "macro_target_tpr": point["macro_target_tpr"],
                    "worst_empirical_fpr": point["worst_empirical_fpr"],
                    "threshold": point["threshold"],
                }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fixed-view, label-leaking CNRF arrow-bank ceiling diagnostics."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--oracle-config",
        type=Path,
        default=Path("counterfactual_risk_field/configs/oracle_v2.json"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pack-budgets")
    parser.add_argument("--max-fprs")
    parser.add_argument("--random-subsets", type=int)
    parser.add_argument(
        "--pack-policy", choices=["abstain_safe", "abstain_risk"]
    )
    parser.add_argument("--pack-max-fpr", type=float)
    parser.add_argument(
        "--skip-axis-subsets", action="store_true", help="Only run pack-bank search."
    )
    parser.add_argument(
        "--skip-pack-search", action="store_true", help="Only run axis subsets."
    )
    parser.add_argument(
        "--max-axis-subsets",
        type=int,
        help="Development-only cap; omit for the complete 2^A-1 sweep.",
    )
    args = parser.parse_args()

    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    oracle_config = json.loads(args.oracle_config.read_text(encoding="utf-8"))
    source_summary = json.loads(args.summary.read_text(encoding="utf-8"))
    fixed_views = fixed_views_from_summary(source_summary)
    selected_readouts = sorted(
        set(view.readout for views in fixed_views.values() for view in views)
    )
    selected_layers = sorted(
        set(view.layer for views in fixed_views.values() for view in views)
    )
    budgets = _comma_ints(
        args.pack_budgets,
        oracle_config.get("pack_budgets", [25, 50, 100, 200]),
    )
    max_fprs = _comma_floats(
        args.max_fprs,
        oracle_config.get("max_fprs", [0.01, 0.05]),
    )
    random_subsets = int(
        args.random_subsets
        if args.random_subsets is not None
        else oracle_config.get("random_subsets", 32)
    )
    pack_policy = str(
        args.pack_policy or oracle_config.get("pack_policy", "abstain_safe")
    )
    pack_max_fpr = float(
        args.pack_max_fpr
        if args.pack_max_fpr is not None
        else oracle_config.get("pack_max_fpr", 0.05)
    )
    max_fprs = sorted(set(max_fprs + [pack_max_fpr]))
    seed = int(oracle_config.get("seed", protocol.get("seed", 20260721)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.out_dir / "run.json"
    started = time.time()

    def progress(stage: str, **extra: Any) -> None:
        report = {
            "format_version": "cnrf_oracle_run_v1",
            "oracle_only": True,
            "warning": ORACLE_WARNING,
            "status": "running",
            "stage": stage,
            "elapsed_seconds": time.time() - started,
            **extra,
        }
        write_json(run_path, report)
        print(canonical_json(report), flush=True)

    progress(
        "load_activations",
        selected_readouts=selected_readouts,
        selected_layers=selected_layers,
    )
    rows = list(read_jsonl(args.manifest))
    activations = load_activation_data(
        args.activations,
        selected_readouts=selected_readouts,
        selected_layers=selected_layers,
    )
    risk = protocol["risk_field"]
    analyzer = OracleAnalyzer(
        rows,
        activations,
        fixed_views,
        k=int(risk["k"]),
        alpha=float(protocol["calibration"]["primary_alpha"]),
        support_quantile_value=float(risk["support_quantile"]),
        min_arrow_norm=float(risk["min_arrow_norm"]),
        score_clip=float(risk["score_clip"]),
        fusion_policy=str(risk.get("fusion_policy", "supported_max")),
    )

    axis_csv_rows: list[dict[str, Any]] = []
    axis_count = 0
    axis_failures: list[dict[str, Any]] = []
    full_prepared: dict[str, Any] = {}
    full_results: dict[str, dict[str, Any]] = {}
    if not args.skip_axis_subsets:
        with JsonlWriter(args.out_dir / "axis_subset_results.jsonl") as writer:
            for branch in analyzer.branches():
                branch_axes = analyzer.axes(branch)
                combinations = list(enumerate_axis_subsets(branch_axes))
                if args.max_axis_subsets is not None:
                    combinations = combinations[: args.max_axis_subsets]
                    if tuple(branch_axes) not in combinations:
                        combinations.append(tuple(branch_axes))
                for axes in combinations:
                    candidate_id = f"axis:{branch}:{'+'.join(axes)}"
                    try:
                        prepared = analyzer.prepare_branch(branch, axes)
                        result = analyzer.evaluate_prepared(
                            prepared,
                            max_fprs=max_fprs,
                        )
                    except ValueError as exc:
                        failure = {
                            "candidate_id": candidate_id,
                            "branch": branch,
                            "axes": list(axes),
                            "status": "invalid_candidate",
                            "error": str(exc),
                        }
                        writer.write(failure)
                        axis_failures.append(failure)
                        continue
                    writer.write({"candidate_id": candidate_id, **result})
                    axis_csv_rows.extend(
                        _oracle_rows(
                            result,
                            candidate_id=candidate_id,
                            candidate_kind="axis_subset",
                            budget="axis_subset",
                        )
                    )
                    axis_count += 1
                    if set(axes) == set(branch_axes):
                        full_prepared[branch] = prepared
                        full_results[branch] = result
                    progress(
                        "axis_subsets",
                        completed=axis_count,
                        failures=len(axis_failures),
                        branch=branch,
                        axes=list(axes),
                    )
        _write_csv(args.out_dir / "axis_subset_results.csv", axis_csv_rows)

    # A capped axis smoke still needs the full all-axis bank for pack analysis.
    for branch in analyzer.branches():
        if branch not in full_prepared:
            full_prepared[branch] = analyzer.prepare_branch(branch, analyzer.axes(branch))
            full_results[branch] = analyzer.evaluate_prepared(
                full_prepared[branch], max_fprs=max_fprs
            )

    progress("sample_pack_influence")
    with JsonlWriter(args.out_dir / "sample_pack_influence.jsonl") as writer:
        influence_rows = 0
        for branch in analyzer.branches():
            for row in analyzer.sample_pack_influence(
                full_prepared[branch], full_results[branch]
            ):
                writer.write(row)
                influence_rows += 1

    pack_influence_count = 0
    pack_candidate_count = 0
    pack_csv_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    best_records: list[dict[str, Any]] = []
    if not args.skip_pack_search:
        with JsonlWriter(args.out_dir / "pack_influence.jsonl") as influence_writer, JsonlWriter(
            args.out_dir / "pack_candidates.jsonl"
        ) as candidate_writer:
            for branch in analyzer.branches():
                prepared = full_prepared[branch]
                full_result = full_results[branch]
                universe = tuple(prepared.pack_universe)
                primary_targets = full_result["oracle"][pack_policy][
                    f"{pack_max_fpr:.6g}"
                ]
                full_values = {
                    target: float(point["objective_tpr"])
                    for target, point in primary_targets.items()
                }
                influence_by_target: dict[str, list[tuple[float, str]]] = {
                    target: [] for target in primary_targets
                }
                for pack in universe:
                    selected = [value for value in universe if value != pack]
                    without = analyzer.evaluate_prepared(
                        prepared,
                        selected_packs=selected,
                        max_fprs=max_fprs,
                    )
                    without_points = without["oracle"][pack_policy][
                        f"{pack_max_fpr:.6g}"
                    ]
                    deltas = {
                        target: full_values[target]
                        - float(without_points[target]["objective_tpr"])
                        for target in primary_targets
                    }
                    influence_writer.write(
                        {
                            "branch": branch,
                            "pack_id": pack,
                            "policy": pack_policy,
                            "max_fpr": pack_max_fpr,
                            "full_objective": full_values,
                            "without_objective": {
                                target: float(point["objective_tpr"])
                                for target, point in without_points.items()
                            },
                            "leave_one_pack_out_delta": deltas,
                        }
                    )
                    for target, delta in deltas.items():
                        influence_by_target[target].append((float(delta), pack))
                    pack_influence_count += 1
                    if pack_influence_count % 10 == 0:
                        progress(
                            "pack_influence",
                            completed=pack_influence_count,
                            branch=branch,
                        )

                rng = np.random.default_rng(
                    seed + sum((index + 1) * ord(char) for index, char in enumerate(branch))
                )
                branch_budgets = sorted(
                    set(
                        [value for value in budgets if analyzer.k < value < len(universe)]
                        + [len(universe)]
                    )
                )
                for budget in branch_budgets:
                    candidate_sets: dict[tuple[str, ...], str] = {}
                    if budget == len(universe):
                        candidate_sets[tuple(universe)] = "full_bank"
                    else:
                        for target, values in influence_by_target.items():
                            ranked = sorted(values, key=lambda item: (-item[0], item[1]))
                            subset = tuple(sorted(pack for _, pack in ranked[:budget]))
                            candidate_sets.setdefault(subset, f"loo_ranked:{target}")
                        for trial in range(random_subsets):
                            subset = tuple(
                                sorted(
                                    rng.choice(
                                        np.asarray(universe),
                                        size=budget,
                                        replace=False,
                                    ).astype(str).tolist()
                                )
                            )
                            candidate_sets.setdefault(subset, f"random:{trial}")

                    best: dict[tuple[str, str, str], tuple[tuple[float, ...], dict[str, Any], str]] = {}
                    for candidate_index, (subset, source) in enumerate(
                        sorted(candidate_sets.items(), key=lambda item: (item[1], item[0]))
                    ):
                        candidate_id = (
                            f"pack:{branch}:b{budget}:c{candidate_index}:"
                            f"{stable_id(*subset)[:8]}"
                        )
                        result = analyzer.evaluate_prepared(
                            prepared,
                            selected_packs=subset,
                            max_fprs=max_fprs,
                        )
                        candidate_writer.write(
                            {
                                "candidate_id": candidate_id,
                                "candidate_source": source,
                                "branch": branch,
                                "budget": budget,
                                "selected_packs": list(subset),
                                "oracle_metrics": _candidate_metrics(result),
                            }
                        )
                        pack_candidate_count += 1
                        for policy, policy_report in result["oracle"].items():
                            for max_fpr, targets in policy_report.items():
                                for target, point in targets.items():
                                    key = (policy, max_fpr, target)
                                    value = _point_key(point, len(subset))
                                    if key not in best or value > best[key][0]:
                                        best[key] = (value, result, candidate_id)
                        if pack_candidate_count % 10 == 0:
                            progress(
                                "pack_candidates",
                                completed=pack_candidate_count,
                                branch=branch,
                                budget=budget,
                            )

                    for (policy, max_fpr, target), (_, result, candidate_id) in sorted(
                        best.items()
                    ):
                        point = result["oracle"][policy][max_fpr][target]
                        record = {
                            "candidate_id": candidate_id,
                            "branch": branch,
                            "budget": budget,
                            "policy": policy,
                            "max_fpr": float(max_fpr),
                            "selection_target": target,
                            "selected_packs": result["selected_packs"],
                            "selection_point": point,
                            "candidate_metrics": _candidate_metrics(result),
                            "views": result["views"],
                            "splits": {
                                split: {
                                    "rows": value["rows"],
                                    "scored_rows": value["scored_rows"],
                                    "score_coverage": value["score_coverage"],
                                    "support_coverage": value["support_coverage"],
                                }
                                for split, value in result["splits"].items()
                            },
                        }
                        best_records.append(record)
                        pack_csv_rows.append(
                            {
                                "candidate_id": candidate_id,
                                "candidate_kind": f"best_for:{target}",
                                "branch": branch,
                                "budget": budget,
                                "pack_count": result["selected_pack_count"],
                                "policy": policy,
                                "max_fpr": float(max_fpr),
                                "target": target,
                                "objective_tpr": point["objective_tpr"],
                                "macro_target_tpr": point["macro_target_tpr"],
                                "worst_target_tpr": point["worst_target_tpr"],
                                "worst_empirical_fpr": point["worst_empirical_fpr"],
                                "worst_fpr_ci95_upper": point[
                                    "worst_fpr_ci95_upper"
                                ],
                                "oracle_threshold": point["threshold"],
                                "calibration_threshold": result["threshold"],
                                "test_support_coverage": result["splits"]
                                .get("test", {})
                                .get("support_coverage"),
                                "external_support_coverage": result["splits"]
                                .get("external", {})
                                .get("support_coverage"),
                            }
                        )
                        group_report = point["by_group"]["groups"]
                        for group, metrics in group_report.items():
                            cross_rows.append(
                                {
                                    "candidate_id": candidate_id,
                                    "branch": branch,
                                    "budget": budget,
                                    "policy": policy,
                                    "max_fpr": float(max_fpr),
                                    "selection_target": target,
                                    "evaluation_group": group,
                                    "n": metrics["n"],
                                    "positive_n": metrics["positive_n"],
                                    "negative_n": metrics["negative_n"],
                                    "tpr": metrics["tpr"],
                                    "fpr": metrics["fpr"],
                                    "threshold": point["threshold"],
                                }
                            )

        with JsonlWriter(args.out_dir / "best_pack_subsets.jsonl") as writer:
            for record in best_records:
                writer.write(record)
        _write_csv(args.out_dir / "pack_budget_oracle.csv", pack_csv_rows)
        _write_csv(args.out_dir / "cross_attack_transfer.csv", cross_rows)

    overview = {
        "format_version": "cnrf_oracle_summary_v1",
        "oracle_only": True,
        "warning": ORACLE_WARNING,
        "source_summary": str(args.summary),
        "manifest": str(args.manifest),
        "activations": str(args.activations),
        "fixed_views": {
            branch: [
                {"readout": view.readout, "layer": view.layer} for view in views
            ]
            for branch, views in fixed_views.items()
        },
        "axes": {branch: list(analyzer.axes(branch)) for branch in analyzer.branches()},
        "max_fprs": max_fprs,
        "pack_budgets": budgets,
        "random_subsets": random_subsets,
        "pack_policy": pack_policy,
        "pack_max_fpr": pack_max_fpr,
        "axis_candidates_completed": axis_count,
        "axis_candidate_failures": axis_failures,
        "sample_pack_influence_rows": influence_rows,
        "pack_influence_rows": pack_influence_count,
        "pack_candidates_evaluated": pack_candidate_count,
        "best_pack_records": len(best_records),
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "axis_subset_results": "axis_subset_results.jsonl",
            "axis_subset_csv": "axis_subset_results.csv",
            "sample_pack_influence": "sample_pack_influence.jsonl",
            "pack_influence": "pack_influence.jsonl",
            "pack_candidates": "pack_candidates.jsonl",
            "best_pack_subsets": "best_pack_subsets.jsonl",
            "pack_budget_oracle": "pack_budget_oracle.csv",
            "cross_attack_transfer": "cross_attack_transfer.csv",
        },
    }
    write_json(args.out_dir / "summary.json", overview)
    write_json(
        run_path,
        {
            **overview,
            "format_version": "cnrf_oracle_run_v1",
            "status": "completed",
            "stage": "completed",
        },
    )
    print(f"Completed CNRF oracle ceiling analysis: {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
