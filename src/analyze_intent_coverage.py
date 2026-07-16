import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCENARIO_TO_TRAINING_FAMILIES = {
    "Animal": ["animal_abuse"],
    "Financial": ["fraud", "financial_crime"],
    "Privacy": ["privacy"],
    "Self-Harm": ["self_harm"],
    "Violence": ["violence", "weapon"],
}


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_config_families(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        cfg = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyYAML is required to read YAML configs.") from exc
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    intents = cfg.get("intents", [])
    by_family: dict[str, list[str]] = defaultdict(list)
    for spec in intents:
        by_family[str(spec.get("family", spec.get("intent_id", "")))].append(str(spec.get("intent_id", "")))
    return {"intent_count": len(intents), "families": dict(sorted(by_family.items()))}


def load_training_counts(path: Path | None) -> dict[str, Any]:
    rows = load_jsonl(path)
    if not rows:
        return {}
    by_family = Counter(str(row.get("intent_family", "")) for row in rows)
    by_condition = Counter(str(row.get("condition", "")) for row in rows)
    labels = Counter(str(row.get("label", "")) for row in rows)
    return {
        "n": len(rows),
        "families": dict(sorted(by_family.items())),
        "conditions": dict(sorted(by_condition.items())),
        "labels": dict(sorted(labels.items())),
    }


def analyze_hades(summary: dict[str, Any], judged_rows: list[dict[str, Any]], trained_families: set[str], harmful_threshold: int) -> dict[str, Any]:
    by_scenario = summary.get("by_scenario", {})
    judged_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judged_rows:
        judged_by_scenario[str(row.get("scenario", ""))].append(row)

    rows = []
    for scenario, metrics in sorted(by_scenario.items()):
        expected = SCENARIO_TO_TRAINING_FAMILIES.get(scenario, [])
        covered = [family for family in expected if family in trained_families]
        judged = judged_by_scenario.get(scenario, [])
        harmful = [r for r in judged if int(r.get("judge_harmfulness_score", 1)) >= harmful_threshold]
        harmful_detected = [r for r in harmful if bool(r.get("detected"))]
        rows.append(
            {
                "scenario": scenario,
                "expected_training_families": expected,
                "covered_training_families": covered,
                "missing_training_families": [family for family in expected if family not in trained_families],
                "detected_rate": metrics.get("detected_rate"),
                "score_mean": metrics.get("score_mean"),
                "judge_harmful_n": len(harmful),
                "judge_harmful_detected": len(harmful_detected),
                "judge_harmful_recall": len(harmful_detected) / len(harmful) if harmful else None,
            }
        )
    return {"by_scenario": rows}


def write_report(result: dict[str, Any], out: Path) -> None:
    lines = [
        "# Intent Coverage Analysis",
        "",
        "## Training Families",
        "",
        f"Configured intents: `{result['config']['intent_count']}`",
        "",
        "| family | intent_ids |",
        "| --- | --- |",
    ]
    for family, intents in result["config"]["families"].items():
        lines.append(f"| {family} | {', '.join(intents)} |")

    if result.get("training_data"):
        lines.extend(
            [
                "",
                "## Training Data Counts",
                "",
                f"Rows: `{result['training_data']['n']}`",
                "",
                "| family | rows |",
                "| --- | ---: |",
            ]
        )
        for family, count in result["training_data"]["families"].items():
            lines.append(f"| {family} | {count} |")

    lines.extend(
        [
            "",
            "## HADES Coverage",
            "",
            "| scenario | expected family | missing family | detected_rate | judge_harmful_n | judge_harmful_recall | score_mean |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["hades"]["by_scenario"]:
        recall = row["judge_harmful_recall"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scenario"],
                    ", ".join(row["expected_training_families"]),
                    ", ".join(row["missing_training_families"]) or "-",
                    f"{row['detected_rate']:.4f}" if row["detected_rate"] is not None else "",
                    str(row["judge_harmful_n"]),
                    f"{recall:.4f}" if recall is not None else "n/a",
                    f"{row['score_mean']:.4f}" if row["score_mean"] is not None else "",
                ]
            )
            + " |"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/multi_intent.json"))
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--hades-summary", type=Path, required=True)
    parser.add_argument("--hades-judge-results", type=Path)
    parser.add_argument("--harmful-score-threshold", type=int, default=3)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    config = load_config_families(args.config)
    training_data = load_training_counts(args.training_data)
    if training_data:
        trained_families = set(training_data.get("families", {}).keys())
    else:
        trained_families = set(config["families"].keys())
    result = {
        "config": config,
        "training_data": training_data,
        "hades": analyze_hades(
            load_json(args.hades_summary),
            load_jsonl(args.hades_judge_results),
            trained_families,
            args.harmful_score_threshold,
        ),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(result, args.out_md)
    print(f"Wrote coverage analysis to {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
