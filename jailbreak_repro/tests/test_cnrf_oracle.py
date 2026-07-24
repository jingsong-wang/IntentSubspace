from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from jailbreak_repro.cnrf_oracle import summarize_oracle_run


FIELDS = [
    "axes",
    "axis_count",
    "branch",
    "budget",
    "candidate_id",
    "external_support_coverage",
    "macro_target_tpr",
    "max_fpr",
    "objective_tpr",
    "pack_count",
    "policy",
    "target",
    "test_support_coverage",
    "worst_empirical_fpr",
    "worst_fpr_ci95_upper",
    "worst_target_tpr",
]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(candidate: str, tpr: float, *, budget: object, pack_count: int) -> dict[str, object]:
    return {
        "axes": "axis_a",
        "axis_count": 1,
        "branch": "image_text",
        "budget": budget,
        "candidate_id": candidate,
        "external_support_coverage": 0.8,
        "macro_target_tpr": tpr,
        "max_fpr": 0.05,
        "objective_tpr": tpr,
        "pack_count": pack_count,
        "policy": "abstain_safe",
        "target": "group:JOOD",
        "test_support_coverage": 0.9,
        "worst_empirical_fpr": 0.05,
        "worst_fpr_ci95_upper": 0.09,
        "worst_target_tpr": tpr,
    }


class CnrfOracleSummaryTests(unittest.TestCase):
    def test_summary_preserves_oracle_boundary_and_selects_best_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            out = root / "out"
            raw.mkdir()
            (raw / "summary.json").write_text(
                json.dumps(
                    {
                        "format_version": "cnrf_oracle_summary_v1",
                        "oracle_only": True,
                        "axis_candidates_completed": 2,
                        "pack_candidates_evaluated": 3,
                    }
                ),
                encoding="utf-8",
            )
            _write_rows(
                raw / "axis_subset_results.csv",
                [
                    _row("axis:weak", 0.3, budget="axis_subset", pack_count=10),
                    _row("axis:best", 0.6, budget="axis_subset", pack_count=5),
                ],
            )
            _write_rows(
                raw / "pack_budget_oracle.csv",
                [
                    _row("pack:full", 0.2, budget=10, pack_count=10),
                    _row("pack:small", 0.7, budget=5, pack_count=5),
                    _row("pack:other", 0.5, budget=7, pack_count=7),
                ],
            )

            summary = summarize_oracle_run(
                raw,
                out,
                model_tag="fixture",
                source_work=root,
            )

            self.assertTrue(summary["oracle_only"])
            self.assertFalse(summary["paper_claim_compatible"])
            self.assertEqual(summary["result_count"], 1)
            result = summary["results"][0]
            self.assertEqual(result["full_bank_candidate_id"], "pack:full")
            self.assertEqual(result["best_axis_candidate_id"], "axis:best")
            self.assertEqual(result["best_pack_candidate_id"], "pack:small")
            self.assertAlmostEqual(result["best_pack_delta_vs_full"], 0.5)
            self.assertTrue((out / "summary.csv").is_file())
            self.assertTrue((out / "summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
