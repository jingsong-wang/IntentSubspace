from __future__ import annotations

import json
import unittest

from counterfactual_risk_field.cnrf.generation import (
    build_generation_requests,
    ingest_generation_responses,
    materialize_approved_pairs,
)


def sample(sample_id: str, source: str, label: int, role: str, image: str | None = None) -> dict:
    return {
        "sample_id": sample_id,
        "source": source,
        "source_file": f"{source}.json",
        "source_row": 0,
        "label": label,
        "prompt_text": f"prompt {sample_id}",
        "semantic_text": f"semantic {sample_id}",
        "image_path": image,
        "image_exists": image is not None,
        "image_sha256": "image-hash" if image else "",
        "schema_version": "ood_intent_sample_v1",
        "metadata": {"protocol_role": role},
    }


class GenerationTests(unittest.TestCase):
    def test_generation_requires_human_approval_for_formal_pairs(self) -> None:
        rows = [
            sample("harm", "AdvBench", 1, "semantic_seed"),
            sample("text", "Alpaca", 0, "carrier_donor"),
            sample("image", "VizWiz-VQA", 0, "carrier_donor", "image.png"),
        ]
        requests = build_generation_requests(
            rows,
            seed=7,
            split_fractions={"reference": 0.7, "validation": 0.15, "calibration": 0.15},
        )
        responses = []
        for request in requests:
            content = {
                "harmful_request": "harmful request only",
                "benign_counterfactual": "how can this harm be prevented and reported?",
                "risk_domain": "test",
                "changed_risk_variable": "facilitation to prevention",
                "minimality_rationale": "topic is preserved",
                "contains_answer": False,
            }
            responses.append(
                {
                    "request_id": request["request_id"],
                    "content": json.dumps(content),
                    "model": "unit-test",
                }
            )
        candidates, failures = ingest_generation_responses(requests, responses)
        self.assertFalse(failures)
        with self.assertRaisesRegex(ValueError, "No fully approved"):
            materialize_approved_pairs(candidates)
        candidates[0]["audit_status"] = "approved"
        with self.assertRaisesRegex(ValueError, "No fully approved"):
            materialize_approved_pairs(candidates)
        candidates[0]["human_review"].update(
            {
                "topic_preserved": True,
                "entities_preserved": True,
                "carrier_preserved": True,
                "risk_is_only_material_change": True,
                "no_answer_or_operational_expansion": True,
                "reviewer": "unit-test-reviewer",
            }
        )
        endpoints = materialize_approved_pairs(candidates)
        self.assertEqual(len(endpoints), 2)
        self.assertEqual({row["label"] for row in endpoints}, {0, 1})
        self.assertEqual(endpoints[0]["metadata"]["pack_id"], endpoints[1]["metadata"]["pack_id"])


if __name__ == "__main__":
    unittest.main()
