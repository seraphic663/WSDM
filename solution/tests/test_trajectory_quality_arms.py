import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from solution.src.build_trajectory_quality_arms import build, row_flags


class TrajectoryQualityArmTests(unittest.TestCase):
    def test_row_flags_do_not_treat_continue_search_as_negative_by_itself(self):
        provenance = {
            "bucket": "stable",
            "event": {
                "negative_cues": [],
                "reasoning_len": 25,
                "answer_token_subset": True,
                "trajectory_steps": 20,
                "next_tool_name": "search",
                "retrieved_rank": 2,
            },
        }
        self.assertEqual(
            row_flags(provenance, long_steps=100, low_rank=8),
            {"continue_search"},
        )

    def test_build_preserves_b_rows_and_normalizes_c_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = root / "pairs.jsonl"
            provenance = root / "provenance.jsonl"
            policy = root / "policy.json"
            output = root / "arms"
            rows = [
                {
                    "query": "q1",
                    "pos": ["p1"],
                    "pos_id": ["1"],
                    "neg": ["n1"],
                    "neg_id": ["2"],
                    "reasoning_len": 10,
                    "satisfied": True,
                    "reweight_rate": 1.0,
                },
                {
                    "query": "q2",
                    "pos": ["p2"],
                    "pos_id": ["3"],
                    "neg": ["n2"],
                    "neg_id": ["4"],
                    "reasoning_len": 20,
                    "satisfied": True,
                    "reweight_rate": 1.0,
                },
            ]
            with pairs.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            provenance_rows = [
                {
                    "row_index": 0,
                    "query_sha256": hashlib.sha256(b"q1").hexdigest(),
                    "pos_id": "1",
                    "reasoning_len": 10,
                    "reweight_rate": 1.0,
                    "bucket": "stable",
                    "event": {
                        "negative_cues": ["explicit_not_relevant"],
                        "reasoning_len": 10,
                        "answer_token_subset": False,
                        "trajectory_steps": 20,
                        "next_tool_name": "search",
                        "retrieved_rank": 1,
                    },
                },
                {
                    "row_index": 1,
                    "query_sha256": hashlib.sha256(b"q2").hexdigest(),
                    "pos_id": "3",
                    "reasoning_len": 20,
                    "reweight_rate": 1.0,
                    "bucket": "stable",
                    "event": {
                        "negative_cues": [],
                        "reasoning_len": 20,
                        "answer_token_subset": True,
                        "trajectory_steps": 20,
                        "next_tool_name": "output",
                        "retrieved_rank": 1,
                    },
                },
            ]
            with provenance.open("w", encoding="utf-8") as handle:
                for row in provenance_rows:
                    handle.write(json.dumps(row) + "\n")
            policy.write_text(
                json.dumps(
                    {
                        "version": "trajectory_quality_policy_v1",
                        "thresholds": {
                            "long_trajectory_min_steps": 126,
                            "low_rank_min": 8,
                        },
                        "hard_delete": {
                            "stable_only": True,
                            "manual_bad_row_indices": [0],
                        },
                        "soft_multipliers": {
                            "explicit_negative_reasoning": 0.25
                        },
                        "minimum_combined_multiplier": 0.2,
                        "manual_review": {"completed": True},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = build(
                pairs,
                provenance,
                policy,
                output,
                expected_rows=2,
                expected_pairs_sha256=None,
            )
            self.assertEqual(result["rows"]["arm_b"], 1)
            self.assertEqual(result["rows"]["arm_c"], 2)
            b_lines = (output / "arm_b_hard_filter.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(b_lines[0])["query"], "q2")
            c_rows = [
                json.loads(line)
                for line in (output / "arm_c_soft_weight.jsonl").read_text().splitlines()
            ]
            self.assertAlmostEqual(
                sum(row["reweight_rate"] for row in c_rows) / 2,
                1.0,
                places=12,
            )
            self.assertLess(c_rows[0]["reweight_rate"], c_rows[1]["reweight_rate"])

    def test_strict_negative_combo_flag_requires_negative_and_second_signal(self):
        base = {
            "bucket": "stable",
            "event": {
                "negative_cues": ["explicit_not_relevant"],
                "reasoning_len": 25,
                "answer_token_subset": False,
                "trajectory_steps": 20,
                "next_tool_name": "search",
                "retrieved_rank": 2,
            },
        }
        self.assertIn(
            "strict_negative_combo",
            row_flags(base, long_steps=100, low_rank=8),
        )


if __name__ == "__main__":
    unittest.main()
