import math
import statistics
import unittest

from solution.src.build_paper_flywheel_pairs import (
    add_paper_weights,
    build_retained_rows,
    collect_browse_candidates,
)


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class BuildPaperFlywheelPairsTests(unittest.TestCase):
    def make_trajectory(self):
        return {
            "result": [
                {
                    "type": "tool_call",
                    "tool_name": "search",
                    "arguments": '{"query":["first query"]}',
                    "output": "DocID: d1\nDocID: d2\nDocID: d3",
                },
                {
                    "type": "tool_call",
                    "tool_name": "get_document",
                    "arguments": '{"docid":"wiki:d2"}',
                },
                {"type": "reasoning", "output": "useful evidence here"},
                {
                    "type": "tool_call",
                    "tool_name": "get_document",
                    "arguments": '{"docid":"d3"}',
                },
                {"type": "reasoning", "output": "irrelevant"},
                {
                    "type": "tool_call",
                    "tool_name": "search",
                    "arguments": '{"query":["second query"]}',
                    "output": "DocID: d4\nDocID: d5",
                },
                {
                    "type": "tool_call",
                    "tool_name": "get_document",
                    "arguments": '{"docid":"d4"}',
                },
                {"type": "reasoning", "output": "second useful evidence"},
            ]
        }

    def test_only_relevant_browse_is_positive_and_negatives_are_search_local(self):
        candidates = collect_browse_candidates(
            self.make_trajectory(), trajectory_path="run_0.json"
        )
        decisions = iter([True, False, True])
        rows, counts = build_retained_rows(
            candidates,
            corpus={f"d{i}": f"text {i}" for i in range(1, 6)},
            tokenizer=TinyTokenizer(),
            judge=lambda _text: next(decisions),
        )
        self.assertEqual(counts["judge_relevant"], 2)
        self.assertEqual(counts["judge_irrelevant"], 1)
        self.assertEqual([row["pos_id"] for row in rows], [["d2"], ["d4"]])
        self.assertEqual(rows[0]["neg_id"], ["d1", "d3"])
        self.assertEqual(rows[1]["neg_id"], ["d5"])
        self.assertTrue(all(row["satisfied"] is True for row in rows))

    def test_weights_follow_equation_three_and_average_one(self):
        rows = [
            {"reasoning_len": 10},
            {"reasoning_len": 20},
            {"reasoning_len": 40},
        ]
        beta, mean_raw = add_paper_weights(rows)
        self.assertEqual(beta, 20)
        expected_raw = [
            1 - math.exp(-math.log(2) * length / beta)
            for length in (10, 20, 40)
        ]
        self.assertAlmostEqual(mean_raw, statistics.fmean(expected_raw))
        self.assertAlmostEqual(
            statistics.fmean(row["reweight_rate"] for row in rows), 1.0
        )


if __name__ == "__main__":
    unittest.main()
