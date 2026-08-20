import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from build_lrat_reliability_arms import event_features, transform_row  # noqa: E402


class ReliabilityArmsTest(unittest.TestCase):
    def test_event_features_distinguish_repetition_and_future_visit(self):
        steps = [
            {"type": "tool_call", "tool_name": "search", "arguments": '{"query":"q"}', "output": "DocID: a\nDocID: b"},
            {"type": "tool_call", "tool_name": "search", "arguments": '{"query":"q2"}', "output": "DocID: b\nDocID: c"},
            {"type": "tool_call", "tool_name": "visit", "arguments": '{"docid":"a"}'},
            {"type": "reasoning", "output": "x"},
            {"type": "tool_call", "tool_name": "visit", "arguments": '{"docid":"b"}'},
        ]
        features = event_features(steps)
        self.assertEqual(features[0]["repeated"], {"b"})
        self.assertEqual(features[0]["later_visited"], {"b"})
        self.assertEqual(features[1]["later_visited"], set())

    def test_arms_preserve_alignment_and_match_random_count(self):
        row = {
            "query": "q",
            "pos": ["p"],
            "pos_id": ["p"],
            "neg": [f"text-{i}" for i in range(8)],
            "neg_id": [str(i) for i in range(8)],
            "reweight_rate": 1.0,
        }
        provenance = {"bucket": "stable", "source_line": 7}
        arms, details = transform_row(
            row,
            provenance,
            {"later_visited": {"1", "4"}, "repeated": {"2"}},
            minimum_negatives=5,
            seed=9,
        )
        self.assertEqual(len(arms["later_visit"]["neg_id"]), 6)
        self.assertEqual(len(arms["random"]["neg_id"]), 6)
        self.assertEqual(len(arms["exposure"]["neg_id"]), 6)
        for arm in arms.values():
            self.assertEqual(len(arm["neg"]), len(arm["neg_id"]))
            self.assertEqual(arm["query"], row["query"])
        self.assertEqual(details["later_removed"], 2)

    def test_minimum_negative_floor_caps_removal(self):
        row = {"neg": list("abcdef"), "neg_id": list("123456")}
        arms, details = transform_row(
            row,
            {"bucket": "stable", "source_line": 1},
            {"later_visited": set("1234"), "repeated": set()},
            minimum_negatives=5,
            seed=1,
        )
        self.assertEqual(len(arms["later_visit"]["neg_id"]), 5)
        self.assertTrue(details["capped"])


if __name__ == "__main__":
    unittest.main()
