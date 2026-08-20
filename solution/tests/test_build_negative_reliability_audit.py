import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from build_negative_reliability_audit import (  # noqa: E402
    normalized_text,
    parse_search_observation,
    parse_trajectory,
    strict_answer_match,
)


def search_message(call_id, hits):
    body = "A local corpus search found results:\n\n## Local Results\n"
    for rank, doc_id, score in hits:
        body += f"{rank}. Document ID: {doc_id}\nScore: {score}\nSnippet: text\n\n"
    return {"role": "tool", "name": "search", "tool_call_id": call_id, "content": body}


class CandidateAuditTest(unittest.TestCase):
    def test_search_parser(self):
        hits = parse_search_observation(search_message("s1", [(1, "10", "0.8"), (2, "20", "0.7")])["content"])
        self.assertEqual([hit["doc_id"] for hit in hits], ["10", "20"])
        self.assertEqual(hits[1]["rank"], 2)
        self.assertAlmostEqual(hits[0]["score"], 0.8)

    def test_visit_and_later_visit_are_separate(self):
        record = {
            "question": " A  question\n",
            "answer": "Alpha University",
            "prediction": "The answer is Alpha University.",
            "termination": "answer",
            "messages": [
                {"role": "assistant", "tool_calls": [{"id": "s1", "name": "search", "arguments": {"query": "one"}}]},
                search_message("s1", [(1, "10", "0.8"), (2, "20", "0.7")]),
                {"role": "assistant", "tool_calls": [{"id": "v1", "name": "visit", "arguments": {"docid": "10"}}]},
                {"role": "tool", "name": "visit", "tool_call_id": "v1", "content": "Successfully visited the docid 10."},
                {"role": "assistant", "tool_calls": [{"id": "s2", "name": "search", "arguments": {"query": "two"}}]},
                search_message("s2", [(1, "20", "0.9"), (2, "30", "0.6")]),
                {"role": "assistant", "tool_calls": [{"id": "v2", "name": "visit", "arguments": {"docid": "20"}}]},
                {"role": "tool", "name": "visit", "tool_call_id": "v2", "content": "Successfully visited the docid 20."},
            ],
        }
        parsed = parse_trajectory(
            record,
            agent="agent",
            trajectory_id="agent:1",
            query_id="1",
            evidence_docs={"20", "30"},
            gold_docs={"30"},
        )
        rows = {(row["search_step"], row["doc_id"]): row for row in parsed.rows}
        self.assertTrue(rows[(0, "10")]["visited"])
        self.assertFalse(rows[(0, "10")]["later_visited"])
        self.assertFalse(rows[(0, "20")]["visited"])
        self.assertTrue(rows[(0, "20")]["later_visited"])
        self.assertTrue(rows[(1, "20")]["visited"])
        self.assertTrue(rows[(1, "20")]["repeated_retrieval"])
        self.assertEqual(rows[(1, "30")]["human_relevance"], "gold_qrel_positive")
        self.assertEqual(rows[(1, "20")]["evidence_state"], "partial")
        self.assertIsNone(rows[(0, "10")]["utilized"])
        self.assertTrue(rows[(0, "10")]["strict_answer_match_proxy"])
        self.assertEqual(parsed.stats["successful_visit_observations_mapped_to_any_surface"], 2)

    def test_text_and_answer_normalization(self):
        self.assertEqual(normalized_text(" A\n B "), "a b")
        self.assertTrue(strict_answer_match("Queen Arwa University", "Queen Arwa University (Arabic)"))
        self.assertFalse(strict_answer_match("Queen Arwa University", "Another institution"))
        self.assertIsNone(strict_answer_match("", "anything"))


if __name__ == "__main__":
    unittest.main()
