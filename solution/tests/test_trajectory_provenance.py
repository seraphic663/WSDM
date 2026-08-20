import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from solution.src.build_trajectory_provenance import (
    answer_features,
    build_feature_report,
    iter_archive_events,
    map_pairs,
    negative_cues,
)


class TrajectoryProvenanceTests(unittest.TestCase):
    def test_answer_features_allow_noncontiguous_gold_tokens(self):
        value = answer_features(
            "Sosnovy Bor, Leningrad Oblast",
            "The answer is Sosnovy Bor, located in the wider Leningrad Oblast region.",
        )
        self.assertFalse(value["answer_exact_substring"])
        self.assertTrue(value["answer_token_subset"])
        self.assertEqual(value["answer_token_coverage"], 1.0)

    def test_negative_cue_detection_is_explicit(self):
        self.assertIn(
            "explicit_not_relevant",
            negative_cues("This page is not directly relevant, so keep searching."),
        )
        self.assertEqual(negative_cues("This confirms the required answer."), [])

    def test_archive_event_preserves_trajectory_context(self):
        trajectory = {
            "query_id": "seed-1",
            "answer": "Alpha",
            "status": "completed",
            "result": [
                {
                    "type": "tool_call",
                    "tool_name": "search",
                    "arguments": json.dumps({"query": ["intermediate query"]}),
                    "output": "DocID: 10\nDocID: 20",
                },
                {
                    "type": "tool_call",
                    "tool_name": "get_document",
                    "arguments": json.dumps({"docid": ["wiki:20"]}),
                    "output": "document text",
                },
                {"type": "reasoning", "output": "This confirms Alpha."},
                {"type": "output_text", "output": "The answer is Alpha."},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "trajectories.tar.gz"
            encoded = json.dumps(trajectory).encode()
            with tarfile.open(archive, "w:gz") as bundle:
                info = tarfile.TarInfo("trajectories/bm25_true/seed-1.json")
                info.size = len(encoded)
                bundle.addfile(info, io.BytesIO(encoded))
            events = list(iter_archive_events(archive))
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["query"], "intermediate query")
        self.assertEqual(event["pos_id"], "20")
        self.assertEqual(event["retrieved_rank"], 2)
        self.assertEqual(event["retriever"], "bm25_true")
        self.assertEqual(event["event_idx"], 0)
        self.assertEqual(event["searched_doc_ids_so_far"], ["10", "20"])
        self.assertEqual(event["trajectory_source_doc_ids"], ["10", "20"])
        self.assertTrue(event["answer_token_subset"])

    def test_mapping_buckets_require_unique_signature(self):
        events = [
            {
                "key_sha256": "key-a",
                "reasoning_len": 12,
                "traj_path": "a.json",
                "query": "q",
                "reasoning_text": "r",
                "answer_token_subset": True,
                "next_tool_name": "search",
                "retrieved_rank": 1,
                "negative_cues": [],
                "trajectory_steps": 5,
                "trajectory_searches": 1,
                "trajectory_browses": 1,
                "search_idx": 0,
                "browse_idx_in_search": 0,
                "next_step_type": "tool_call",
                "searched_doc_ids_so_far": ["n1", "n2"],
                "trajectory_source_doc_ids": ["n1", "n2"],
            },
            {
                "key_sha256": "key-b",
                "reasoning_len": 20,
                "traj_path": "b1.json",
                "query": "q",
                "reasoning_text": "r",
                "searched_doc_ids_so_far": ["n1", "n2"],
                "trajectory_source_doc_ids": ["n1", "n2"],
            },
            {
                "key_sha256": "key-b",
                "reasoning_len": 20,
                "traj_path": "b2.json",
                "query": "q",
                "reasoning_text": "r",
                "searched_doc_ids_so_far": ["n1", "n2"],
                "trajectory_source_doc_ids": ["n1", "n2"],
            },
        ]
        base = {
            "row_index": 0,
            "source_line": 1,
            "query": "q",
            "query_sha256": "q",
            "normalized_query_sha256": "n",
            "pos_id": "p",
            "reweight_rate": 1.0,
            "satisfied": True,
            "negative_count": 5,
            "_negative_ids": ["n1", "n2"],
        }
        pairs = [
            {**base, "key_sha256": "key-a", "reasoning_len": 12},
            {**base, "row_index": 1, "source_line": 2, "key_sha256": "key-b", "reasoning_len": 20},
            {**base, "row_index": 2, "source_line": 3, "key_sha256": "key-c", "reasoning_len": 7},
        ]
        mapped, summary = map_pairs(pairs, events)
        self.assertEqual([item["bucket"] for item in mapped], ["stable", "ambiguous", "mismatch"])
        self.assertEqual(summary["buckets"], {"stable": 1, "ambiguous": 1, "mismatch": 1})


if __name__ == "__main__":
    unittest.main()
