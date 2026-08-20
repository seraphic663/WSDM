import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from solution.src.audit_pair_documents_against_trajectories import (
    audit,
    canonical_document_text,
    parse_document_output,
)


class PairDocumentTrajectoryAuditTests(unittest.TestCase):
    def test_parse_document_output(self):
        step = {
            "type": "tool_call",
            "tool_name": "get_document",
            "arguments": '{"docid": 7}',
            "output": "Document 7:\nTitle\nContent",
        }
        self.assertEqual(parse_document_output(step), ("7", "Title\nContent"))
        self.assertEqual(canonical_document_text("Title\nContent. \n"), "Title\nContent.")

    def test_full_traceability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "trajectories.tar.gz"
            trajectory = {
                "result": [
                    {
                        "type": "tool_call",
                        "tool_name": "get_document",
                        "arguments": '{"docid": "p"}',
                        "output": "Document p:\nPositive\nBody",
                    },
                    {
                        "type": "tool_call",
                        "tool_name": "visit",
                        "arguments": '{"id": "n"}',
                        "output": "Document n:\nNegative\nBody",
                    },
                    {
                        "type": "tool_call",
                        "tool_name": "get_document",
                        "arguments": '{"docid": "broken"}',
                        "output": "Document other:\nWrong\nBody",
                    },
                ]
            }
            payload = json.dumps(trajectory).encode()
            with tarfile.open(archive, "w:gz") as bundle:
                info = tarfile.TarInfo("trajectories/test/sample.json")
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))
            pairs = root / "pairs.jsonl"
            pairs.write_text(
                json.dumps(
                    {
                        "query": "q",
                        "pos": ["Positive\nBody"],
                        "pos_id": ["p"],
                        "neg": ["Negative\nBody"],
                        "neg_id": ["n"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = audit(archive, pairs, expected_rows=1)
            self.assertTrue(result["full_document_traceability_verified"])
            self.assertEqual(result["pair_references"]["matched_total"], 2)
            self.assertEqual(result["pair_references"]["unresolved_total"], 0)
            self.assertEqual(
                result["trajectory_documents"]["parse_issue_counts"],
                {"argument_output_doc_id_mismatch": 1},
            )


if __name__ == "__main__":
    unittest.main()
