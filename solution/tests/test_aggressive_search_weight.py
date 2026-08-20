import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from solution.src.analyze_search_weight_effects import analyze, normalized_query_sha
from solution.src.audit_aggressive_search_weight_arm import audit
from solution.src.build_aggressive_search_weight_arm import build


def pair(query, pos_id, weight):
    return {
        "query": query,
        "pos": [f"p-{pos_id}"],
        "pos_id": [pos_id],
        "neg": ["negative"],
        "neg_id": ["n"],
        "reasoning_len": 10,
        "satisfied": True,
        "reweight_rate": weight,
    }


def provenance(row_index, row, bucket, search_idx=None):
    value = {
        "row_index": row_index,
        "query": row["query"],
        "normalized_query_sha256": normalized_query_sha(row["query"]),
        "pos_id": row["pos_id"][0],
        "bucket": bucket,
    }
    if bucket == "stable":
        value["event"] = {"search_idx": search_idx}
    return value


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def eval_file(path, input_path, queries, ranks):
    write_jsonl(
        path,
        [],
    )
    path.write_text(
        json.dumps(
            {
                "input": str(input_path),
                "rows": len(queries),
                "details": [
                    {
                        "row_index": index,
                        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                        "best_positive_rank": rank,
                    }
                    for index, (query, rank) in enumerate(zip(queries, ranks))
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


class AggressiveSearchWeightTests(unittest.TestCase):
    def test_build_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                pair("q0", "p0", 0.2),
                pair("q1", "p1", 0.8),
                pair("q2", "p2", 1.2),
                pair("q3", "p3", 2.0),
            ]
            provenance_rows = [
                provenance(0, rows[0], "stable", 0),
                provenance(1, rows[1], "stable", 1),
                provenance(2, rows[2], "stable", 4),
                provenance(3, rows[3], "ambiguous"),
            ]
            pairs_path = root / "pairs.jsonl"
            provenance_path = root / "provenance.jsonl"
            config_path = root / "config.json"
            output_path = root / "candidate.jsonl"
            manifest_path = root / "manifest.json"
            audit_path = root / "audit.json"
            write_jsonl(pairs_path, rows)
            write_jsonl(provenance_path, provenance_rows)
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "method": "test",
                        "hypothesis": "test",
                        "search_stage_buckets": {
                            "idx0": {"description": "first", "raw_weight": 0.75},
                            "idx1_2": {"description": "middle", "raw_weight": 1.0},
                            "idx3plus": {"description": "late", "raw_weight": 1.5},
                        },
                        "ambiguous_weight": 1.0,
                        "locked_test_used": False,
                        "external_data_used": False,
                    }
                ),
                encoding="utf-8",
            )
            manifest = build(
                pairs_path,
                provenance_path,
                config_path,
                output_path,
                manifest_path,
                expected_pairs_sha256=hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
                expected_provenance_sha256=hashlib.sha256(
                    provenance_path.read_bytes()
                ).hexdigest(),
                expected_rows=4,
            )
            candidate = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertAlmostEqual(
                sum(row["reweight_rate"] for row in candidate), 4.0
            )
            self.assertEqual(candidate[3]["reweight_rate"], 1.0)
            self.assertAlmostEqual(
                candidate[2]["reweight_rate"] / candidate[0]["reweight_rate"], 2.0
            )
            self.assertEqual(manifest["contract"]["only_modified_field"], "reweight_rate")
            report = audit(
                pairs_path,
                output_path,
                provenance_path,
                config_path,
                manifest_path,
            )
            audit_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertTrue(report["passed"])

    def test_explanation_groups_by_earliest_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev = root / "dev.jsonl"
            provenance_path = root / "full_provenance.jsonl"
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            output = root / "analysis.json"
            queries = ["Q zero", "Q later", "Q ambiguous"]
            write_jsonl(dev, [{"query": query} for query in queries])
            write_jsonl(
                provenance_path,
                [
                    {
                        "normalized_query_sha256": normalized_query_sha("Q zero"),
                        "bucket": "stable",
                        "event": {"search_idx": 0},
                    },
                    {
                        "normalized_query_sha256": normalized_query_sha("Q later"),
                        "bucket": "stable",
                        "event": {"search_idx": 4},
                    },
                    {
                        "normalized_query_sha256": normalized_query_sha("Q ambiguous"),
                        "bucket": "ambiguous",
                    },
                ],
            )
            eval_file(baseline, dev, queries, [2, 3, 1])
            eval_file(candidate, dev, queries, [1, 2, 1])
            report = analyze(
                dev,
                provenance_path,
                candidate,
                [("baseline", baseline)],
            )
            output.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(
                report["stage_counts"],
                {"ambiguous_only": 1, "earliest_idx0": 1, "earliest_idx3plus": 1},
            )
            self.assertEqual(
                report["comparisons"]["baseline"]["overall"]["improved_rank"], 2
            )


if __name__ == "__main__":
    unittest.main()
