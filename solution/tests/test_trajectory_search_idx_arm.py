import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from solution.src.audit_trajectory_search_idx_arm import audit
from solution.src.build_trajectory_search_idx_arm import build, search_multiplier


def pair(query: str, pos_id: str, weight: float = 1.0) -> dict:
    return {
        "query": query,
        "pos": [f"positive {pos_id}"],
        "pos_id": [pos_id],
        "neg": ["negative"],
        "neg_id": ["n"],
        "reasoning_len": 7,
        "satisfied": True,
        "reweight_rate": weight,
    }


def provenance(row_index: int, row: dict, bucket: str, search_idx=None) -> dict:
    value = {
        "row_index": row_index,
        "query_sha256": hashlib.sha256(row["query"].encode()).hexdigest(),
        "pos_id": row["pos_id"][0],
        "reasoning_len": row["reasoning_len"],
        "reweight_rate": row["reweight_rate"],
        "bucket": bucket,
    }
    if bucket == "stable":
        value["event"] = {"search_idx": search_idx}
    return value


class TrajectorySearchIndexArmTests(unittest.TestCase):
    def test_zero_based_multiplier_contract(self):
        config = {"stable_multipliers": {"0": 1.0, "1": 1.1, "2": 1.1, "3_plus": 1.2}}
        self.assertEqual(search_multiplier(0, config), 1.0)
        self.assertEqual(search_multiplier(1, config), 1.1)
        self.assertEqual(search_multiplier(2, config), 1.1)
        self.assertEqual(search_multiplier(3, config), 1.2)
        self.assertEqual(search_multiplier(12, config), 1.2)
        with self.assertRaises(ValueError):
            search_multiplier(-1, config)

    def test_build_and_audit_preserve_neutral_rows_and_total_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = [pair("q0", "p0"), pair("q1", "p1"), pair("q2", "p2"), pair("q3", "p3")]
            provenance_rows = [
                provenance(0, pairs[0], "stable", 0),
                provenance(1, pairs[1], "stable", 1),
                provenance(2, pairs[2], "stable", 3),
                provenance(3, pairs[3], "ambiguous"),
            ]
            pairs_path = root / "pairs.jsonl"
            provenance_path = root / "provenance.jsonl"
            config_path = root / "config.json"
            output_root = root / "output"
            pairs_path.write_text(
                "".join(json.dumps(row) + "\n" for row in pairs), encoding="utf-8"
            )
            provenance_path.write_text(
                "".join(json.dumps(row) + "\n" for row in provenance_rows),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "feature": "event.search_idx",
                        "stable_multipliers": {
                            "0": 1.0,
                            "1": 1.1,
                            "2": 1.1,
                            "3_plus": 1.2,
                        },
                        "neutral_buckets": ["ambiguous", "mismatch"],
                        "locked_test_used": False,
                    }
                ),
                encoding="utf-8",
            )

            manifest = build(
                pairs_path,
                provenance_path,
                config_path,
                output_root,
                expected_rows=4,
            )
            output_rows = [
                json.loads(line)
                for line in (output_root / "train_search_idx_soft.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertAlmostEqual(sum(row["reweight_rate"] for row in output_rows), 4.0)
            self.assertEqual(output_rows[3]["reweight_rate"], 1.0)
            self.assertGreater(output_rows[2]["reweight_rate"], output_rows[0]["reweight_rate"])
            self.assertEqual(manifest["bucket_counts"], {"ambiguous": 1, "stable": 3})
            self.assertEqual(manifest["changed_rows"], 2)

            report = audit(
                pairs_path,
                output_root / "train_search_idx_soft.jsonl",
                provenance_path,
                output_root / "manifest.json",
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["neutral_changed_rows"], 0)


if __name__ == "__main__":
    unittest.main()
