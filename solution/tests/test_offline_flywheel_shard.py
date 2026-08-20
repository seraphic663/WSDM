import json
import tempfile
import unittest
from pathlib import Path

from solution.src.audit_offline_flywheel_shard import audit
from solution.src.build_offline_flywheel_shard import (
    candidate_row,
    negative_pool_indices,
    query_bucket,
    query_hash,
    sha256_file,
)


def row(query: str, count: int = 10) -> dict:
    return {
        "query": query,
        "pos": [f"positive {query}"],
        "pos_id": [f"p-{query}"],
        "neg": [f"negative {query} {index}" for index in range(count)],
        "neg_id": [f"n-{query}-{index}" for index in range(count)],
        "reasoning_len": 10,
        "satisfied": True,
        "reweight_rate": 1.0,
    }


class OfflineFlywheelShardTests(unittest.TestCase):
    def test_bucket_groups_normalized_queries_and_pool_is_deterministic(self) -> None:
        self.assertEqual(
            query_bucket("  Mixed   CASE ", salt="s", modulus=16),
            query_bucket("mixed case", salt="s", modulus=16),
        )
        value = row("q", 20)
        first = negative_pool_indices(
            value, pool_size=8, normalized_query_hash=query_hash("q")
        )
        second = negative_pool_indices(
            value, pool_size=8, normalized_query_hash=query_hash("q")
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(len(set(first)), 8)

    def test_candidate_changes_only_aligned_negatives(self) -> None:
        source = row("q")
        result = candidate_row(source, [8, 2, 5, 1, 0])
        self.assertEqual(result["neg_id"], [source["neg_id"][i] for i in [8, 2, 5, 1, 0]])
        self.assertEqual(result["neg"], [source["neg"][i] for i in [8, 2, 5, 1, 0]])
        self.assertEqual(
            {k: v for k, v in result.items() if k not in {"neg", "neg_id"}},
            {k: v for k, v in source.items() if k not in {"neg", "neg_id"}},
        )

    def test_independent_audit_accepts_traceable_synthetic_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source_rows = [row("q1"), row("q2")]
            source.write_text(
                "".join(json.dumps(value) + "\n" for value in source_rows),
                encoding="utf-8",
            )
            shard = root / "shard"
            shard.mkdir()
            selected = [source_rows[0]]
            candidate = [candidate_row(source_rows[0], [3, 2, 1, 0, 4])]
            metadata = [
                {
                    "source_line": 1,
                    "query_sha256": query_hash("q1"),
                    "selected_negative_ids": candidate[0]["neg_id"],
                    "selected_negative_scores": [0.9, 0.8, 0.7, 0.6, 0.5],
                    "best_positive_score": 0.4,
                    "best_negative_score": 0.9,
                    "positive_negative_margin": -0.5,
                }
            ]
            for name, values in (
                ("control.jsonl", selected),
                ("candidate.jsonl", candidate),
                ("mining.jsonl", metadata),
            ):
                (shard / name).write_text(
                    "".join(json.dumps(value) + "\n" for value in values),
                    encoding="utf-8",
                )
            bucket = query_bucket("q1", salt="s", modulus=16)
            manifest = {
                "inputs": {"source": {"sha256": sha256_file(source)}},
                "shard": {"salt": "s", "modulus": 16, "bucket": bucket},
                "outputs": {
                    "control": {"sha256": sha256_file(shard / "control.jsonl")},
                    "candidate": {"sha256": sha256_file(shard / "candidate.jsonl")},
                    "metadata": {"sha256": sha256_file(shard / "mining.jsonl")},
                },
            }
            (shard / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            result = audit(source=source, shard_dir=shard, output=shard / "audit.json")
            self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
