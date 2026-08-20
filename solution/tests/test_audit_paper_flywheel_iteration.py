import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path

from solution.src.audit_paper_flywheel_iteration import (
    audit_iteration,
    audit_preflight,
)


class AuditPaperFlywheelIterationTests(unittest.TestCase):
    def make_repo(self, root: Path) -> tuple[Path, Path, Path]:
        for relative in (
            "search_agent/tongyi_client.py",
            "src/data_builder.py",
            "solution/experiments/train.sh",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# ok\n", encoding="utf-8")
        (root / "FlagEmbedding").mkdir()
        research = root / "ccir/research/paper_flywheel_v1"
        inputs = research / "inputs"
        inputs.mkdir(parents=True)
        pool = inputs / "queries.tsv"
        pool.write_text("0\tQuestion zero\n", encoding="utf-8")
        (inputs / "seed.json").write_text(
            json.dumps(
                {
                    "external_data_used": True,
                    "competition_submission_eligible": False,
                }
            ),
            encoding="utf-8",
        )
        raw = root / "ccir/data/raw"
        raw.mkdir(parents=True)
        (raw / "trajectory.tar.gz").write_bytes(b"archive")
        (raw / "corpus.jsonl").write_text(
            '{"docid":"d0","text":"text"}\n', encoding="utf-8"
        )
        model = root / "ccir/models/base"
        model.mkdir(parents=True)
        (model / "model.safetensors").write_bytes(b"model")
        config = {
            "research_only": True,
            "competition_submission_eligible": False,
            "paper_contract": {
                "seed_queries": 1,
                "train_epochs_per_loop": 2,
                "learning_rate": 1e-6,
                "temperature": 0.02,
                "weighted_loss_reduction": "mean_over_minibatch",
                "reported_train_batch_size": 32,
                "train_group_size": 10,
                "query_max_tokens": 512,
                "passage_max_tokens": 512,
            },
            "paths": {
                "research_root": "ccir/research/paper_flywheel_v1",
                "seed_pool": "ccir/research/paper_flywheel_v1/inputs/queries.tsv",
                "seed_manifest": "ccir/research/paper_flywheel_v1/inputs/seed.json",
                "trajectory_archive": "ccir/data/raw/trajectory.tar.gz",
                "corpus": "ccir/data/raw/corpus.jsonl",
                "initial_retriever": "ccir/models/base",
            },
            "runtime": {
                "agent_model": "/missing/agent",
                "judge_model": "/missing/judge",
                "training_profile": "paper_reported_batch32",
                "training_profiles": {
                    "paper_reported_batch32": {
                        "world_size": 2,
                        "per_device_train_batch_size": 16,
                        "gradient_accumulation_steps": 1,
                        "global_microbatch_queries": 32,
                        "exact_in_batch_negative_pool": True,
                    }
                },
            },
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path, research, model

    def test_preflight_reports_missing_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, _ = self.make_repo(root)
            result = audit_preflight(config, root)
            self.assertFalse(result["ready_for_full_collection"])
            self.assertEqual(result["query_count"], 1)
            self.assertEqual(len(result["blockers"]), 2)

    def test_iteration_checks_weight_formula_and_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, research, model = self.make_repo(root)
            loop = research / "loop_00"
            trajectories = loop / "trajectories"
            index = loop / "index"
            trajectories.mkdir(parents=True)
            index.mkdir()
            query_path = loop / "queries.tsv"
            query_path.write_text("0\tQuestion zero\n", encoding="utf-8")
            import hashlib

            query_sha = hashlib.sha256(query_path.read_bytes()).hexdigest()
            (loop / "queries.manifest.json").write_text(
                json.dumps({"output_sha256": query_sha, "sample_count": 1}),
                encoding="utf-8",
            )
            (trajectories / "run_0.json").write_text(
                json.dumps(
                    {
                        "query_id": "0",
                        "status": "completed",
                        "result": [
                            {
                                "type": "tool_call",
                                "tool_name": "search",
                                "arguments": '{"query":["q"]}',
                                "output": "DocID: p\nDocID: n",
                            },
                            {
                                "type": "tool_call",
                                "tool_name": "get_document",
                                "arguments": '{"docid":"p"}',
                            },
                            {"type": "reasoning", "output": "first"},
                            {
                                "type": "tool_call",
                                "tool_name": "get_document",
                                "arguments": '{"docid":"p"}',
                            },
                            {"type": "reasoning", "output": "second"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (trajectories / "COMPLETED").write_text("ok\n")
            (trajectories / "COLLECTION_MANIFEST.json").write_text(
                json.dumps({"complete": True}), encoding="utf-8"
            )

            model_sha = hashlib.sha256(b"model").hexdigest()
            shard = index / "index-00.pkl"
            shard.write_bytes(b"index")
            (index / "manifest.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "retriever_model_sha256": model_sha,
                        "shards": [
                            {
                                "path": str(shard),
                                "bytes": shard.stat().st_size,
                                "sha256": hashlib.sha256(
                                    shard.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            lengths = [10.0, 20.0]
            beta = statistics.median(lengths)
            raw = [1 - math.exp(-math.log(2) * value / beta) for value in lengths]
            mean_raw = statistics.fmean(raw)
            rows = []
            for length, weight in zip(lengths, (value / mean_raw for value in raw)):
                rows.append(
                    {
                        "query": "q",
                        "pos": ["p"],
                        "neg": ["n"],
                        "pos_id": ["p"],
                        "neg_id": ["n"],
                        "reasoning_len": length,
                        "satisfied": True,
                        "reweight_rate": weight,
                        "source_trajectory": "run_0.json",
                    }
                )
            (loop / "training_pairs.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            (loop / "training_pairs.summary.json").write_text(
                json.dumps(
                    {
                        "paper_contract": {
                            "irrelevant_browses_removed_from_positives": True,
                            "negatives_only_from_corresponding_search_candidate_set": True,
                            "immediate_post_browse_reasoning_used": True,
                            "global_eq3_weights": True,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (loop / "PAIRS_COMPLETED").write_text("ok\n")
            result = audit_iteration(
                config, root, loop_dir=loop, current_model=model
            )
            self.assertTrue(result["paper_contract_passed"])
            self.assertEqual(result["search_to_browse_transitions"], 1)
            self.assertLessEqual(result["maximum_weight_formula_error"], 1e-9)

            rows[0]["reweight_rate"] += 0.1
            (loop / "training_pairs.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "formula mismatch"):
                audit_iteration(config, root, loop_dir=loop, current_model=model)


if __name__ == "__main__":
    unittest.main()
