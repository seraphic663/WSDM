import json
import unittest
from pathlib import Path


class PaperFlywheelContractTests(unittest.TestCase):
    @property
    def root(self):
        return Path(__file__).resolve().parents[2]

    def test_config_matches_reported_lrat_method(self):
        config = json.loads(
            (self.root / "solution/configs/paper_flywheel_v1.json").read_text()
        )
        paper = config["paper_contract"]
        self.assertEqual(paper["seed_queries"], 10_000)
        self.assertEqual(paper["train_group_size"], 10)
        self.assertEqual(paper["query_max_tokens"], 512)
        self.assertEqual(paper["passage_max_tokens"], 512)
        self.assertEqual(paper["train_epochs_per_loop"], 2)
        self.assertEqual(paper["temperature"], 0.02)
        self.assertEqual(
            paper["weighted_loss_reduction"], "mean_over_minibatch"
        )
        self.assertTrue(config["research_only"])
        self.assertFalse(config["competition_submission_eligible"])

    def test_runner_uses_fresh_loop_queries_and_paper_pair_builder(self):
        runner = (
            self.root / "solution/scripts/run_paper_flywheel_v1.sh"
        ).read_text()
        self.assertIn('query_tsv="${loop_dir}/queries.tsv"', runner)
        self.assertIn("--mode sample-loop", runner)
        self.assertIn("build_paper_flywheel_pairs.py", runner)
        self.assertIn("--require-completed-training", runner)

    def test_training_entrypoint_preserves_simultaneous_batch_distinction(self):
        trainer = (
            self.root
            / "solution/scripts/train_paper_flywheel_retriever.sh"
        ).read_text()
        self.assertIn('--train_group_size "${train_group_size}"', trainer)
        self.assertIn("--same_dataset_within_batch True", trainer)
        self.assertIn("save_strategy=epoch", trainer)
        self.assertIn('--save_strategy "${save_strategy}"', trainer)
        self.assertIn("ALLOW_SCALED_PAPER_PROFILE", trainer)
        self.assertIn("exact_in_batch_negative_pool", trainer)
        self.assertIn("PAPER_SMOKE_MAX_STEPS", trainer)
        self.assertIn('"${resolved_research}/smoke/"*', trainer)
        self.assertIn('smoke_args+=(--max_steps "${smoke_max_steps}")', trainer)
        self.assertIn(
            'PYTHONPATH="${project_root}/FlagEmbedding${PYTHONPATH:+:${PYTHONPATH}}"',
            trainer,
        )
        self.assertIn('PATH="${project_root}/.venv/bin:${PATH}"', trainer)
        self.assertIn("LRAT_WEIGHT_REDUCTION=paper_mean", trainer)
        self.assertIn('"runtime_weight_reduction":"paper_mean"', trainer)
        self.assertIn("weight_reduction=paper_mean", trainer)


if __name__ == "__main__":
    unittest.main()
