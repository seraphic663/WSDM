import json
import tempfile
import unittest
from pathlib import Path

from solution.src.prepare_strategy_suite import aligned, percentile
from solution.src.finalize_strategy_suite import validate_and_classify
from solution.src.run_strategy_gpu_suite import parse_train_log


class StrategySuiteTest(unittest.TestCase):
    def test_config_is_short_and_covers_expected_matrix(self):
        config = json.loads(Path("solution/configs/strategy_suite_20260718.json").read_text())
        self.assertEqual(config["max_steps"], 10)
        self.assertEqual(len(config["training"]), 7)
        self.assertEqual(len(config["evaluations"]), 4)
        self.assertEqual({item["group_size"] for item in config["training"]}, {6, 10})
        self.assertEqual({item["learning_rate"] for item in config["training"]}, {"1e-6", "5e-7", "3e-7"})

    def test_data_alignment_and_percentile_helpers(self):
        aligned({"query": "q", "pos": ["p"], "pos_id": [1], "neg": ["n"], "neg_id": [2]}, 1)
        with self.assertRaises(ValueError):
            aligned({"query": "q", "pos": ["p"], "pos_id": [], "neg": ["n"], "neg_id": [2]}, 1)
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)

    def test_train_log_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.log"
            path.write_text("1/10\n10/10\n{'train_runtime': 42.5, 'x': 1, 'train_loss': 0.75}\n")
            result = parse_train_log(path)
            self.assertEqual(result["runtime_seconds"], 42.5)
            self.assertEqual(result["train_loss"], 0.75)
            self.assertEqual(result["max_step"], 10)
            self.assertEqual(result["traceback_count"], 0)

    def test_final_classification_preserves_official_corpus_limit(self):
        training = []
        for name in {
            "b_original_g6_lr1e6", "b_cleaned_g6_lr1e6", "shared_weighted_g6_lr1e6",
            "c_unit_g6_lr1e6", "e_group10_lr1e6", "f_group6_lr5e7", "f_group6_lr3e7",
        }:
            training.append({
                "name": name, "status": "PASS", "return_code": 0, "max_step": 10,
                "traceback_count": 0, "oom_count": 0, "nccl_error_count": 0,
            })
        evaluations = [
            {"name": name, "status": "PASS", "return_code": 0}
            for name in {
                "a_m00_query_isolated_diag32", "d_m00_shielded_existing_neg16",
                "g_tail_avg2_diag32", "g_tail_avg3_diag32",
            }
        ]
        cpu = {"official_corpus_present": False, "outputs": {"a_train64": {"rows": 64}, "a_diag32": {"rows": 32}}}
        validation = {"H_sampling": {"all_rows_expose_every_proven_positive": True}}
        outcomes = validate_and_classify(cpu, validation, {"simulation_only": True, "training": training, "evaluations": evaluations})
        self.assertEqual(outcomes["D"]["status"], "PARTIAL")
        self.assertEqual(outcomes["I"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
