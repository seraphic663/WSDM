import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from analyze_reliability_arm_grid import aggregate_contrast, metric_values  # noqa: E402


class ArmGridAnalysisTest(unittest.TestCase):
    def test_metric_values(self):
        values = metric_values([1, 2, 11])
        self.assertEqual(values["recall_at_1"], [1.0, 0.0, 0.0])
        self.assertEqual(values["recall_at_10"], [1.0, 1.0, 0.0])

    def test_aggregate_contrast_averages_seeds_before_query_bootstrap(self):
        left = {1: [1, 1, 2], 2: [1, 2, 2], 3: [1, 1, 2]}
        right = {1: [2, 2, 3], 2: [2, 3, 3], 3: [2, 2, 3]}
        result = aggregate_contrast(left, right, samples=1000, seed=1)
        self.assertGreater(result["metrics"]["mrr"]["delta"], 0)
        self.assertEqual(result["metrics"]["mrr"]["nonnegative_seeds"], 3)
        self.assertEqual(result["queries"], 3)


if __name__ == "__main__":
    unittest.main()
