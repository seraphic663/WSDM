import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from analyze_negative_reliability import (  # noqa: E402
    cluster_difference_ci,
    cluster_ratio_ci,
    exposure_bucket,
    rank_bucket,
    search_phase,
)


class ReliabilityAnalysisTest(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(rank_bucket(1), "1")
        self.assertEqual(rank_bucket(5), "3-5")
        self.assertEqual(search_phase(0.2), "early")
        self.assertEqual(search_phase(0.6), "middle")
        self.assertEqual(search_phase(1.0), "late")
        self.assertEqual(exposure_bucket(6), "5-6")

    def test_cluster_ratio_is_candidate_weighted_with_query_resampling(self):
        counts = {"q1": [1, 2], "q2": [0, 2]}
        value = cluster_ratio_ci(counts, ["q1", "q2"], samples=1000, seed=1)
        self.assertAlmostEqual(value["estimate"], 0.25)
        self.assertEqual(value["candidate_occurrences"], 4)
        self.assertEqual(value["queries_with_candidates"], 2)

    def test_cluster_difference(self):
        left = {"q1": [2, 2], "q2": [2, 2], "q3": [2, 2]}
        right = {"q1": [0, 2], "q2": [0, 2], "q3": [0, 2]}
        value = cluster_difference_ci(left, right, ["q1", "q2", "q3"], samples=1000, seed=2)
        self.assertEqual(value["difference"], 1.0)
        self.assertTrue(value["excludes_zero"])


if __name__ == "__main__":
    unittest.main()
