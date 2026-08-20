import unittest

from solution.src.select_multiseed_trajectory_gate import aggregate


def gate(passed: bool, delta: float) -> dict:
    comparisons = []
    for label in ("checkpoint-500", "final-1000"):
        comparisons.append(
            {
                "label": label,
                "recall_at_1": {"delta": delta},
                "recall_at_5": {"delta": delta},
                "recall_at_10": {"delta": delta},
                "mrr": {"delta": delta},
            }
        )
    return {
        "bootstrap_samples": 10000,
        "gate": {"passed": passed},
        "comparisons": comparisons,
    }


class MultiseedTrajectoryGateTests(unittest.TestCase):
    def test_two_passing_positive_seeds_authorize_full_epoch(self):
        result = aggregate(
            [(1, gate(True, 0.01)), (2, gate(True, 0.02)), (3, gate(False, -0.001))]
        )
        self.assertTrue(result["full_epoch_authorized"])
        self.assertEqual(result["selected_arm"], "search_idx_soft_v2")

    def test_positive_means_do_not_override_insufficient_seed_passes(self):
        result = aggregate(
            [(1, gate(True, 0.02)), (2, gate(False, 0.01)), (3, gate(False, 0.01))]
        )
        self.assertFalse(result["full_epoch_authorized"])

    def test_negative_mean_blocks_even_two_seed_passes(self):
        result = aggregate(
            [(1, gate(True, 0.01)), (2, gate(True, 0.01)), (3, gate(False, -0.05))]
        )
        self.assertFalse(result["full_epoch_authorized"])


if __name__ == "__main__":
    unittest.main()
