import unittest

from solution.src.select_trajectory_quality_gate import select


def gate(passed, mrr, recall, raw_prefix="a"):
    return {
        "bootstrap_samples": 10000,
        "gate": {"passed": passed},
        "comparisons": [
            {
                "label": "checkpoint-500",
                "raw_eval": f"{raw_prefix}-500",
                "mrr": {"delta": 0.001},
                "recall_at_1": {"delta": 0.001},
            },
            {
                "label": "final-1000",
                "raw_eval": f"{raw_prefix}-1000",
                "mrr": {"delta": mrr},
                "recall_at_1": {"delta": recall},
            },
        ],
    }


class TrajectoryQualityGateTests(unittest.TestCase):
    def test_no_pass_stops_full_epoch(self):
        value = select(gate(False, 0.1, 0.1), gate(False, 0.2, 0.2))
        self.assertFalse(value["full_epoch_authorized"])
        self.assertIsNone(value["selected_arm"])

    def test_higher_final_mrr_wins(self):
        value = select(gate(True, 0.02, 0.03), gate(True, 0.01, 0.05))
        self.assertEqual(value["selected_arm"], "B")

    def test_exact_tie_prefers_non_deleting_c(self):
        value = select(gate(True, 0.02, 0.03), gate(True, 0.02, 0.03))
        self.assertEqual(value["selected_arm"], "C")

    def test_mismatched_arm_a_is_rejected(self):
        with self.assertRaises(ValueError):
            select(gate(True, 0.02, 0.03, "a"), gate(True, 0.02, 0.03, "other"))


if __name__ == "__main__":
    unittest.main()
