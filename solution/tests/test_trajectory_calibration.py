import unittest

from solution.src.sample_trajectory_calibration import select_samples, stable_strata


def record(index, **event_overrides):
    event = {
        "answer_token_subset": True,
        "trajectory_steps": 20,
        "next_tool_name": "output",
        "retrieved_rank": 1,
        "negative_cues": [],
        "reasoning_len": 20,
    }
    event.update(event_overrides)
    return {
        "bucket": "stable",
        "row_index": index,
        "source_line": index + 1,
        "query": f"q{index}",
        "pos_id": str(index),
        "reasoning_len": 20,
        "reweight_rate": 1.0,
        "negative_count": 5,
        "event": event,
    }


class TrajectoryCalibrationTests(unittest.TestCase):
    def test_strata_are_independent_flags(self):
        value = record(
            1,
            answer_token_subset=False,
            trajectory_steps=200,
            next_tool_name="search",
            retrieved_rank=9,
            negative_cues=["explicit_not_relevant"],
        )
        self.assertEqual(
            stable_strata(value, 126, 8),
            [
                "answer_unmatched",
                "long_trajectory",
                "continue_search",
                "low_rank",
                "explicit_negative_reasoning",
                "strict_negative_combo",
            ],
        )

    def test_selection_is_deterministic_and_unique(self):
        values = [
            record(i, negative_cues=["explicit_not_relevant"]) for i in range(4)
        ] + [record(i) for i in range(4, 8)]
        first = select_samples(
            values,
            samples_per_stratum=2,
            seed=7,
            long_steps=126,
            low_rank=8,
        )
        second = select_samples(
            values,
            samples_per_stratum=2,
            seed=7,
            long_steps=126,
            low_rank=8,
        )
        self.assertEqual(first, second)
        indexes = [item["row_index"] for item in first]
        self.assertEqual(len(indexes), len(set(indexes)))
        self.assertEqual(
            {item["stratum"] for item in first},
            {"explicit_negative_reasoning", "control"},
        )

    def test_targeted_strict_negative_combo(self):
        values = [
            record(
                1,
                answer_token_subset=False,
                negative_cues=["explicit_not_relevant"],
            ),
            record(
                2,
                retrieved_rank=9,
                negative_cues=["explicit_not_relevant"],
            ),
            record(3, negative_cues=["explicit_not_relevant"]),
        ]
        selected = select_samples(
            values,
            samples_per_stratum=10,
            seed=7,
            long_steps=126,
            low_rank=8,
            only_strata={"strict_negative_combo"},
        )
        self.assertEqual({item["row_index"] for item in selected}, {1, 2})
        self.assertTrue(
            all(item["stratum"] == "strict_negative_combo" for item in selected)
        )


if __name__ == "__main__":
    unittest.main()
