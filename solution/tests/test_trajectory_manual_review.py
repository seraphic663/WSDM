import json
import tempfile
import unittest
from pathlib import Path

from solution.src.apply_trajectory_manual_review import apply_review


class TrajectoryManualReviewTests(unittest.TestCase):
    def test_review_requires_exact_coverage_and_reports_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            review = root / "review.jsonl"
            output = root / "out"
            samples.write_text(
                "\n".join(
                    [
                        json.dumps({"sample_id": "a", "stratum": "negative"}),
                        json.dumps({"sample_id": "b", "stratum": "negative"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            review.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "sample_id": "a",
                                "manual_label": "bad_positive",
                                "manual_reason": "explicitly says irrelevant",
                            }
                        ),
                        json.dumps(
                            {
                                "sample_id": "b",
                                "manual_label": "keep",
                                "manual_reason": "extracts the requested fact",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            result = apply_review(samples, review, output)
            self.assertEqual(result["sample_rows"], 2)
            self.assertEqual(
                result["per_stratum"]["negative"]["bad_positive_precision"], 0.5
            )

    def test_incomplete_review_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            review = root / "review.jsonl"
            samples.write_text(
                json.dumps({"sample_id": "a", "stratum": "control"}) + "\n",
                encoding="utf-8",
            )
            review.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                apply_review(samples, review, root / "out")


if __name__ == "__main__":
    unittest.main()
