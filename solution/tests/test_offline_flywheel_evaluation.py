import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from solution.src.analyze_offline_flywheel_effects import analyze
from solution.src.evaluate_offline_flywheel_loop import compare_loop


def write_eval(path: Path, ranks: list[int]) -> None:
    details = [
        {
            "row_index": index,
            "query_sha256": hashlib.sha256(f"q-{index}".encode()).hexdigest(),
            "best_positive_rank": rank,
        }
        for index, rank in enumerate(ranks)
    ]
    path.write_text(
        json.dumps({"input": "/dev.jsonl", "rows": len(ranks), "details": details}),
        encoding="utf-8",
    )


class OfflineFlywheelEvaluationTests(unittest.TestCase):
    def test_gate_and_difficulty_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent.json"
            control = root / "control.json"
            candidate = root / "candidate.json"
            write_eval(parent, [1, 3, 7, 20])
            write_eval(control, [1, 4, 6, 15])
            write_eval(candidate, [1, 2, 5, 10])
            gate = compare_loop(
                parent_eval=parent,
                control_eval=control,
                candidate_eval=candidate,
                output=root / "gate.json",
                bootstrap_samples=1000,
                seed=1,
            )
            self.assertIn("candidate_vs_control", gate["comparisons"])
            effects = analyze(
                parent_eval=parent,
                control_eval=control,
                candidate_eval=candidate,
                output=root / "effects.json",
            )
            self.assertEqual(sum(effects["parent_bucket_counts"].values()), 4)
            self.assertEqual(
                effects["candidate_vs_control"]["overall"]["improved_rank"], 3
            )


if __name__ == "__main__":
    unittest.main()
