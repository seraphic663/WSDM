import json
import tempfile
import unittest
from pathlib import Path

from solution.src.prune_short_training_run import INFERENCE_FILES, prune


def write_eval(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "rows": 1500,
                "input": "/data/dev.jsonl",
                "details": [{"rank": 1}] * 1500,
            }
        ),
        encoding="utf-8",
    )


class PruneShortTrainingRunTests(unittest.TestCase):
    def test_prune_preserves_step_500_and_final_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            checkpoint_500 = root / "checkpoint-500"
            checkpoint_1000 = root / "checkpoint-1000"
            checkpoint_500.mkdir(parents=True)
            checkpoint_1000.mkdir()
            (root / "COMPLETED").write_text("ok\n", encoding="utf-8")
            for checkpoint, weight in ((checkpoint_500, b"five"), (checkpoint_1000, b"final")):
                for name in INFERENCE_FILES:
                    (checkpoint / name).write_bytes(weight if name == "model.safetensors" else b"metadata")
                (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
                (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
                (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
                (checkpoint / "training_args.bin").write_bytes(b"args")
                (checkpoint / "rng_state_0.pth").write_bytes(b"rng")
            (root / "model.safetensors").write_bytes(b"final")
            eval_500 = root / "eval_500.json"
            eval_1000 = root / "eval_1000.json"
            write_eval(eval_500)
            write_eval(eval_1000)

            result = prune(root, eval_500, eval_1000)
            self.assertTrue((checkpoint_500 / "model.safetensors").is_file())
            self.assertFalse((checkpoint_500 / "optimizer.pt").exists())
            self.assertFalse(checkpoint_1000.exists())
            self.assertTrue((root / "SHORT_RUN_PRUNE_MANIFEST.json").is_file())
            self.assertEqual(
                result["weights"]["checkpoint_1000_sha256"],
                result["weights"]["final_root_sha256"],
            )
            self.assertFalse(result["contract"]["locked_test_used"])

            repeated = prune(root, eval_500, eval_1000)
            self.assertEqual(repeated["weights"], result["weights"])


if __name__ == "__main__":
    unittest.main()
