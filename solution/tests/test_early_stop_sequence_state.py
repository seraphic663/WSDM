import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "scripts/lib/early_stop_sequence_state.sh"


def ready(output: Path, target: int, max_steps: int) -> bool:
    command = (
        f"source {LIBRARY!s}; "
        f"early_stop_model_ready {output!s} {target} {max_steps}"
    )
    return subprocess.run(["bash", "-c", command], check=False).returncode == 0


def touch_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_bytes(b"model")


class EarlyStopSequenceStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.output = Path(self.tempdir.name) / "run"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_intermediate_checkpoint_is_ready(self) -> None:
        touch_model(self.output / "checkpoint-1000")
        self.assertTrue(ready(self.output, 1000, 2000))

    def test_root_model_from_pause_is_not_final(self) -> None:
        touch_model(self.output)
        (self.output / "PAUSED_AT_STEP").write_text(
            "global_step=1000\n", encoding="utf-8"
        )
        self.assertFalse(ready(self.output, 2000, 2000))

    def test_completed_root_model_is_final(self) -> None:
        touch_model(self.output)
        (self.output / "COMPLETED").write_text(
            "completed_at=test\n", encoding="utf-8"
        )
        self.assertTrue(ready(self.output, 2000, 2000))

    def test_stale_pause_marker_blocks_final_readiness(self) -> None:
        touch_model(self.output)
        (self.output / "COMPLETED").write_text(
            "completed_at=test\n", encoding="utf-8"
        )
        (self.output / "PAUSED_AT_STEP").write_text(
            "global_step=1000\n", encoding="utf-8"
        )
        self.assertFalse(ready(self.output, 2000, 2000))


if __name__ == "__main__":
    unittest.main()
