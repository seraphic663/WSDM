import json
import os
import tempfile
import unittest
from pathlib import Path

from solution.src.prune_completed_checkpoint_state import prune_completed_run


class PruneCompletedCheckpointStateTests(unittest.TestCase):
    def make_run(self, root: Path) -> Path:
        run = root / "run"
        run.mkdir()
        (run / "COMPLETED").write_text("ok\n", encoding="utf-8")
        (run / "model.safetensors").write_bytes(b"final")
        for step in (500, 1000):
            checkpoint = run / f"checkpoint-{step}"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(f"model-{step}".encode())
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"global_step": step}), encoding="utf-8"
            )
            (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
            (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
            (checkpoint / "rng_state_0.pth").write_bytes(b"rng")
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        return run

    def test_prunes_only_recovery_state_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.make_run(root)
            result = prune_completed_run(run, allowed_root=root)
            self.assertTrue(result["completed"])
            self.assertEqual(result["removed_bytes"], 42)
            for step in (500, 1000):
                checkpoint = run / f"checkpoint-{step}"
                self.assertTrue((checkpoint / "model.safetensors").is_file())
                self.assertTrue((checkpoint / "trainer_state.json").is_file())
                self.assertTrue((checkpoint / "config.json").is_file())
                self.assertFalse((checkpoint / "optimizer.pt").exists())
                self.assertFalse((checkpoint / "scheduler.pt").exists())
                self.assertFalse((checkpoint / "rng_state_0.pth").exists())
            again = prune_completed_run(run, allowed_root=root)
            self.assertEqual(again, result)

    def test_requires_completed_marker_and_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.make_run(root)
            (run / "COMPLETED").unlink()
            with self.assertRaisesRegex(ValueError, "COMPLETED"):
                prune_completed_run(run, allowed_root=root)
            outside = root.parent / f"{root.name}-outside"
            with self.assertRaisesRegex(ValueError, "outside allowed root"):
                prune_completed_run(outside, allowed_root=root)

    def test_prunes_extra_checkpoint_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.make_run(root)
            backup_root = root / "checkpoint_backups"
            backup = backup_root / "checkpoint-1500"
            backup.mkdir(parents=True)
            (backup / "model.safetensors").write_bytes(b"backup-model")
            (backup / "trainer_state.json").write_text(
                json.dumps({"global_step": 1500}), encoding="utf-8"
            )
            (backup / "optimizer.pt").write_bytes(b"backup-optimizer")

            result = prune_completed_run(
                run,
                allowed_root=root,
                extra_checkpoint_roots=[backup_root],
            )
            self.assertTrue(result["completed"])
            self.assertFalse((backup / "optimizer.pt").exists())
            self.assertTrue((backup / "model.safetensors").is_file())
            self.assertTrue((backup / "trainer_state.json").is_file())
            self.assertEqual(len(result["checkpoint_roots"]), 2)

    def test_allows_only_superseded_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.make_run(root)
            failed = run / "FAILED"
            failed.write_text("old attempt\n", encoding="utf-8")
            completed = run / "COMPLETED"
            os.utime(failed, ns=(1, 1))
            os.utime(completed, ns=(2, 2))
            result = prune_completed_run(
                run,
                allowed_root=root,
                allow_superseded_failure=True,
                dry_run=True,
            )
            self.assertTrue(result["contract"]["superseded_failure_accepted"])

            os.utime(failed, ns=(3, 3))
            with self.assertRaisesRegex(ValueError, "not older"):
                prune_completed_run(
                    run,
                    allowed_root=root,
                    allow_superseded_failure=True,
                    dry_run=True,
                )


if __name__ == "__main__":
    unittest.main()
