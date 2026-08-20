import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from solution.src.remove_rebuildable_artifacts import remove_rebuildable_artifacts


class RemoveRebuildableArtifactsTests(unittest.TestCase):
    def make_case(self, root: Path) -> tuple[Path, Path, Path, Path]:
        allowed = root / "data"
        allowed.mkdir()
        evidence = root / "builder.py"
        evidence.write_text("# rebuild\n", encoding="utf-8")
        target = allowed / "large.jsonl"
        target.write_bytes(b"rebuildable")
        plan = root / "plan.json"
        plan.write_text(
            json.dumps(
                {
                    "targets": [
                        {
                            "path": "data/large.jsonl",
                            "bytes": target.stat().st_size,
                            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                            "reason": "closed diagnostic",
                            "rebuild_evidence": ["builder.py"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return allowed, evidence, target, plan

    def test_dry_run_then_remove_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowed, evidence, target, plan = self.make_case(root)
            manifest = root / "cleanup.json"
            preview = remove_rebuildable_artifacts(
                repo_root=root,
                allowed_root=allowed,
                plan_path=plan,
                manifest_path=manifest,
                dry_run=True,
            )
            self.assertFalse(preview["completed"])
            self.assertTrue(target.is_file())
            self.assertFalse(manifest.exists())

            result = remove_rebuildable_artifacts(
                repo_root=root,
                allowed_root=allowed,
                plan_path=plan,
                manifest_path=manifest,
            )
            self.assertTrue(result["completed"])
            self.assertEqual(result["removed_bytes"], len(b"rebuildable"))
            self.assertFalse(target.exists())
            self.assertTrue(evidence.is_file())
            again = remove_rebuildable_artifacts(
                repo_root=root,
                allowed_root=allowed,
                plan_path=plan,
                manifest_path=manifest,
            )
            self.assertEqual(again, result)

    def test_rejects_escape_and_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowed, _, _, plan = self.make_case(root)
            manifest = root / "cleanup.json"
            value = json.loads(plan.read_text(encoding="utf-8"))
            value["targets"][0]["path"] = "../outside"
            plan.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "safe relative"):
                remove_rebuildable_artifacts(
                    repo_root=root,
                    allowed_root=allowed,
                    plan_path=plan,
                    manifest_path=manifest,
                    dry_run=True,
                )

            target = allowed / "large.jsonl"
            value = json.loads(plan.read_text(encoding="utf-8"))
            value["targets"][0]["path"] = "data/large.jsonl"
            value["targets"][0]["sha256"] = "0" * 64
            plan.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                remove_rebuildable_artifacts(
                    repo_root=root,
                    allowed_root=allowed,
                    plan_path=plan,
                    manifest_path=manifest,
                    dry_run=True,
                )


if __name__ == "__main__":
    unittest.main()
