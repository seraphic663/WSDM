import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from solution.src.freeze_inference_model import REQUIRED, freeze_model, rename_noreplace


class FreezeInferenceModelTest(unittest.TestCase):
    def test_atomic_copy_hashes_and_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (source / "COMPLETED").write_text("completed\n")
            for index, name in enumerate(REQUIRED):
                (source / name).write_bytes(f"file-{index}".encode())
            report = freeze_model(source, target, lambda _: {"pid": 123, "device": "cpu", "class": "TestModel"})
            self.assertEqual(report["new_cpu_process_load"]["device"], "cpu")
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o555)
            self.assertEqual(stat.S_IMODE((target / "model.safetensors").stat().st_mode), 0o444)
            self.assertTrue((target / "FREEZE_MANIFEST.json").is_file())
            manifest = json.loads((target / "FREEZE_MANIFEST.json").read_text())
            self.assertEqual(len(manifest["files"]), len(REQUIRED))

    def test_refuses_existing_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"; source.mkdir()
            target = root / "target"; target.mkdir()
            with self.assertRaises(FileExistsError):
                freeze_model(source, target, lambda _: {})

    def test_distributed_filesystem_fallback_reserves_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"; source.mkdir()
            target = root / "target"
            fake = mock.Mock(return_value=-1)
            with mock.patch("solution.src.freeze_inference_model.ctypes.CDLL") as library:
                library.return_value.renameat2 = fake
                with mock.patch("solution.src.freeze_inference_model.ctypes.get_errno", return_value=22):
                    rename_noreplace(source, target)
            self.assertTrue(target.is_dir())
            self.assertFalse(source.exists())


if __name__ == "__main__":
    unittest.main()
