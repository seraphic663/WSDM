import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "b/reproduction" / "run_public_m10_reproduction.py"
SPEC = importlib.util.spec_from_file_location("run_public_m10_reproduction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PublicM10ReproductionTests(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value"
            path.write_bytes(b"abc")
            self.assertEqual(
                MODULE.sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_immutable_identities(self):
        self.assertEqual(len(MODULE.REPO_COMMIT), 40)
        self.assertEqual(len(MODULE.HF_COMMIT), 40)
        self.assertEqual(len(MODULE.M10_SHA256), 64)
        self.assertEqual(MODULE.EXPECTED_METRICS["recall_at_1"], 0.63)

    def test_fetch_json_uses_local_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            path.write_text('{"sha":"fixed"}\n', encoding="utf-8")
            self.assertEqual(
                MODULE.fetch_json("https://invalid.example.test", path),
                {"sha": "fixed"},
            )


if __name__ == "__main__":
    unittest.main()
