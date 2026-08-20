import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from solution.src.build_unit_weight_pairs import build


class BuildUnitWeightPairsTests(unittest.TestCase):
    def test_only_reweight_rate_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            output = root / "unit.jsonl"
            manifest = root / "manifest.json"
            rows = [
                {"query": "q1", "pos": ["p1"], "neg": ["n1"], "reweight_rate": 0.5},
                {"query": "q2", "pos": ["p2"], "neg": ["n2"], "reweight_rate": 1.0},
                {"query": "q3", "pos": ["p3"], "neg": ["n3"], "reweight_rate": 2.0},
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            result = build(
                source,
                output,
                manifest,
                expected_input_sha256=source_sha,
                expected_rows=3,
            )
            actual = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            for original, candidate in zip(rows, actual):
                self.assertEqual(
                    {key: value for key, value in candidate.items() if key != "reweight_rate"},
                    {key: value for key, value in original.items() if key != "reweight_rate"},
                )
                self.assertEqual(candidate["reweight_rate"], 1.0)
            self.assertEqual(result["changed_rows"], 2)
            self.assertEqual(result["unit_reweight_rate"]["sum"], 3.0)
            self.assertEqual(
                result["output"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest()
            )

    def test_refuses_wrong_input_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                json.dumps({"query": "q", "reweight_rate": 1.0}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                build(
                    source,
                    root / "unit.jsonl",
                    root / "manifest.json",
                    expected_input_sha256="0" * 64,
                    expected_rows=1,
                )


if __name__ == "__main__":
    unittest.main()
