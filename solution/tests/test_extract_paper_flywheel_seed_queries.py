import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from solution.src.extract_paper_flywheel_seed_queries import (
    build_source_pool,
    extract,
    sample_loop_queries,
)


class ExtractPaperFlywheelSeedQueriesTests(unittest.TestCase):
    def test_union_and_conflict_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "trajectories.tar.gz"
            rows = {
                "trajectories/a/0.json": {
                    "query_id": 0,
                    "raw_messages": [{"role": "user", "content": "Question zero"}],
                },
                "trajectories/a/1.json": {
                    "query_id": 1,
                    "raw_messages": [{"role": "user", "content": "Question one"}],
                },
                "trajectories/b/1.json": {
                    "query_id": "1",
                    "question": "Question   one",
                },
            }
            with tarfile.open(archive, "w:gz") as bundle:
                for name, value in rows.items():
                    data = json.dumps(value).encode()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    bundle.addfile(info, io.BytesIO(data))
            output = root / "queries.tsv"
            manifest = root / "manifest.json"
            result = extract(
                archive, output, manifest, expected_count=2
            )
            self.assertEqual(output.read_text(), "0\tQuestion zero\n1\tQuestion one\n")
            self.assertEqual(result["unique_query_count"], 2)
            self.assertEqual(result["duplicate_observations"], 1)

            conflict_archive = root / "conflict.tar.gz"
            rows["trajectories/b/1.json"]["question"] = "Different"
            with tarfile.open(conflict_archive, "w:gz") as bundle:
                for name, value in rows.items():
                    data = json.dumps(value).encode()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    bundle.addfile(info, io.BytesIO(data))
            with self.assertRaisesRegex(ValueError, "conflict"):
                extract(
                    conflict_archive,
                    root / "conflict.tsv",
                    root / "conflict.json",
                    expected_count=2,
                )

    def test_source_qa_completes_missing_archive_queries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "trajectories.tar.gz"
            value = {
                "query_id": 1,
                "query": "Question one",
            }
            with tarfile.open(archive, "w:gz") as bundle:
                data = json.dumps(value).encode()
                info = tarfile.TarInfo("trajectories/a/1.json")
                info.size = len(data)
                bundle.addfile(info, io.BytesIO(data))
            source = root / "InfoSeekQA.jsonl"
            source.write_text(
                '{"question":"Question zero","answer":"0"}\n'
                '{"question":"Question one","answer":"1"}\n',
                encoding="utf-8",
            )
            output = root / "queries.tsv"
            manifest = root / "manifest.json"
            result = extract(
                archive,
                output,
                manifest,
                expected_count=2,
                source_qa_jsonl=source,
                source_revision="fixed",
            )
            self.assertEqual(
                output.read_text(), "0\tQuestion zero\n1\tQuestion one\n"
            )
            self.assertEqual(result["archive_unique_query_count"], 1)
            self.assertEqual(result["source_qa"]["archive_subset_validated"], 1)

    def test_source_pool_and_loop_sampling_are_recorded_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "trajectories.tar.gz"
            value = {"query_id": 1, "query": "Question one"}
            with tarfile.open(archive, "w:gz") as bundle:
                data = json.dumps(value).encode()
                info = tarfile.TarInfo("trajectories/a/1.json")
                info.size = len(data)
                bundle.addfile(info, io.BytesIO(data))
            source = root / "InfoSeekQA.jsonl"
            source.write_text(
                "".join(
                    json.dumps({"question": f"Question {name}"}) + "\n"
                    for name in ("zero", "one", "two", "three")
                ),
                encoding="utf-8",
            )
            pool = root / "pool.tsv"
            pool_manifest = root / "pool.json"
            result = build_source_pool(
                archive,
                source,
                pool,
                pool_manifest,
                source_revision="fixed",
                minimum_count=4,
            )
            self.assertEqual(result["pool_rows"], 4)
            self.assertTrue(result["external_data_used"])

            first = root / "loop0.tsv"
            first_manifest = root / "loop0.json"
            second = root / "loop0-copy.tsv"
            second_manifest = root / "loop0-copy.json"
            sample_loop_queries(
                pool,
                first,
                first_manifest,
                count=2,
                seed=2025,
                loop_number=0,
            )
            sample_loop_queries(
                pool,
                second,
                second_manifest,
                count=2,
                seed=2025,
                loop_number=0,
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                json.loads(first_manifest.read_text())["output_sha256"],
                json.loads(second_manifest.read_text())["output_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
