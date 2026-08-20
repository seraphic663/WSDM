#!/usr/bin/env python3
"""Checks for exact-query cross-row false-negative cleaning."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from clean_query_conflicts import clean_conflicts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source, output, report = root / "source.jsonl", root / "clean.jsonl", root / "report.json"
        rows = [
            {
                "query": "same",
                "pos": ["doc-a"],
                "pos_id": ["a"],
                "neg": ["doc-b", "doc-c", "doc-d", "doc-e", "doc-f"],
                "neg_id": ["b", "c", "d", "e", "f"],
                "reweight_rate": 1.0,
            },
            {
                "query": "same",
                "pos": ["doc-b"],
                "pos_id": ["b"],
                "neg": ["doc-a", "doc-g", "doc-h", "doc-i", "doc-j"],
                "neg_id": ["a", "g", "h", "i", "j"],
                "reweight_rate": 2.0,
            },
            {
                "query": "different",
                "pos": ["doc-x"],
                "pos_id": ["x"],
                "neg": ["doc-a", "doc-k", "doc-l", "doc-m", "doc-n"],
                "neg_id": ["a", "k", "l", "m", "n"],
                "reweight_rate": 3.0,
            },
        ]
        write_jsonl(source, rows)
        result = clean_conflicts(source, output, report)
        cleaned = [json.loads(line) for line in output.read_text().splitlines()]

        assert cleaned[0]["neg_id"] == ["c", "d", "e", "f"]
        assert cleaned[1]["neg_id"] == ["g", "h", "i", "j"]
        assert cleaned[2]["neg_id"] == ["a", "k", "l", "m", "n"]
        assert [row["reweight_rate"] for row in cleaned] == [1.0, 2.0, 3.0]
        assert result["conflict_query_groups"] == 1
        assert result["rows_changed"] == 2
        assert result["negatives_removed"] == 2
        assert result["rows_below_five_negatives"] == 2
        assert json.loads(report.read_text()) == result

        try:
            clean_conflicts(source, output, root / "other-report.json")
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing output was overwritten")
    print("query conflict cleaning: ok")


if __name__ == "__main__":
    main()
