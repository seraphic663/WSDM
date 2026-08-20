#!/usr/bin/env python3
"""Remove cross-row false negatives for identical queries without rewriting source."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import tempfile
from pathlib import Path


def require_aligned_ids(row: dict, line_number: int) -> tuple[list, list]:
    positives = row.get("pos")
    positive_ids = row.get("pos_id")
    negatives = row.get("neg")
    negative_ids = row.get("neg_id")
    if not isinstance(row.get("query"), str):
        raise ValueError(f"line {line_number}: query must be a string")
    for name, value in (
        ("pos", positives),
        ("pos_id", positive_ids),
        ("neg", negatives),
        ("neg_id", negative_ids),
    ):
        if not isinstance(value, list):
            raise ValueError(f"line {line_number}: {name} must be a list")
    if not positives or len(positives) != len(positive_ids):
        raise ValueError(f"line {line_number}: pos and pos_id are empty or misaligned")
    if not negatives or len(negatives) != len(negative_ids):
        raise ValueError(f"line {line_number}: neg and neg_id are empty or misaligned")
    return positive_ids, negative_ids


def clean_conflicts(input_path: Path, output_path: Path, report_path: Path) -> dict:
    for path in (output_path, report_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    positives_by_query: dict[str, set[str]] = collections.defaultdict(set)
    query_rows: collections.Counter[str] = collections.Counter()
    input_digest = hashlib.sha256()
    row_count = 0
    with input_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            input_digest.update(raw_line)
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            positive_ids, _ = require_aligned_ids(row, line_number)
            query = row["query"]
            positives_by_query[query].update(map(str, positive_ids))
            query_rows[query] += 1
            row_count += 1
    if row_count == 0:
        raise ValueError("input contains no rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_fd, output_temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.tmp.", dir=output_path.parent
    )
    report_fd, report_temp_name = tempfile.mkstemp(
        prefix=f".{report_path.name}.tmp.", dir=report_path.parent
    )
    os.close(report_fd)
    output_temp = Path(output_temp_name)
    report_temp = Path(report_temp_name)
    output_digest = hashlib.sha256()
    rows_changed = 0
    negatives_removed = 0
    rows_below_five_negatives = 0
    conflict_queries: set[str] = set()
    try:
        with input_path.open("r", encoding="utf-8") as source, os.fdopen(
            output_fd, "wb"
        ) as destination:
            for line_number, line in enumerate(source, 1):
                row = json.loads(line)
                positive_ids, negative_ids = require_aligned_ids(row, line_number)
                positive_union = positives_by_query[row["query"]]
                keep = [
                    index
                    for index, negative_id in enumerate(negative_ids)
                    if str(negative_id) not in positive_union
                ]
                removed = len(negative_ids) - len(keep)
                if removed:
                    conflict_queries.add(row["query"])
                    rows_changed += 1
                    negatives_removed += removed
                    row["neg"] = [row["neg"][index] for index in keep]
                    row["neg_id"] = [negative_ids[index] for index in keep]
                if not row["neg"]:
                    raise ValueError(f"line {line_number}: cleaning removed every negative")
                if len(row["neg"]) < 5:
                    rows_below_five_negatives += 1
                encoded = (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                destination.write(encoded)
                output_digest.update(encoded)

        report = {
            "input": str(input_path.resolve()),
            "output": str(output_path.resolve()),
            "rows": row_count,
            "unique_queries": len(query_rows),
            "duplicate_query_groups": sum(value > 1 for value in query_rows.values()),
            "duplicate_extra_rows": sum(value - 1 for value in query_rows.values()),
            "conflict_query_groups": len(conflict_queries),
            "rows_changed": rows_changed,
            "negatives_removed": negatives_removed,
            "rows_below_five_negatives": rows_below_five_negatives,
            "input_sha256": input_digest.hexdigest(),
            "output_sha256": output_digest.hexdigest(),
        }
        report_temp.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(output_temp, output_path)
        os.replace(report_temp, report_path)
        return report
    except BaseException:
        output_temp.unlink(missing_ok=True)
        report_temp.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            clean_conflicts(args.input, args.output, args.report),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
