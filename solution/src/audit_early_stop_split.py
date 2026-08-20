#!/usr/bin/env python3
"""Audit the legacy dev500, literal source tail, and proposed early-stop split."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import statistics
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any


WS = re.compile(r"\s+")


def norm_query(value: str) -> str:
    return WS.sub(" ", value.strip()).casefold()


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
    }


def row_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "reasoning_len": describe([float(row["reasoning_len"]) for row in rows]),
        "reweight_rate": describe([float(row["reweight_rate"]) for row in rows]),
        "negative_count": describe([float(len(row["neg"])) for row in rows]),
        "query_characters": describe([float(len(row["query"])) for row in rows]),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(source: Path, legacy_dev: Path, proposed_manifest: Path) -> dict[str, Any]:
    legacy_lines = legacy_dev.read_bytes().splitlines(keepends=True)
    legacy_rows = [json.loads(line) for line in legacy_lines]
    legacy_keys = {norm_query(row["query"]) for row in legacy_rows}
    legacy_line_counts = collections.Counter(legacy_lines)

    source_group_counts: collections.Counter[str] = collections.Counter()
    exact_matches = 0
    source_rows = 0
    tail: deque[tuple[int, dict[str, Any]]] = deque(maxlen=500)
    with source.open("rb") as handle:
        for source_rows, line in enumerate(handle, 1):
            row = json.loads(line)
            source_group_counts[norm_query(row["query"])] += 1
            tail.append((source_rows, row))
            if legacy_line_counts[line] > 0:
                exact_matches += 1
                legacy_line_counts[line] -= 1

    tail_rows = [row for _, row in tail]
    tail_keys = [norm_query(row["query"]) for row in tail_rows]
    tail_counts = collections.Counter(tail_keys)
    appeared_earlier_keys = {
        key for key, count in tail_counts.items() if source_group_counts[key] > count
    }
    proposed = json.loads(proposed_manifest.read_text(encoding="utf-8"))
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "source": {
            "path": str(source),
            "rows": source_rows,
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
            "normalized_query_groups_including_empty": len(source_group_counts),
        },
        "legacy_dev500": {
            "path": str(legacy_dev),
            "rows": len(legacy_rows),
            "bytes": legacy_dev.stat().st_size,
            "sha256": sha256(legacy_dev),
            "unique_normalized_queries": len(legacy_keys),
            "exact_source_record_matches": exact_matches,
            "source_rows_sharing_these_queries": sum(source_group_counts[key] for key in legacy_keys),
            "query_groups_with_multiple_source_rows": sum(source_group_counts[key] > 1 for key in legacy_keys),
            "extra_source_rows_beyond_selected_records": sum(source_group_counts[key] for key in legacy_keys) - len(legacy_rows),
            "training_overlap_interpretation": "all legacy dev queries occur in the full training source",
            "row_distributions": row_stats(legacy_rows),
        },
        "literal_last_500_source_rows": {
            "source_line_start": tail[0][0],
            "source_line_end": tail[-1][0],
            "rows": len(tail_rows),
            "unique_normalized_queries": len(tail_counts),
            "within_tail_duplicate_rows_by_query": len(tail_rows) - len(tail_counts),
            "query_groups_already_seen_before_tail": len(appeared_earlier_keys),
            "tail_rows_whose_query_was_seen_before_tail": sum(tail_counts[key] for key in appeared_earlier_keys),
            "valid_query_disjoint_holdout": False,
            "row_distributions": row_stats(tail_rows),
        },
        "proposed_early_stop_v1": {
            "manifest_path": str(proposed_manifest),
            "source_sha256_matches": proposed["source"]["sha256"] == sha256(source),
            "train_rows": proposed["train"]["rows"],
            "dev_query_groups": proposed["dev"]["query_groups"],
            "test_query_groups": proposed["test"]["query_groups"],
            "source_rows_excluded": proposed["train"]["source_rows_excluded"],
            "normalized_query_overlap": proposed["train"]["normalized_query_overlap_with_dev_or_test"],
            "dev_candidates_per_query": proposed["dev"]["candidates_per_query"],
            "test_candidates_per_query": proposed["test"]["candidates_per_query"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--legacy-dev", required=True, type=Path)
    parser.add_argument("--proposed-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.source, args.legacy_dev, args.proposed_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
