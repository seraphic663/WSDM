#!/usr/bin/env python3
"""Independently audit the aggressive search-stage reweighting artifact."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from solution.src.build_aggressive_search_weight_arm import (
        load_config,
        provenance_distribution,
        read_jsonl,
        sha256_file,
        stage_bucket,
    )
except ModuleNotFoundError:
    from build_aggressive_search_weight_arm import (
        load_config,
        provenance_distribution,
        read_jsonl,
        sha256_file,
        stage_bucket,
    )


def without_weight(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "reweight_rate"}


def audit(
    pairs_path: Path,
    candidate_path: Path,
    provenance_path: Path,
    config_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_expected = manifest.get("rows")
    if not isinstance(rows_expected, int) or rows_expected <= 0:
        raise ValueError("manifest has invalid row count")
    if sha256_file(pairs_path) != manifest["inputs"]["pairs"]["sha256"]:
        raise ValueError("pairs identity mismatch")
    if sha256_file(provenance_path) != manifest["inputs"]["provenance"]["sha256"]:
        raise ValueError("provenance identity mismatch")
    if sha256_file(config_path) != manifest["inputs"]["config"]["sha256"]:
        raise ValueError("config identity mismatch")
    candidate_sha = sha256_file(candidate_path)
    if candidate_sha != manifest["output"]["sha256"]:
        raise ValueError("candidate identity mismatch")

    config = load_config(config_path)
    expected_counts, expected_scale, expected_weights = provenance_distribution(
        provenance_path, config, rows_expected
    )
    counts: Counter[str] = Counter()
    total_weight = 0.0
    changed_rows = 0
    source_iter = read_jsonl(pairs_path)
    candidate_iter = read_jsonl(candidate_path)
    provenance_iter = read_jsonl(provenance_path)
    rows = 0
    while True:
        items = []
        for iterator in (source_iter, candidate_iter, provenance_iter):
            try:
                items.append(next(iterator))
            except StopIteration:
                items.append(None)
        if all(item is None for item in items):
            break
        if any(item is None for item in items):
            raise ValueError("audit inputs have different row counts")
        (_, source), (_, candidate), (_, provenance) = items
        if provenance.get("row_index") != rows:
            raise ValueError(f"provenance row_index mismatch at {rows}")
        if without_weight(source) != without_weight(candidate):
            raise ValueError(f"non-weight field changed at row {rows}")
        bucket = stage_bucket(provenance)
        expected_weight = expected_weights[bucket]
        actual_weight = candidate.get("reweight_rate")
        if not isinstance(actual_weight, (int, float)) or not math.isclose(
            float(actual_weight), expected_weight, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(f"candidate weight mismatch at row {rows}")
        counts[bucket] += 1
        total_weight += float(actual_weight)
        changed_rows += int(
            not math.isclose(
                float(source["reweight_rate"]), float(actual_weight), rel_tol=0, abs_tol=1e-15
            )
        )
        rows += 1

    passed = (
        rows == rows_expected
        and counts == expected_counts
        and math.isclose(total_weight, float(rows), rel_tol=0, abs_tol=1e-6)
        and math.isclose(manifest["stable_scale"], expected_scale, rel_tol=0, abs_tol=1e-15)
        and manifest["normalized_weights"] == expected_weights
        and changed_rows == manifest["changed_rows"]
        and manifest["contract"]["only_modified_field"] == "reweight_rate"
        and manifest["contract"]["locked_test_used"] is False
        and manifest["contract"]["external_data_used"] is False
    )
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "passed": passed,
        "rows": rows,
        "changed_rows": changed_rows,
        "bucket_counts": dict(sorted(counts.items())),
        "total_weight": total_weight,
        "mean_weight": total_weight / rows,
        "normalized_weights": expected_weights,
        "candidate_sha256": candidate_sha,
        "manifest_sha256": sha256_file(manifest_path),
        "locked_test_used": False,
        "external_data_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(
        args.pairs,
        args.candidate,
        args.provenance,
        args.config,
        args.manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
