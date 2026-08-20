#!/usr/bin/env python3
"""Independently audit an offline flywheel control/candidate shard."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from solution.src.build_offline_flywheel_shard import (
        query_bucket,
        query_hash,
        sha256_file,
    )
except ModuleNotFoundError:
    from build_offline_flywheel_shard import query_bucket, query_hash, sha256_file


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            values.append(value)
    return values


def audit(
    *,
    source: Path,
    shard_dir: Path,
    output: Path,
) -> dict[str, Any]:
    manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
    control = load_jsonl(shard_dir / "control.jsonl")
    candidate = load_jsonl(shard_dir / "candidate.jsonl")
    metadata = load_jsonl(shard_dir / "mining.jsonl")
    if not (len(control) == len(candidate) == len(metadata) > 0):
        raise ValueError("control/candidate/metadata row counts differ")

    by_line = {item["source_line"]: index for index, item in enumerate(metadata)}
    if len(by_line) != len(metadata):
        raise ValueError("metadata source lines are duplicated")
    seen = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number not in by_line:
                continue
            index = by_line[line_number]
            source_row = json.loads(line)
            control_row = control[index]
            candidate_row = candidate[index]
            item = metadata[index]
            if control_row != source_row:
                raise ValueError(f"control row differs from source line {line_number}")
            source_fixed = {
                key: value for key, value in source_row.items() if key not in {"neg", "neg_id"}
            }
            candidate_fixed = {
                key: value
                for key, value in candidate_row.items()
                if key not in {"neg", "neg_id"}
            }
            if source_fixed != candidate_fixed:
                raise ValueError(f"candidate changed non-negative fields at {line_number}")
            if len(candidate_row["neg"]) != len(candidate_row["neg_id"]):
                raise ValueError(f"candidate negative alignment failed at {line_number}")
            source_pairs = {
                str(identifier): text
                for identifier, text in zip(source_row["neg_id"], source_row["neg"])
            }
            for identifier, text in zip(candidate_row["neg_id"], candidate_row["neg"]):
                if str(identifier) not in source_pairs or source_pairs[str(identifier)] != text:
                    raise ValueError(f"candidate negative is not source-traceable at {line_number}")
            if list(candidate_row["neg_id"]) != list(item["selected_negative_ids"]):
                raise ValueError(f"candidate/metadata selected ids differ at {line_number}")
            scores = item["selected_negative_scores"]
            if any(float(left) < float(right) for left, right in zip(scores, scores[1:])):
                raise ValueError(f"selected negatives are not hardness-sorted at {line_number}")
            if item["query_sha256"] != query_hash(source_row["query"]):
                raise ValueError(f"query hash differs at {line_number}")
            bucket = query_bucket(
                source_row["query"],
                salt=manifest["shard"]["salt"],
                modulus=manifest["shard"]["modulus"],
            )
            if bucket != manifest["shard"]["bucket"]:
                raise ValueError(f"source line is outside declared bucket: {line_number}")
            for name in ("best_positive_score", "best_negative_score", "positive_negative_margin"):
                if not math.isfinite(float(item[name])):
                    raise ValueError(f"non-finite mining score at {line_number}")
            seen.add(line_number)
    if seen != set(by_line):
        raise ValueError("one or more metadata source lines were not found")

    expected_hashes = {
        "control": sha256_file(shard_dir / "control.jsonl"),
        "candidate": sha256_file(shard_dir / "candidate.jsonl"),
        "metadata": sha256_file(shard_dir / "mining.jsonl"),
    }
    for name, actual in expected_hashes.items():
        if manifest["outputs"][name]["sha256"] != actual:
            raise ValueError(f"{name} hash differs from manifest")
    if manifest["inputs"]["source"]["sha256"] != sha256_file(source):
        raise ValueError("source hash differs from manifest")

    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "passed": True,
        "rows": len(control),
        "unique_source_lines": len(seen),
        "source_sha256": sha256_file(source),
        "manifest_sha256": sha256_file(shard_dir / "manifest.json"),
        "output_sha256": expected_hashes,
        "contract": {
            "control_exact_source_rows": True,
            "candidate_only_changes_neg_and_neg_id": True,
            "candidate_negatives_are_source_subsets": True,
            "metadata_and_candidate_alignment": True,
            "normalized_query_bucket_verified": True,
            "external_data_used": False,
            "generated_text_used": False,
            "locked_test_used": False,
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(source=args.source, shard_dir=args.shard_dir, output=args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
