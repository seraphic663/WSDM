#!/usr/bin/env python3
"""Explain paired retrieval changes by the earliest supporting search stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from solution.src.compare_paired_evals import load_ranks
except ModuleNotFoundError:
    from compare_paired_evals import load_ranks


def normalized_query_sha(query: str) -> str:
    normalized = re.sub(r"\s+", " ", query.strip()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def provenance_stages(path: Path) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    ambiguous: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query_sha = row["normalized_query_sha256"]
            if row.get("bucket") == "stable":
                search_idx = row.get("event", {}).get("search_idx")
                if isinstance(search_idx, int) and not isinstance(search_idx, bool) and search_idx >= 0:
                    result[query_sha].append(search_idx)
            elif row.get("bucket") == "ambiguous":
                ambiguous.add(query_sha)
    for query_sha in ambiguous:
        result.setdefault(query_sha, [])
    return result


def stage_name(indexes: list[int] | None) -> str:
    if indexes is None:
        return "unmapped"
    if not indexes:
        return "ambiguous_only"
    earliest = min(indexes)
    if earliest == 0:
        return "earliest_idx0"
    if earliest <= 2:
        return "earliest_idx1_2"
    return "earliest_idx3plus"


def metric_summary(raw: list[int], candidate: list[int]) -> dict[str, Any]:
    count = len(raw)
    return {
        "queries": count,
        "improved_rank": sum(c < r for r, c in zip(raw, candidate)),
        "tied_rank": sum(c == r for r, c in zip(raw, candidate)),
        "degraded_rank": sum(c > r for r, c in zip(raw, candidate)),
        "delta_recall_at_1": (
            sum(float(c <= 1) - float(r <= 1) for r, c in zip(raw, candidate)) / count
        ),
        "delta_recall_at_5": (
            sum(float(c <= 5) - float(r <= 5) for r, c in zip(raw, candidate)) / count
        ),
        "delta_recall_at_10": (
            sum(float(c <= 10) - float(r <= 10) for r, c in zip(raw, candidate)) / count
        ),
        "delta_mrr": (
            sum((1.0 / c) - (1.0 / r) for r, c in zip(raw, candidate)) / count
        ),
    }


def analyze(
    dev_path: Path,
    provenance_path: Path,
    candidate_eval_path: Path,
    baselines: list[tuple[str, Path]],
) -> dict[str, Any]:
    candidate_value, candidate_ranks, candidate_hashes = load_ranks(candidate_eval_path)
    dev_rows = [json.loads(line) for line in dev_path.read_text(encoding="utf-8").splitlines()]
    if len(dev_rows) != len(candidate_ranks):
        raise ValueError("dev/evaluation row count mismatch")
    stages = provenance_stages(provenance_path)
    labels = [stage_name(stages.get(normalized_query_sha(row["query"]))) for row in dev_rows]
    report: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "definition": (
            "Each dev query is assigned by the earliest stable search_idx among all "
            "official full-pair provenance rows sharing its normalized query."
        ),
        "candidate_eval": str(candidate_eval_path.resolve()),
        "dev": str(dev_path.resolve()),
        "provenance": str(provenance_path.resolve()),
        "stage_counts": {
            label: labels.count(label) for label in sorted(set(labels))
        },
        "comparisons": {},
        "locked_test_used": False,
    }
    for baseline_name, baseline_path in baselines:
        baseline_value, baseline_ranks, baseline_hashes = load_ranks(baseline_path)
        if baseline_hashes != candidate_hashes:
            raise ValueError(f"{baseline_name}: paired query hashes differ")
        if Path(baseline_value["input"]).resolve() != Path(candidate_value["input"]).resolve():
            raise ValueError(f"{baseline_name}: evaluation inputs differ")
        by_stage = {}
        for label in sorted(set(labels)):
            indexes = [index for index, value in enumerate(labels) if value == label]
            by_stage[label] = metric_summary(
                [baseline_ranks[index] for index in indexes],
                [candidate_ranks[index] for index in indexes],
            )
        report["comparisons"][baseline_name] = {
            "baseline_eval": str(baseline_path.resolve()),
            "overall": metric_summary(baseline_ranks, candidate_ranks),
            "by_earliest_search_stage": by_stage,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--candidate-eval", required=True, type=Path)
    parser.add_argument(
        "--baseline", nargs=2, action="append", metavar=("LABEL", "EVAL"), required=True
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(
        args.dev,
        args.provenance,
        args.candidate_eval,
        [(label, Path(path)) for label, path in args.baseline],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
