#!/usr/bin/env python3
"""Explain a flywheel update by the parent retriever's dev difficulty."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from solution.src.compare_paired_evals import load_ranks
except ModuleNotFoundError:
    from compare_paired_evals import load_ranks


def rank_bucket(rank: int) -> str:
    if rank == 1:
        return "parent_rank1"
    if rank <= 5:
        return "parent_rank2_5"
    if rank <= 10:
        return "parent_rank6_10"
    return "parent_rank11plus"


def summarize(reference: list[int], candidate: list[int], indexes: list[int]) -> dict[str, Any]:
    if not indexes:
        return {
            "queries": 0,
            "improved_rank": 0,
            "tied_rank": 0,
            "degraded_rank": 0,
            "delta_recall_at_1": 0.0,
            "delta_recall_at_5": 0.0,
            "delta_recall_at_10": 0.0,
            "delta_mrr": 0.0,
        }
    count = len(indexes)
    return {
        "queries": count,
        "improved_rank": sum(candidate[i] < reference[i] for i in indexes),
        "tied_rank": sum(candidate[i] == reference[i] for i in indexes),
        "degraded_rank": sum(candidate[i] > reference[i] for i in indexes),
        "delta_recall_at_1": sum(
            float(candidate[i] <= 1) - float(reference[i] <= 1) for i in indexes
        )
        / count,
        "delta_recall_at_5": sum(
            float(candidate[i] <= 5) - float(reference[i] <= 5) for i in indexes
        )
        / count,
        "delta_recall_at_10": sum(
            float(candidate[i] <= 10) - float(reference[i] <= 10) for i in indexes
        )
        / count,
        "delta_mrr": sum(
            (1.0 / candidate[i]) - (1.0 / reference[i]) for i in indexes
        )
        / count,
    }


def analyze(
    *,
    parent_eval: Path,
    control_eval: Path,
    candidate_eval: Path,
    output: Path,
) -> dict[str, Any]:
    parent_value, parent, parent_hashes = load_ranks(parent_eval)
    control_value, control, control_hashes = load_ranks(control_eval)
    candidate_value, candidate, candidate_hashes = load_ranks(candidate_eval)
    if not (
        parent_value.get("input")
        == control_value.get("input")
        == candidate_value.get("input")
    ):
        raise ValueError("evaluation inputs differ")
    if not (parent_hashes == control_hashes == candidate_hashes):
        raise ValueError("query identities differ")

    buckets: dict[str, list[int]] = {
        "parent_rank1": [],
        "parent_rank2_5": [],
        "parent_rank6_10": [],
        "parent_rank11plus": [],
    }
    for index, rank in enumerate(parent):
        buckets[rank_bucket(rank)].append(index)
    all_indexes = list(range(len(parent)))
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "definition": "dev queries grouped by positive rank under the parent retriever before the offline flywheel update",
        "rows": len(parent),
        "evals": {
            "parent": str(parent_eval.resolve()),
            "control": str(control_eval.resolve()),
            "candidate": str(candidate_eval.resolve()),
        },
        "candidate_vs_control": {
            "overall": summarize(control, candidate, all_indexes),
            "by_parent_difficulty": {
                name: summarize(control, candidate, indexes)
                for name, indexes in buckets.items()
            },
        },
        "candidate_vs_parent": {
            "overall": summarize(parent, candidate, all_indexes),
            "by_parent_difficulty": {
                name: summarize(parent, candidate, indexes)
                for name, indexes in buckets.items()
            },
        },
        "parent_bucket_counts": {name: len(indexes) for name, indexes in buckets.items()},
        "locked_test_used": False,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-eval", required=True, type=Path)
    parser.add_argument("--control-eval", required=True, type=Path)
    parser.add_argument("--candidate-eval", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                parent_eval=args.parent_eval,
                control_eval=args.control_eval,
                candidate_eval=args.candidate_eval,
                output=args.output,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
