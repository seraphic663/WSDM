#!/usr/bin/env python3
"""Paired gate for one offline retriever-in-the-loop update."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from solution.src.compare_paired_evals import load_ranks, paired_delta
except ModuleNotFoundError:
    from compare_paired_evals import load_ranks, paired_delta


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def compare_loop(
    *,
    parent_eval: Path,
    control_eval: Path,
    candidate_eval: Path,
    output: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if bootstrap_samples < 1000:
        raise ValueError("bootstrap_samples must be at least 1000")
    parent_value, parent_ranks, parent_hashes = load_ranks(parent_eval)
    control_value, control_ranks, control_hashes = load_ranks(control_eval)
    candidate_value, candidate_ranks, candidate_hashes = load_ranks(candidate_eval)
    if not (
        parent_value.get("input")
        == control_value.get("input")
        == candidate_value.get("input")
    ):
        raise ValueError("evaluation inputs differ")
    if not (parent_hashes == control_hashes == candidate_hashes):
        raise ValueError("paired query identities differ")
    if len(set(parent_hashes)) != len(parent_hashes):
        raise ValueError("evaluation query identities are duplicated")

    candidate_vs_control = paired_delta(
        control_ranks, candidate_ranks, bootstrap_samples, seed
    )
    candidate_vs_parent = paired_delta(
        parent_ranks, candidate_ranks, bootstrap_samples, seed + 1
    )
    control_vs_parent = paired_delta(
        parent_ranks, control_ranks, bootstrap_samples, seed + 2
    )
    directional_vs_control = all(
        candidate_vs_control[name]["delta"] > 0 for name in ("recall_at_1", "mrr")
    )
    robust_vs_control = all(
        candidate_vs_control[name]["ci95_low"] >= 0
        for name in ("recall_at_1", "mrr")
    )
    no_topk_regression_vs_control = all(
        candidate_vs_control[name]["delta"] >= 0
        for name in ("recall_at_5", "recall_at_10")
    )
    improves_parent = all(
        candidate_vs_parent[name]["delta"] > 0 for name in ("recall_at_1", "mrr")
    )
    no_topk_regression_vs_parent = all(
        candidate_vs_parent[name]["delta"] >= 0
        for name in ("recall_at_5", "recall_at_10")
    )
    gate = {
        "passed": (
            directional_vs_control
            and robust_vs_control
            and no_topk_regression_vs_control
            and improves_parent
            and no_topk_regression_vs_parent
        ),
        "criteria": {
            "candidate_r1_and_mrr_positive_vs_uniform_control": directional_vs_control,
            "candidate_r1_and_mrr_ci95_lower_nonnegative_vs_uniform_control": robust_vs_control,
            "candidate_r5_and_r10_nonnegative_vs_uniform_control": no_topk_regression_vs_control,
            "candidate_r1_and_mrr_positive_vs_parent": improves_parent,
            "candidate_r5_and_r10_nonnegative_vs_parent": no_topk_regression_vs_parent,
        },
        "interpretation": (
            "Only a robust gain over the same-shard uniform control plus a non-regressing "
            "gain over the parent authorizes the next offline flywheel loop."
        ),
    }
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "rows": len(parent_ranks),
        "input": parent_value.get("input"),
        "evals": {
            "parent": str(parent_eval.resolve()),
            "control": str(control_eval.resolve()),
            "candidate": str(candidate_eval.resolve()),
        },
        "comparisons": {
            "candidate_vs_control": candidate_vs_control,
            "candidate_vs_parent": candidate_vs_parent,
            "control_vs_parent": control_vs_parent,
        },
        "gate": gate,
        "locked_test_used": False,
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-eval", required=True, type=Path)
    parser.add_argument("--control-eval", required=True, type=Path)
    parser.add_argument("--candidate-eval", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    print(
        json.dumps(
            compare_loop(
                parent_eval=args.parent_eval,
                control_eval=args.control_eval,
                candidate_eval=args.candidate_eval,
                output=args.output,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
