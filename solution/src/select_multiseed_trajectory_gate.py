#!/usr/bin/env python3
"""Aggregate predeclared single-seed trajectory gates."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


def read_gate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("bootstrap_samples", 0) < 10000:
        raise ValueError(f"{path}: expected at least 10000 bootstrap samples")
    comparisons = value.get("comparisons")
    if not isinstance(comparisons, list) or {
        item.get("label") for item in comparisons
    } != {"checkpoint-500", "final-1000"}:
        raise ValueError(f"{path}: invalid comparison labels")
    if not isinstance(value.get("gate", {}).get("passed"), bool):
        raise ValueError(f"{path}: missing gate result")
    return value


def aggregate(seed_gates: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    if len(seed_gates) != 3 or len({seed for seed, _ in seed_gates}) != 3:
        raise ValueError("exactly three unique seed gates are required")
    labels = ("checkpoint-500", "final-1000")
    metrics = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr")
    summaries: dict[str, dict[str, float]] = {}
    for label in labels:
        summaries[label] = {}
        for metric in metrics:
            values = []
            for _, gate in seed_gates:
                comparison = next(
                    item for item in gate["comparisons"] if item["label"] == label
                )
                values.append(float(comparison[metric]["delta"]))
            summaries[label][metric] = sum(values) / len(values)

    seed_passes = {
        str(seed): bool(gate["gate"]["passed"]) for seed, gate in seed_gates
    }
    directional = all(
        summaries[label][metric] > 0
        for label in labels
        for metric in ("recall_at_1", "mrr")
    )
    final_topk = all(
        summaries["final-1000"][metric] >= 0
        for metric in ("recall_at_5", "recall_at_10")
    )
    passed_seed_count = sum(seed_passes.values())
    passed = passed_seed_count >= 2 and directional and final_topk
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "predeclared_criteria": {
            "at_least_two_of_three_single_seed_gates_pass": passed_seed_count >= 2,
            "mean_r1_and_mrr_positive_at_500_and_1000": directional,
            "mean_final_r5_and_r10_nonnegative": final_topk,
        },
        "seed_passes": seed_passes,
        "passed_seed_count": passed_seed_count,
        "mean_deltas": summaries,
        "full_epoch_authorized": passed,
        "selected_arm": "search_idx_soft_v2" if passed else None,
        "locked_test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate",
        action="append",
        nargs=2,
        metavar=("SEED", "PATH"),
        required=True,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    seed_gates = [(int(seed), read_gate(Path(path))) for seed, path in args.gate]
    result = aggregate(seed_gates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{args.output.name}.tmp.", dir=args.output.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, args.output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
