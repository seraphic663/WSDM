#!/usr/bin/env python3
"""Compare multi-seed reliability arms with query-cluster uncertainty."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from compare_paired_evals import load_ranks


def metric_values(ranks: list[int]) -> dict[str, list[float]]:
    return {
        "mrr": [1.0 / rank for rank in ranks],
        "recall_at_1": [float(rank <= 1) for rank in ranks],
        "recall_at_5": [float(rank <= 5) for rank in ranks],
        "recall_at_10": [float(rank <= 10) for rank in ranks],
    }


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def aggregate_contrast(
    left_by_seed: dict[int, list[int]],
    right_by_seed: dict[int, list[int]],
    *,
    samples: int,
    seed: int,
) -> dict:
    seeds = sorted(left_by_seed)
    if seeds != sorted(right_by_seed):
        raise ValueError("contrast seed sets differ")
    n = len(left_by_seed[seeds[0]])
    per_metric = {name: [0.0] * n for name in metric_values(left_by_seed[seeds[0]])}
    seed_deltas = {name: [] for name in per_metric}
    for run_seed in seeds:
        left = metric_values(left_by_seed[run_seed])
        right = metric_values(right_by_seed[run_seed])
        for name in per_metric:
            values = [a - b for a, b in zip(left[name], right[name])]
            seed_deltas[name].append(sum(values) / n)
            for index, value in enumerate(values):
                per_metric[name][index] += value / len(seeds)
    rng = random.Random(seed)
    distributions = {name: [] for name in per_metric}
    for _ in range(samples):
        indexes = [rng.randrange(n) for _ in range(n)]
        for name, values in per_metric.items():
            distributions[name].append(sum(values[index] for index in indexes) / n)
    result = {}
    for name, values in per_metric.items():
        result[name] = {
            "delta": sum(values) / n,
            "ci95": [percentile(distributions[name], 0.025), percentile(distributions[name], 0.975)],
            "seed_deltas": seed_deltas[name],
            "nonnegative_seeds": sum(value >= 0 for value in seed_deltas[name]),
        }
    return {"seeds": seeds, "queries": n, "metrics": result}


def analyze(args: argparse.Namespace) -> dict:
    arm_results = json.loads(args.arm_results.read_text(encoding="utf-8"))
    paths: dict[str, dict[int, Path]] = {}
    for record in arm_results["runs"]:
        if record.get("status") != "completed" or record.get("locked_test_used") is not False:
            raise ValueError(f"unhealthy arm record: {record.get('run_id')}")
        paths.setdefault(str(record["arm"]), {})[int(record["seed"])] = Path(record["evaluation"])
    paths["vanilla"] = {int(seed): Path(path) for seed, path in args.vanilla}
    seed_set = set(paths["vanilla"])
    if len(seed_set) < 3 or any(set(values) != seed_set for values in paths.values()):
        raise ValueError("all arms must have the same set of at least three seeds")

    ranks: dict[str, dict[int, list[int]]] = {}
    canonical_hashes = None
    canonical_input = None
    for arm, values in paths.items():
        ranks[arm] = {}
        for run_seed, path in values.items():
            raw, run_ranks, hashes = load_ranks(path)
            if canonical_hashes is None:
                canonical_hashes, canonical_input = hashes, raw.get("input")
            if hashes != canonical_hashes or len(set(hashes)) != len(hashes):
                raise ValueError(f"query identity mismatch: {arm}/{run_seed}")
            if raw.get("input") != canonical_input:
                raise ValueError(f"evaluation input mismatch: {arm}/{run_seed}")
            ranks[arm][run_seed] = run_ranks

    specs = [
        ("later_visit_vs_vanilla", "later_visit", "vanilla"),
        ("later_visit_vs_random", "later_visit", "random"),
        ("later_visit_vs_exposure", "later_visit", "exposure"),
        ("exposure_vs_random", "exposure", "random"),
    ]
    contrasts = {}
    for index, (label, left, right) in enumerate(specs):
        contrasts[label] = aggregate_contrast(
            ranks[left], ranks[right], samples=args.bootstrap_samples, seed=args.seed + index
        )

    required = [contrasts["later_visit_vs_vanilla"], contrasts["later_visit_vs_random"]]
    primary = all(
        comparison["metrics"][metric]["ci95"][0] >= 0
        for comparison in required
        for metric in ("mrr", "recall_at_1")
    )
    topk = all(
        comparison["metrics"][metric]["delta"] >= 0
        for comparison in required
        for metric in ("recall_at_5", "recall_at_10")
    )
    seed_consistency = all(
        comparison["metrics"]["mrr"]["nonnegative_seeds"] >= math.ceil(len(seed_set) * 2 / 3)
        for comparison in required
    )
    gate = {
        "passed": primary and topk and seed_consistency,
        "criteria": {
            "mrr_and_r1_query_bootstrap_lower_nonnegative_vs_vanilla_and_random": primary,
            "r5_and_r10_point_delta_nonnegative_vs_vanilla_and_random": topk,
            "mrr_nonnegative_in_at_least_two_thirds_of_seeds": seed_consistency,
        },
        "interpretation": "Passing supports the tested later-visit filtering rule; failure retains the diagnostic contribution and rejects a training-gain claim.",
    }
    report = {
        "schema_version": "reliability_arm_grid_analysis.v1",
        "bootstrap": {"cluster": "query_id", "seed_averaging": "per-query before resampling", "samples": args.bootstrap_samples, "seed": args.seed},
        "evaluation_input": canonical_input,
        "seeds": sorted(seed_set),
        "contrasts": contrasts,
        "gate": gate,
        "locked_test_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-results", required=True, type=Path)
    parser.add_argument("--vanilla", nargs=2, action="append", metavar=("SEED", "EVAL"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(analyze(args)["gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
