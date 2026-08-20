#!/usr/bin/env python3
"""Build conservative hard-filter and soft-weight LRAT training arms."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from solution.src.build_trajectory_provenance import file_sha256


ALLOWED_FLAGS = {
    "explicit_negative_reasoning",
    "empty_reasoning",
    "answer_unmatched",
    "long_trajectory",
    "continue_search",
    "low_rank",
    "rank_missing",
    "strict_negative_combo",
    "mapping_mismatch",
    "mapping_ambiguous",
}


def row_flags(
    provenance: dict[str, Any], *, long_steps: int, low_rank: int
) -> set[str]:
    bucket = provenance["bucket"]
    if bucket == "mismatch":
        return {"mapping_mismatch"}
    if bucket == "ambiguous":
        return {"mapping_ambiguous"}
    if bucket != "stable":
        raise ValueError(f"unsupported provenance bucket: {bucket}")
    event = provenance["event"]
    flags = set()
    if event.get("negative_cues"):
        flags.add("explicit_negative_reasoning")
    if event.get("reasoning_len") == 0:
        flags.add("empty_reasoning")
    if not event.get("answer_token_subset"):
        flags.add("answer_unmatched")
    if event.get("trajectory_steps", 0) >= long_steps:
        flags.add("long_trajectory")
    if event.get("next_tool_name") == "search":
        flags.add("continue_search")
    rank = event.get("retrieved_rank")
    if isinstance(rank, int) and rank >= low_rank:
        flags.add("low_rank")
    if rank is None:
        flags.add("rank_missing")
    if (
        "explicit_negative_reasoning" in flags
        and ("answer_unmatched" in flags or "low_rank" in flags)
    ):
        flags.add("strict_negative_combo")
    return flags


def load_policy(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("version") != "trajectory_quality_policy_v1":
        raise ValueError("unsupported policy version")
    hard = value.get("hard_delete")
    soft = value.get("soft_multipliers")
    thresholds = value.get("thresholds")
    if not isinstance(hard, dict) or not isinstance(soft, dict) or not isinstance(thresholds, dict):
        raise ValueError("policy missing hard_delete, soft_multipliers, or thresholds")
    rules = hard.get("rules", [])
    manual_bad_rows = hard.get("manual_bad_row_indices", [])
    if not isinstance(rules, list):
        raise ValueError("hard_delete.rules must be a list")
    if (
        not isinstance(manual_bad_rows, list)
        or any(not isinstance(value, int) or value < 0 for value in manual_bad_rows)
        or len(manual_bad_rows) != len(set(manual_bad_rows))
    ):
        raise ValueError("hard_delete.manual_bad_row_indices must be unique non-negative integers")
    if not rules and not manual_bad_rows:
        raise ValueError("hard_delete requires rules or manual_bad_row_indices")
    used = set(soft)
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"hard_delete.rules[{index}] must be an object")
        all_of = rule.get("all_of", [])
        any_of = rule.get("any_of", [])
        if not isinstance(all_of, list) or not isinstance(any_of, list):
            raise ValueError(f"hard_delete.rules[{index}] flags must be lists")
        if not all_of and not any_of:
            raise ValueError(f"hard_delete.rules[{index}] must match at least one flag")
        used.update(all_of)
        used.update(any_of)
    unknown = used - ALLOWED_FLAGS
    if unknown:
        raise ValueError(f"unknown policy flags: {sorted(unknown)}")
    for flag, multiplier in soft.items():
        if not isinstance(multiplier, (int, float)) or not (0 < multiplier <= 1):
            raise ValueError(f"invalid multiplier for {flag}")
    minimum = value.get("minimum_combined_multiplier")
    if not isinstance(minimum, (int, float)) or not (0 < minimum <= 1):
        raise ValueError("minimum_combined_multiplier must be in (0, 1]")
    for key in ("long_trajectory_min_steps", "low_rank_min"):
        if not isinstance(thresholds.get(key), int) or thresholds[key] < 1:
            raise ValueError(f"invalid threshold: {key}")
    if not value.get("manual_review", {}).get("completed"):
        raise ValueError("policy must cite a completed manual review")
    return value


def matching_hard_rules(
    flags: set[str], rules: list[dict[str, Any]]
) -> list[str]:
    matches = []
    for index, rule in enumerate(rules):
        all_of = set(rule.get("all_of", []))
        any_of = set(rule.get("any_of", []))
        if all_of <= flags and (not any_of or bool(any_of & flags)):
            matches.append(str(rule.get("name") or f"rule_{index}"))
    return matches


def validate_alignment(
    row: dict[str, Any], provenance: dict[str, Any], row_index: int
) -> None:
    if provenance.get("row_index") != row_index:
        raise ValueError(f"provenance row index mismatch at {row_index}")
    query = row.get("query")
    positive_ids = row.get("pos_id")
    if not isinstance(query, str) or not isinstance(positive_ids, list) or len(positive_ids) != 1:
        raise ValueError(f"invalid pair identity at row {row_index}")
    if hashlib.sha256(query.encode()).hexdigest() != provenance.get("query_sha256"):
        raise ValueError(f"query hash mismatch at row {row_index}")
    if str(positive_ids[0]) != provenance.get("pos_id"):
        raise ValueError(f"positive id mismatch at row {row_index}")
    if row.get("reasoning_len") != provenance.get("reasoning_len"):
        raise ValueError(f"reasoning length mismatch at row {row_index}")
    weight = row.get("reweight_rate")
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"invalid base weight at row {row_index}")
    if not math.isclose(float(weight), float(provenance.get("reweight_rate")), rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"base weight mismatch at row {row_index}")


def iter_aligned(
    pairs_path: Path, provenance_path: Path
) -> Iterable[tuple[int, str, dict[str, Any], dict[str, Any]]]:
    with pairs_path.open("r", encoding="utf-8") as pairs, provenance_path.open(
        "r", encoding="utf-8"
    ) as provenance:
        row_index = 0
        while True:
            pair_line = pairs.readline()
            provenance_line = provenance.readline()
            if not pair_line and not provenance_line:
                break
            if not pair_line or not provenance_line:
                raise ValueError("pair/provenance row count mismatch")
            row = json.loads(pair_line)
            provenance_row = json.loads(provenance_line)
            validate_alignment(row, provenance_row, row_index)
            yield row_index, pair_line, row, provenance_row
            row_index += 1


def describe(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    at = lambda fraction: ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": at(0.9),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def build(
    pairs_path: Path,
    provenance_path: Path,
    policy_path: Path,
    output_root: Path,
    *,
    expected_rows: int | None,
    expected_pairs_sha256: str | None,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    policy = load_policy(policy_path)
    pair_sha256 = file_sha256(pairs_path)
    if expected_pairs_sha256 and pair_sha256 != expected_pairs_sha256:
        raise ValueError("pair data SHA-256 mismatch")
    thresholds = policy["thresholds"]
    hard_rules = policy["hard_delete"].get("rules", [])
    manual_bad_rows = set(policy["hard_delete"].get("manual_bad_row_indices", []))
    stable_only = bool(policy["hard_delete"].get("stable_only", True))
    soft_multipliers = {
        flag: float(multiplier)
        for flag, multiplier in policy["soft_multipliers"].items()
    }
    minimum_multiplier = float(policy["minimum_combined_multiplier"])

    metadata = []
    flag_counts: collections.Counter[str] = collections.Counter()
    deleted_flag_counts: collections.Counter[str] = collections.Counter()
    base_weights = []
    raw_soft_weights = []
    for row_index, _, row, provenance in iter_aligned(pairs_path, provenance_path):
        flags = row_flags(
            provenance,
            long_steps=thresholds["long_trajectory_min_steps"],
            low_rank=thresholds["low_rank_min"],
        )
        flag_counts.update(flags)
        hard_matches = matching_hard_rules(flags, hard_rules)
        if row_index in manual_bad_rows:
            hard_matches.append("manual_review_confirmed_bad")
        delete = bool(hard_matches) and (
            not stable_only or provenance["bucket"] == "stable"
        )
        if delete:
            deleted_flag_counts.update(hard_matches)
        multiplier = 1.0
        applied_soft_flags = []
        for flag in sorted(flags):
            if flag in soft_multipliers:
                multiplier *= soft_multipliers[flag]
                applied_soft_flags.append(flag)
        multiplier = max(minimum_multiplier, multiplier)
        base_weight = float(row["reweight_rate"])
        metadata.append(
            {
                "row_index": row_index,
                "delete_b": delete,
                "hard_rule_matches": hard_matches,
                "flags": sorted(flags),
                "applied_soft_flags": applied_soft_flags,
                "soft_multiplier_before_normalization": multiplier,
            }
        )
        base_weights.append(base_weight)
        raw_soft_weights.append(base_weight * multiplier)
    if expected_rows is not None and len(metadata) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(metadata)}")
    missing_manual_rows = manual_bad_rows - {
        decision["row_index"] for decision in metadata
    }
    if missing_manual_rows:
        raise ValueError(
            f"manual bad row indexes are outside the pair data: {sorted(missing_manual_rows)[:10]}"
        )
    if not metadata:
        raise ValueError("empty training data")
    target_weight_mean = statistics.fmean(base_weights)
    raw_soft_mean = statistics.fmean(raw_soft_weights)
    normalization_factor = target_weight_mean / raw_soft_mean

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging.", dir=output_root.parent)
    )
    try:
        arm_b_path = staging / "arm_b_hard_filter.jsonl"
        arm_c_path = staging / "arm_c_soft_weight.jsonl"
        decision_path = staging / "row_decisions.jsonl"
        b_rows = 0
        c_weights = []
        with arm_b_path.open("x", encoding="utf-8") as arm_b, arm_c_path.open(
            "x", encoding="utf-8"
        ) as arm_c, decision_path.open("x", encoding="utf-8") as decisions:
            for (row_index, pair_line, row, provenance), decision in zip(
                iter_aligned(pairs_path, provenance_path), metadata
            ):
                if row_index != decision["row_index"]:
                    raise AssertionError("row decision alignment failed")
                if not decision["delete_b"]:
                    arm_b.write(pair_line if pair_line.endswith("\n") else pair_line + "\n")
                    b_rows += 1
                new_weight = (
                    float(row["reweight_rate"])
                    * decision["soft_multiplier_before_normalization"]
                    * normalization_factor
                )
                if not math.isfinite(new_weight) or new_weight <= 0:
                    raise RuntimeError(f"invalid soft weight at row {row_index}")
                row["reweight_rate"] = new_weight
                arm_c.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                c_weights.append(new_weight)
                decisions.write(
                    json.dumps(
                        {
                            **decision,
                            "bucket": provenance["bucket"],
                            "base_weight": provenance["reweight_rate"],
                            "soft_weight": new_weight,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        if b_rows < 1 or b_rows >= len(metadata):
            raise ValueError(f"hard-filter arm must delete some but not all rows: {b_rows}")
        if not math.isclose(
            statistics.fmean(c_weights), target_weight_mean, rel_tol=0, abs_tol=1e-10
        ):
            raise AssertionError("soft-weight normalization changed the global mean")
        manifest = {
            "created_at": datetime.now().astimezone().isoformat(),
            "contract": {
                "arm_a": "unchanged early_stop_v1 train",
                "arm_b": "delete only policy high-confidence rows",
                "arm_c": "retain every row and change only reweight_rate",
                "soft_weight_mean_matches_arm_a": True,
                "locked_test_used": False,
            },
            "inputs": {
                "pairs": {
                    "path": str(pairs_path.resolve()),
                    "rows": len(metadata),
                    "bytes": pairs_path.stat().st_size,
                    "sha256": pair_sha256,
                },
                "provenance": {
                    "path": str(provenance_path.resolve()),
                    "sha256": file_sha256(provenance_path),
                },
                "policy": {
                    "path": str(policy_path.resolve()),
                    "sha256": file_sha256(policy_path),
                    "value": policy,
                },
            },
            "rows": {
                "arm_a": len(metadata),
                "arm_b": b_rows,
                "arm_c": len(metadata),
                "deleted_from_b": len(metadata) - b_rows,
            },
            "flags": {
                "all_rows": dict(flag_counts),
                "hard_deleted_matches": dict(deleted_flag_counts),
            },
            "weights": {
                "arm_a": describe(base_weights),
                "arm_c": describe(c_weights),
                "normalization_factor": normalization_factor,
            },
            "outputs": {},
        }
        for path in (arm_b_path, arm_c_path, decision_path):
            manifest["outputs"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-pairs-sha256")
    args = parser.parse_args()
    result = build(
        args.pairs,
        args.provenance,
        args.policy,
        args.output_root,
        expected_rows=args.expected_rows,
        expected_pairs_sha256=args.expected_pairs_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
