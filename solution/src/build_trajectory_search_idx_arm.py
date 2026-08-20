#!/usr/bin/env python3
"""Build a row-aligned search-index soft-weight arm from official LRAT data."""

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
from typing import Any

from solution.src.build_trajectory_quality_arms import iter_aligned


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported config schema")
    if value.get("feature") != "event.search_idx":
        raise ValueError("config feature must be event.search_idx")
    multipliers = value.get("stable_multipliers")
    if not isinstance(multipliers, dict) or set(multipliers) != {
        "0",
        "1",
        "2",
        "3_plus",
    }:
        raise ValueError("stable_multipliers must define 0, 1, 2, and 3_plus")
    for name, multiplier in multipliers.items():
        if (
            not isinstance(multiplier, (int, float))
            or not math.isfinite(multiplier)
            or multiplier <= 0
        ):
            raise ValueError(f"invalid multiplier {name}")
    if value.get("locked_test_used") is not False:
        raise ValueError("locked_test_used must be false")
    return value


def search_multiplier(search_idx: int, config: dict[str, Any]) -> float:
    if not isinstance(search_idx, int) or isinstance(search_idx, bool) or search_idx < 0:
        raise ValueError(f"invalid zero-based search_idx: {search_idx!r}")
    key = str(search_idx) if search_idx < 3 else "3_plus"
    return float(config["stable_multipliers"][key])


def row_multiplier(provenance: dict[str, Any], config: dict[str, Any]) -> tuple[float, int | None]:
    bucket = provenance.get("bucket")
    if bucket != "stable":
        if bucket not in set(config.get("neutral_buckets", [])):
            raise ValueError(f"unexpected provenance bucket: {bucket!r}")
        return 1.0, None
    event = provenance.get("event")
    if not isinstance(event, dict):
        raise ValueError("stable provenance row is missing event")
    search_idx = event.get("search_idx")
    return search_multiplier(search_idx, config), search_idx


def build(
    pairs_path: Path,
    provenance_path: Path,
    config_path: Path,
    output_root: Path,
    *,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    config = load_config(config_path)

    bucket_counts: collections.Counter[str] = collections.Counter()
    search_counts: collections.Counter[int] = collections.Counter()
    source_total = 0.0
    neutral_total = 0.0
    stable_adjusted_total = 0.0
    rows = 0

    for _, _, row, provenance in iter_aligned(pairs_path, provenance_path):
        base_weight = float(row["reweight_rate"])
        multiplier, search_idx = row_multiplier(provenance, config)
        source_total += base_weight
        bucket = str(provenance["bucket"])
        bucket_counts[bucket] += 1
        if search_idx is None:
            neutral_total += base_weight
        else:
            search_counts[search_idx] += 1
            stable_adjusted_total += base_weight * multiplier
        rows += 1

    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {rows}")
    if rows == 0 or stable_adjusted_total <= 0:
        raise ValueError("no stable rows were available for weighting")
    if bucket_counts["stable"] == 0:
        raise ValueError("no stable provenance rows")

    stable_target_total = source_total - neutral_total
    stable_normalization = stable_target_total / stable_adjusted_total
    if not math.isfinite(stable_normalization) or stable_normalization <= 0:
        raise RuntimeError("invalid stable normalization factor")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging.", dir=output_root.parent)
    )
    output_data = staging / "train_search_idx_soft.jsonl"
    decisions_path = staging / "row_decisions.jsonl"
    new_weights: list[float] = []
    changed_rows = 0
    try:
        with output_data.open("x", encoding="utf-8") as output, decisions_path.open(
            "x", encoding="utf-8"
        ) as decisions:
            for row_index, _, row, provenance in iter_aligned(
                pairs_path, provenance_path
            ):
                base_weight = float(row["reweight_rate"])
                multiplier, search_idx = row_multiplier(provenance, config)
                if search_idx is None:
                    new_weight = base_weight
                    applied_normalization = 1.0
                else:
                    new_weight = base_weight * multiplier * stable_normalization
                    applied_normalization = stable_normalization
                if not math.isfinite(new_weight) or new_weight <= 0:
                    raise RuntimeError(f"invalid output weight at row {row_index}")
                if not math.isclose(new_weight, base_weight, rel_tol=0, abs_tol=1e-15):
                    changed_rows += 1
                row["reweight_rate"] = new_weight
                output.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                decisions.write(
                    json.dumps(
                        {
                            "row_index": row_index,
                            "bucket": provenance["bucket"],
                            "search_idx": search_idx,
                            "base_weight": base_weight,
                            "feature_multiplier": multiplier,
                            "normalization": applied_normalization,
                            "new_weight": new_weight,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                new_weights.append(new_weight)

        if changed_rows == 0:
            raise RuntimeError("search-index arm changed zero rows")
        if not math.isclose(
            sum(new_weights), source_total, rel_tol=0, abs_tol=1e-8
        ):
            raise RuntimeError("output total weight differs from source total")

        manifest = {
            "created_at": datetime.now().astimezone().isoformat(),
            "contract": {
                "method": "stable provenance event.search_idx soft weighting",
                "source_fields_preserved_except": ["reweight_rate"],
                "neutral_buckets_unchanged": config["neutral_buckets"],
                "stable_weight_total_preserved": True,
                "locked_test_used": False,
            },
            "inputs": {
                "pairs": str(pairs_path.resolve()),
                "pairs_sha256": sha256_file(pairs_path),
                "provenance": str(provenance_path.resolve()),
                "provenance_sha256": sha256_file(provenance_path),
                "config": str(config_path.resolve()),
                "config_sha256": sha256_file(config_path),
            },
            "rows": rows,
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "search_idx_counts": {
                str(key): value for key, value in sorted(search_counts.items())
            },
            "changed_rows": changed_rows,
            "source_weight": {
                "total": source_total,
                "mean": source_total / rows,
            },
            "output_weight": {
                "total": sum(new_weights),
                "mean": statistics.fmean(new_weights),
                "min": min(new_weights),
                "max": max(new_weights),
            },
            "stable_normalization": stable_normalization,
            "outputs": {
                "train": output_data.name,
                "train_sha256": sha256_file(output_data),
                "decisions": decisions_path.name,
                "decisions_sha256": sha256_file(decisions_path),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    result = build(
        args.pairs,
        args.provenance,
        args.config,
        args.output_root,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
