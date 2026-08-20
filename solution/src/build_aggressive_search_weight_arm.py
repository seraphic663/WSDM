#!/usr/bin/env python3
"""Build an interpretable, normalized search-stage reweighting arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield line_number, value


def stage_bucket(provenance: dict[str, Any]) -> str:
    bucket = provenance.get("bucket")
    if bucket == "ambiguous":
        return "ambiguous"
    if bucket != "stable":
        raise ValueError(f"unsupported provenance bucket: {bucket!r}")
    search_idx = provenance.get("event", {}).get("search_idx")
    if not isinstance(search_idx, int) or isinstance(search_idx, bool) or search_idx < 0:
        raise ValueError(f"invalid stable event.search_idx: {search_idx!r}")
    if search_idx == 0:
        return "idx0"
    if search_idx <= 2:
        return "idx1_2"
    return "idx3plus"


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported config schema")
    if value.get("locked_test_used") is not False:
        raise ValueError("config must prohibit locked test use")
    if value.get("external_data_used") is not False:
        raise ValueError("config must prohibit external data")
    buckets = value.get("search_stage_buckets")
    if set(buckets or {}) != {"idx0", "idx1_2", "idx3plus"}:
        raise ValueError("config must declare exactly three stable search-stage buckets")
    for name, item in buckets.items():
        weight = item.get("raw_weight")
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"invalid raw weight for {name}")
    ambiguous = value.get("ambiguous_weight")
    if not isinstance(ambiguous, (int, float)) or ambiguous != 1.0:
        raise ValueError("ambiguous_weight must remain exactly 1.0")
    return value


def provenance_distribution(
    provenance_path: Path, config: dict[str, Any], expected_rows: int
) -> tuple[Counter[str], float, dict[str, float]]:
    counts: Counter[str] = Counter()
    raw_weights = {
        name: float(item["raw_weight"])
        for name, item in config["search_stage_buckets"].items()
    }
    for row_index, (_, record) in enumerate(read_jsonl(provenance_path)):
        if record.get("row_index") != row_index:
            raise ValueError(f"provenance row_index mismatch at {row_index}")
        counts[stage_bucket(record)] += 1
    if sum(counts.values()) != expected_rows:
        raise ValueError(f"provenance rows mismatch: {sum(counts.values())}")
    stable_rows = expected_rows - counts["ambiguous"]
    stable_raw_sum = sum(counts[name] * raw_weights[name] for name in raw_weights)
    if stable_rows <= 0 or stable_raw_sum <= 0:
        raise ValueError("invalid stable provenance distribution")
    stable_scale = stable_rows / stable_raw_sum
    normalized = {name: raw_weights[name] * stable_scale for name in raw_weights}
    normalized["ambiguous"] = 1.0
    return counts, stable_scale, normalized


def atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def build(
    pairs_path: Path,
    provenance_path: Path,
    config_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    expected_pairs_sha256: str,
    expected_provenance_sha256: str,
    expected_rows: int,
) -> dict[str, Any]:
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite an existing search-weight artifact")
    for path in (pairs_path, provenance_path, config_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"invalid input: {path}")
    pairs_sha = sha256_file(pairs_path)
    provenance_sha = sha256_file(provenance_path)
    config_sha = sha256_file(config_path)
    if pairs_sha != expected_pairs_sha256:
        raise ValueError(f"pairs SHA mismatch: {pairs_sha}")
    if provenance_sha != expected_provenance_sha256:
        raise ValueError(f"provenance SHA mismatch: {provenance_sha}")

    config = load_config(config_path)
    counts, stable_scale, normalized = provenance_distribution(
        provenance_path, config, expected_rows
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    output_digest = hashlib.sha256()
    rows = 0
    changed_rows = 0
    total_weight = 0.0
    pair_iter = read_jsonl(pairs_path)
    provenance_iter = read_jsonl(provenance_path)
    try:
        with os.fdopen(fd, "wb") as target:
            while True:
                try:
                    pair_item = next(pair_iter)
                except StopIteration:
                    pair_item = None
                try:
                    provenance_item = next(provenance_iter)
                except StopIteration:
                    provenance_item = None
                if pair_item is None and provenance_item is None:
                    break
                if pair_item is None or provenance_item is None:
                    raise ValueError("pairs/provenance row count differs")
                pair_line, row = pair_item
                _, provenance = provenance_item
                if provenance.get("row_index") != rows:
                    raise ValueError(f"provenance row_index mismatch at {rows}")
                if row.get("query") != provenance.get("query"):
                    raise ValueError(f"query mismatch at line {pair_line}")
                if not row.get("pos_id") or row["pos_id"][0] != provenance.get("pos_id"):
                    raise ValueError(f"positive id mismatch at line {pair_line}")
                old_weight = row.get("reweight_rate")
                if (
                    not isinstance(old_weight, (int, float))
                    or isinstance(old_weight, bool)
                    or not math.isfinite(float(old_weight))
                    or float(old_weight) <= 0
                ):
                    raise ValueError(f"invalid source weight at line {pair_line}")
                bucket = stage_bucket(provenance)
                new_weight = normalized[bucket]
                if not math.isclose(float(old_weight), new_weight, rel_tol=0, abs_tol=1e-15):
                    changed_rows += 1
                row["reweight_rate"] = new_weight
                total_weight += new_weight
                encoded = (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                target.write(encoded)
                output_digest.update(encoded)
                rows += 1
            target.flush()
            os.fsync(target.fileno())
        if rows != expected_rows:
            raise ValueError(f"row count mismatch: {rows} != {expected_rows}")
        if not math.isclose(total_weight, float(rows), rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"normalized weight sum mismatch: {total_weight} != {rows}")
        os.replace(temporary, output_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise

    sum_squares = sum(counts[name] * normalized[name] ** 2 for name in counts)
    effective_sample_size = total_weight**2 / sum_squares
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "method": config["method"],
        "hypothesis": config["hypothesis"],
        "contract": {
            "only_modified_field": "reweight_rate",
            "official_reasoning_length_weight_removed": True,
            "stable_weight_ratio_idx3plus_to_idx0": (
                normalized["idx3plus"] / normalized["idx0"]
            ),
            "mean_weight_preserved_at_one": True,
            "ambiguous_weight_preserved_at_one": True,
            "locked_test_used": False,
            "external_data_used": False,
        },
        "inputs": {
            "pairs": {"path": str(pairs_path.resolve()), "sha256": pairs_sha},
            "provenance": {
                "path": str(provenance_path.resolve()),
                "sha256": provenance_sha,
            },
            "config": {"path": str(config_path.resolve()), "sha256": config_sha},
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": output_digest.hexdigest(),
            "bytes": output_path.stat().st_size,
        },
        "rows": rows,
        "changed_rows": changed_rows,
        "bucket_counts": dict(sorted(counts.items())),
        "stable_scale": stable_scale,
        "normalized_weights": normalized,
        "total_weight": total_weight,
        "effective_sample_size": effective_sample_size,
        "effective_sample_size_ratio": effective_sample_size / rows,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-pairs-sha256", required=True)
    parser.add_argument("--expected-provenance-sha256", required=True)
    parser.add_argument("--expected-rows", required=True, type=int)
    args = parser.parse_args()
    result = build(
        args.pairs,
        args.provenance,
        args.config,
        args.output,
        args.manifest,
        expected_pairs_sha256=args.expected_pairs_sha256,
        expected_provenance_sha256=args.expected_provenance_sha256,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
