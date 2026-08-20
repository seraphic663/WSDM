#!/usr/bin/env python3
"""Build a training-pairs JSONL that differs only by unit reweight_rate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    expected_input_sha256: str,
    expected_rows: int,
) -> dict[str, Any]:
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite an existing unit-weight artifact")
    if input_path.is_symlink() or not input_path.is_file():
        raise ValueError(f"invalid input: {input_path}")
    actual_input_sha256 = sha256_file(input_path)
    if actual_input_sha256 != expected_input_sha256:
        raise ValueError(
            f"input SHA-256 mismatch: {actual_input_sha256} != {expected_input_sha256}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    rows = 0
    changed_rows = 0
    original_weight_sum = 0.0
    original_weight_min = math.inf
    original_weight_max = -math.inf
    output_digest = hashlib.sha256()
    try:
        with input_path.open("r", encoding="utf-8") as source, os.fdopen(
            fd, "wb"
        ) as target:
            for line_number, line in enumerate(source, 1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number}: expected JSON object")
                weight = row.get("reweight_rate")
                if (
                    not isinstance(weight, (int, float))
                    or isinstance(weight, bool)
                    or not math.isfinite(float(weight))
                    or float(weight) <= 0
                ):
                    raise ValueError(f"line {line_number}: invalid reweight_rate")
                numeric_weight = float(weight)
                original_weight_sum += numeric_weight
                original_weight_min = min(original_weight_min, numeric_weight)
                original_weight_max = max(original_weight_max, numeric_weight)
                if numeric_weight != 1.0:
                    changed_rows += 1
                row["reweight_rate"] = 1.0
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
        os.replace(temporary, output_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise

    output_sha256 = output_digest.hexdigest()
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "method": "replace only reweight_rate with 1.0",
        "contract": {
            "base_model": "Qwen/Qwen3-Embedding-0.6B",
            "source_fields_preserved_except": ["reweight_rate"],
            "external_data_used": False,
            "locked_test_used": False,
        },
        "input": {
            "path": str(input_path.resolve()),
            "sha256": actual_input_sha256,
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": output_sha256,
            "bytes": output_path.stat().st_size,
        },
        "rows": rows,
        "changed_rows": changed_rows,
        "original_reweight_rate": {
            "min": original_weight_min,
            "max": original_weight_max,
            "sum": original_weight_sum,
            "mean": original_weight_sum / rows,
        },
        "unit_reweight_rate": {
            "min": 1.0,
            "max": 1.0,
            "sum": float(rows),
            "mean": 1.0,
        },
    }
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-rows", required=True, type=int)
    args = parser.parse_args()
    result = build(
        args.input,
        args.output,
        args.manifest,
        expected_input_sha256=args.expected_input_sha256,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
