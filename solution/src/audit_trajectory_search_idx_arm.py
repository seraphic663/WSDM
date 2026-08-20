#!/usr/bin/env python3
"""Independently audit a trajectory search-index soft-weight dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(source: Path, output: Path, provenance: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_rows = int(manifest["rows"])
    changed = 0
    neutral_changed = 0
    source_total = 0.0
    output_total = 0.0
    rows = 0
    immutable_fields = {
        "query",
        "pos",
        "pos_id",
        "neg",
        "neg_id",
        "reasoning_len",
        "satisfied",
    }

    with source.open(encoding="utf-8") as source_handle, output.open(
        encoding="utf-8"
    ) as output_handle, provenance.open(encoding="utf-8") as provenance_handle:
        while True:
            source_line = source_handle.readline()
            output_line = output_handle.readline()
            provenance_line = provenance_handle.readline()
            if not source_line and not output_line and not provenance_line:
                break
            if not source_line or not output_line or not provenance_line:
                raise ValueError("row count mismatch")
            source_row = json.loads(source_line)
            output_row = json.loads(output_line)
            provenance_row = json.loads(provenance_line)
            if provenance_row.get("row_index") != rows:
                raise ValueError(f"provenance row index mismatch at {rows}")
            if set(source_row) != set(output_row):
                raise ValueError(f"field set changed at row {rows}")
            for field in immutable_fields:
                if source_row.get(field) != output_row.get(field):
                    raise ValueError(f"{field} changed at row {rows}")
            source_weight = float(source_row["reweight_rate"])
            output_weight = float(output_row["reweight_rate"])
            if not math.isfinite(output_weight) or output_weight <= 0:
                raise ValueError(f"invalid output weight at row {rows}")
            source_total += source_weight
            output_total += output_weight
            if not math.isclose(source_weight, output_weight, rel_tol=0, abs_tol=1e-15):
                changed += 1
                if provenance_row.get("bucket") != "stable":
                    neutral_changed += 1
            rows += 1

    if rows != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {rows}")
    if changed != manifest["changed_rows"]:
        raise ValueError("changed-row count differs from manifest")
    if neutral_changed:
        raise ValueError("neutral provenance rows changed weight")
    if not math.isclose(source_total, output_total, rel_tol=0, abs_tol=1e-8):
        raise ValueError("total weight changed")
    if sha256_file(output) != manifest["outputs"]["train_sha256"]:
        raise ValueError("output SHA differs from manifest")
    return {
        "passed": True,
        "rows": rows,
        "changed_rows": changed,
        "neutral_changed_rows": neutral_changed,
        "source_total_weight": source_total,
        "output_total_weight": output_total,
        "output_sha256": sha256_file(output),
        "locked_test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit(args.source, args.output, args.provenance, args.manifest)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
