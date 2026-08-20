#!/usr/bin/env python3
"""Build a query-unique, traceable multi-positive LRAT training set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE = Path("/root/data/LRAT/ccir/data/experiments/early_stop_v1/train.jsonl")
DEFAULT_OUTPUT = Path("/root/data/LRAT/ccir/data/experiments/m09_multipositive_v2/train.jsonl")
DEFAULT_MANIFEST = Path("/root/data/LRAT/ccir/data/experiments/m09_multipositive_v2/manifest.json")
EXPECTED_SOURCE_SHA256 = "158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9"


def normalize_query(value: str) -> str:
    return " ".join(value.strip().lower().split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired_documents(row: dict, text_key: str, id_key: str) -> list[tuple[str, str]]:
    texts = row.get(text_key)
    ids = row.get(id_key)
    if not isinstance(texts, list) or not texts:
        raise ValueError(f"{text_key} must be a non-empty list")
    if not isinstance(ids, list) or len(ids) != len(texts):
        raise ValueError(f"{id_key} must align with {text_key}")
    pairs: list[tuple[str, str]] = []
    for doc_id, text in zip(ids, texts):
        if not isinstance(text, str) or not text:
            raise ValueError(f"{text_key} contains an empty/non-string document")
        pairs.append((str(doc_id), text))
    return pairs


def merge_query_group(rows: list[dict]) -> tuple[dict, dict]:
    if not rows:
        raise ValueError("cannot merge an empty query group")
    normalized = {normalize_query(row.get("query", "")) for row in rows}
    if len(normalized) != 1:
        raise ValueError("query group is not normalized-query consistent")
    if "" in normalized and len(rows) != 1:
        raise ValueError("empty queries must remain source-row unique")

    positives: OrderedDict[tuple[str, str], None] = OrderedDict()
    negatives: OrderedDict[tuple[str, str], None] = OrderedDict()
    weights: list[float] = []
    reasoning_lengths: list[int] = []
    satisfied_values: list[bool] = []

    for row in rows:
        for pair in _paired_documents(row, "pos", "pos_id"):
            positives.setdefault(pair, None)
        for pair in _paired_documents(row, "neg", "neg_id"):
            negatives.setdefault(pair, None)
        weight = float(row.get("reweight_rate", 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("reweight_rate must be finite and non-negative")
        weights.append(weight)
        reasoning_lengths.append(int(row.get("reasoning_len", 0)))
        satisfied_values.append(bool(row.get("satisfied", False)))

    positive_ids = {doc_id for doc_id, _ in positives}
    positive_texts = {text for _, text in positives}
    filtered_negatives = OrderedDict(
        (pair, None)
        for pair in negatives
        if pair[0] not in positive_ids and pair[1] not in positive_texts
    )
    if not filtered_negatives:
        raise ValueError("query group has no negative document after positive-conflict removal")

    merged = {
        "query": rows[0]["query"],
        "pos": [text for _, text in positives],
        "neg": [text for _, text in filtered_negatives],
        "pos_id": [doc_id for doc_id, _ in positives],
        "neg_id": [doc_id for doc_id, _ in filtered_negatives],
        "reasoning_len": max(reasoning_lengths),
        "satisfied": all(satisfied_values),
        "reweight_rate": sum(weights) / len(weights),
        "_num_positives": len(positives),
        "_source_rows": len(rows),
    }
    stats = {
        "source_rows": len(rows),
        "positive_count": len(positives),
        "negative_count_before_filter": len(negatives),
        "negative_count_after_filter": len(filtered_negatives),
        "removed_positive_conflicts": len(negatives) - len(filtered_negatives),
    }
    return merged, stats


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield value


def build(source: Path, output: Path, manifest_path: Path, expected_source_sha256: str) -> dict:
    if output.exists() or manifest_path.exists():
        raise FileExistsError("output and manifest must not already exist")
    source_sha256 = sha256_file(source)
    if source_sha256 != expected_source_sha256:
        raise ValueError(f"source SHA mismatch: {source_sha256}")

    groups: OrderedDict[tuple[str, int | None], list[dict]] = OrderedDict()
    source_rows = 0
    empty_query_rows = 0
    for row in iter_jsonl(source):
        query_key = normalize_query(row.get("query", ""))
        source_row_index = source_rows + 1
        if query_key:
            group_key = (query_key, None)
        else:
            group_key = ("", source_row_index)
            empty_query_rows += 1
        groups.setdefault(group_key, []).append(row)
        source_rows += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.tmp.", dir=output.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    positive_histogram: dict[int, int] = {}
    multi_positive_rows = 0
    total_positive_signals = 0
    removed_positive_conflicts = 0
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            for rows in groups.values():
                merged, stats = merge_query_group(rows)
                count = stats["positive_count"]
                positive_histogram[count] = positive_histogram.get(count, 0) + 1
                multi_positive_rows += int(count > 1)
                total_positive_signals += count
                removed_positive_conflicts += stats["removed_positive_conflicts"]
                handle.write(json.dumps(merged, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temp_path, output)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    manifest = {
        "method": "query-unique multi-positive InfoNCE",
        "source": str(source.resolve()),
        "source_sha256": source_sha256,
        "source_rows": source_rows,
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "output_bytes": output.stat().st_size,
        "output_rows": len(groups),
        "nonempty_unique_normalized_queries": len(groups) - empty_query_rows,
        "empty_query_rows_preserved_individually": empty_query_rows,
        "multi_positive_rows": multi_positive_rows,
        "total_positive_signals": total_positive_signals,
        "removed_within_query_positive_negative_conflicts": removed_positive_conflicts,
        "positive_count_histogram": {str(key): positive_histogram[key] for key in sorted(positive_histogram)},
        "max_positives_per_training_sample": 3,
        "locked_test_used": False,
    }
    fd, temp_name = tempfile.mkstemp(prefix=f".{manifest_path.name}.tmp.", dir=manifest_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, manifest_path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.manifest, args.expected_source_sha256), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
