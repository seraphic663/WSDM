#!/usr/bin/env python3
"""Stream statistics for LRAT JSONL training pairs.

This script never loads the 3.88GB file into memory. Token counts are
whitespace-token approximations unless a tokenizer compatible with
transformers is supplied.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

WS = re.compile(r"\s+")


def norm_query(value: str) -> str:
    return WS.sub(" ", value.strip()).casefold()


def approx_tokens(value: str) -> int:
    return len(value.split())


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    total = 0
    invalid = 0
    empty_query = empty_pos = empty_neg = 0
    satisfied = Counter()
    duplicate_queries = 0
    seen_queries: set[str] = set()
    pos_neg_overlap = 0
    neg_counts: list[int] = []
    query_chars: list[int] = []
    pos_chars: list[int] = []
    neg_chars: list[int] = []
    query_tokens: list[int] = []
    pos_tokens: list[int] = []
    neg_tokens: list[int] = []
    weights: list[float] = []
    reasoning_lengths: list[float] = []
    fields = Counter()

    with args.input.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record is not an object")
            except Exception as exc:
                invalid += 1
                continue
            total += 1
            fields.update(row.keys())
            query = row.get("query") or ""
            pos = row.get("pos") or []
            neg = row.get("neg") or []
            pos_ids = set(map(str, row.get("pos_id") or []))
            neg_ids = set(map(str, row.get("neg_id") or []))
            if not query.strip():
                empty_query += 1
            qkey = norm_query(query)
            if qkey in seen_queries:
                duplicate_queries += 1
            seen_queries.add(qkey)
            if not pos:
                empty_pos += 1
            if not neg:
                empty_neg += 1
            if pos_ids & neg_ids:
                pos_neg_overlap += 1
            neg_counts.append(len(neg))
            query_chars.append(len(query))
            pos_chars.append(sum(len(str(x)) for x in pos))
            neg_chars.append(sum(len(str(x)) for x in neg))
            query_tokens.append(approx_tokens(query))
            pos_tokens.append(sum(approx_tokens(str(x)) for x in pos))
            neg_tokens.append(sum(approx_tokens(str(x)) for x in neg))
            value = row.get("satisfied")
            satisfied[str(value).lower()] += 1
            if isinstance(row.get("reweight_rate"), (int, float)):
                weights.append(float(row["reweight_rate"]))
            if isinstance(row.get("reasoning_len"), (int, float)):
                reasoning_lengths.append(float(row["reasoning_len"]))

    report = {
        "input": str(args.input),
        "records": total,
        "invalid_json_lines": invalid,
        "fields": sorted(fields),
        "empty": {"query": empty_query, "pos": empty_pos, "neg": empty_neg},
        "duplicate_normalized_queries_after_first": duplicate_queries,
        "unique_normalized_queries": len(seen_queries),
        "pos_neg_id_overlap_records": pos_neg_overlap,
        "satisfied_counts": dict(satisfied),
        "negative_count": describe([float(x) for x in neg_counts]),
        "query_char_length": describe([float(x) for x in query_chars]),
        "positive_char_length_total": describe([float(x) for x in pos_chars]),
        "negative_char_length_total": describe([float(x) for x in neg_chars]),
        "query_whitespace_token_approx": describe([float(x) for x in query_tokens]),
        "positive_whitespace_token_approx_total": describe([float(x) for x in pos_tokens]),
        "negative_whitespace_token_approx_total": describe([float(x) for x in neg_tokens]),
        "reasoning_len": describe(reasoning_lengths),
        "reweight_rate": describe(weights),
        "token_note": "Token lengths are whitespace approximations; install transformers and add tokenizer support before using them for model limits.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
