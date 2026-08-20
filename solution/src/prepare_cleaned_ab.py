#!/usr/bin/env python3
"""Prepare full paired train arms and a deterministic query-held-out diagnostic set."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


WS = re.compile(r"\s+")


def norm_query(value: str) -> str:
    return WS.sub(" ", value.strip()).casefold()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def held_out(query_key: str, threshold: int) -> bool:
    if not query_key:
        return False
    return int.from_bytes(hashlib.sha256(query_key.encode()).digest()[:2], "big") < threshold


def validate_pair(raw: dict, cleaned: dict, line_number: int) -> tuple[str, bool, bool]:
    if not isinstance(raw, dict) or not isinstance(cleaned, dict):
        raise ValueError(f"line {line_number}: rows must be JSON objects")
    query = raw.get("query")
    if not isinstance(query, str):
        raise ValueError(f"line {line_number}: query must be a string")
    if cleaned.get("query") != query:
        raise ValueError(f"line {line_number}: raw/cleaned query mismatch")
    for row_name, row in (("raw", raw), ("cleaned", cleaned)):
        for text_key, id_key in (("pos", "pos_id"), ("neg", "neg_id")):
            if not isinstance(row.get(text_key), list) or not isinstance(row.get(id_key), list):
                raise ValueError(f"line {line_number}: invalid {row_name} {text_key}/{id_key}")
            if not row[text_key] or len(row[text_key]) != len(row[id_key]):
                raise ValueError(f"line {line_number}: empty or misaligned {row_name} {text_key}/{id_key}")
    raw_fixed = {key: value for key, value in raw.items() if key not in {"neg", "neg_id"}}
    cleaned_fixed = {key: value for key, value in cleaned.items() if key not in {"neg", "neg_id"}}
    if raw_fixed != cleaned_fixed:
        raise ValueError(f"line {line_number}: cleaning changed non-negative fields")
    raw_neg_ids = list(map(str, raw["neg_id"]))
    cleaned_neg_ids = list(map(str, cleaned["neg_id"]))
    if not set(cleaned_neg_ids).issubset(raw_neg_ids):
        raise ValueError(f"line {line_number}: cleaned negatives are not a raw subset")
    return norm_query(query), raw_neg_ids != cleaned_neg_ids, len(cleaned_neg_ids) < 5


def prepare(raw_path: Path, cleaned_path: Path, output_root: Path, split_threshold: int) -> dict:
    for path in (raw_path, cleaned_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {output_root}")
    if not 1 <= split_threshold <= 65535:
        raise ValueError("split_threshold must be between 1 and 65535")

    raw_digest = hashlib.sha256()
    cleaned_digest = hashlib.sha256()
    source_rows = 0
    source_changed = 0
    empty_query_rows = 0
    heldout_source_rows = 0
    heldout_groups: dict[str, dict] = {}
    short_query_groups: set[str] = set()
    group_rows: collections.Counter[str] = collections.Counter()

    with raw_path.open("rb") as raw_handle, cleaned_path.open("rb") as cleaned_handle:
        for line_number, (raw_line, cleaned_line) in enumerate(zip(raw_handle, cleaned_handle), 1):
            source_rows += 1
            raw_digest.update(raw_line)
            cleaned_digest.update(cleaned_line)
            raw = json.loads(raw_line)
            cleaned = json.loads(cleaned_line)
            query_key, changed, below_five = validate_pair(raw, cleaned, line_number)
            source_changed += int(changed)
            empty_query_rows += int(not query_key)
            group_rows[query_key] += 1
            if below_five:
                short_query_groups.add(query_key)
            if not held_out(query_key, split_threshold):
                continue
            heldout_source_rows += 1
            group = heldout_groups.setdefault(
                query_key,
                {
                    "query": raw["query"],
                    "source_lines": [],
                    "positives": {},
                    "negatives": {},
                },
            )
            group["source_lines"].append(line_number)
            for identifier, text in zip(raw["pos_id"], raw["pos"]):
                group["positives"].setdefault(str(identifier), text)
            for identifier, text in zip(raw["neg_id"], raw["neg"]):
                group["negatives"].setdefault(str(identifier), text)
        if raw_handle.readline() or cleaned_handle.readline():
            raise ValueError("raw and cleaned row counts differ")
    if not heldout_groups:
        raise ValueError("query hash split produced no held-out groups")
    if "" in short_query_groups:
        raise ValueError("empty-query row unexpectedly has fewer than five cleaned negatives")

    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging.", dir=output_root.parent))
    try:
        raw_output = staging / "raw_train.jsonl"
        cleaned_output = staging / "cleaned_train.jsonl"
        diag_output = staging / "query_heldout_diag.jsonl"
        train_rows = 0
        train_changed = 0
        short_group_rows_excluded = 0
        heldout_rows_excluded = 0
        train_queries: set[str] = set()
        with raw_path.open("rb") as raw_handle, cleaned_path.open("rb") as cleaned_handle, raw_output.open("xb") as raw_dest, cleaned_output.open("xb") as clean_dest:
            for line_number, (raw_line, cleaned_line) in enumerate(zip(raw_handle, cleaned_handle), 1):
                raw = json.loads(raw_line)
                cleaned = json.loads(cleaned_line)
                query_key, changed, _ = validate_pair(raw, cleaned, line_number)
                if held_out(query_key, split_threshold):
                    heldout_rows_excluded += 1
                    continue
                if query_key in short_query_groups:
                    short_group_rows_excluded += 1
                    continue
                raw_dest.write(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")
                clean_dest.write(cleaned_line if cleaned_line.endswith(b"\n") else cleaned_line + b"\n")
                train_rows += 1
                train_changed += int(changed)
                train_queries.add(query_key)

        diag_rows = 0
        diag_passages = 0
        diag_max_candidates = 0
        diag_provenance = []
        with diag_output.open("x", encoding="utf-8") as handle:
            for query_key in sorted(heldout_groups):
                group = heldout_groups[query_key]
                positives = group["positives"]
                negatives = {identifier: text for identifier, text in group["negatives"].items() if identifier not in positives}
                if not positives or not negatives:
                    raise ValueError(f"held-out query lacks candidates: {hashlib.sha256(query_key.encode()).hexdigest()}")
                row = {
                    "query": group["query"],
                    "pos_id": list(positives),
                    "pos": list(positives.values()),
                    "neg_id": list(negatives),
                    "neg": list(negatives.values()),
                }
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                candidates = len(positives) + len(negatives)
                diag_rows += 1
                diag_passages += candidates
                diag_max_candidates = max(diag_max_candidates, candidates)
                diag_provenance.append(
                    {
                        "query_sha256": hashlib.sha256(group["query"].encode()).hexdigest(),
                        "normalized_query_sha256": hashlib.sha256(query_key.encode()).hexdigest(),
                        "source_lines": group["source_lines"],
                        "positive_count": len(positives),
                        "negative_count": len(negatives),
                    }
                )

        overlap = train_queries.intersection(heldout_groups)
        if overlap or heldout_rows_excluded != heldout_source_rows:
            raise AssertionError((len(overlap), heldout_rows_excluded, heldout_source_rows))
        expected_train = source_rows - heldout_source_rows - short_group_rows_excluded
        if train_rows != expected_train:
            raise AssertionError((train_rows, expected_train))
        report = {
            "created_at": datetime.now().astimezone().isoformat(),
            "split": {
                "method": "normalized-query SHA-256 first uint16 below threshold",
                "threshold": split_threshold,
                "denominator": 65536,
            },
            "source_rows": source_rows,
            "source_changed_rows": source_changed,
            "empty_query_rows_retained_in_train": empty_query_rows,
            "train_rows_per_arm": train_rows,
            "train_changed_rows": train_changed,
            "heldout_query_groups": diag_rows,
            "heldout_source_rows": heldout_source_rows,
            "heldout_passages": diag_passages,
            "heldout_max_candidates": diag_max_candidates,
            "normalized_query_overlap": len(overlap),
            "short_negative_query_groups_excluded_from_both_arms": len(short_query_groups),
            "short_negative_group_source_rows_excluded": short_group_rows_excluded,
            "short_negative_policy": "exclude each complete normalized-query group from both short-training arms; full cleaned epoch retains all official rows and uses FlagEmbedding repeat sampling",
            "inputs": {
                "raw": {"path": str(raw_path.resolve()), "bytes": raw_path.stat().st_size, "sha256": raw_digest.hexdigest()},
                "cleaned": {"path": str(cleaned_path.resolve()), "bytes": cleaned_path.stat().st_size, "sha256": cleaned_digest.hexdigest()},
            },
            "outputs": {
                "raw_train": {"path": str((output_root / raw_output.name).absolute()), "rows": train_rows, "bytes": raw_output.stat().st_size, "sha256": sha256(raw_output)},
                "cleaned_train": {"path": str((output_root / cleaned_output.name).absolute()), "rows": train_rows, "bytes": cleaned_output.stat().st_size, "sha256": sha256(cleaned_output)},
                "query_heldout_diag": {"path": str((output_root / diag_output.name).absolute()), "rows": diag_rows, "bytes": diag_output.stat().st_size, "sha256": sha256(diag_output)},
            },
            "diag_provenance": diag_provenance,
        }
        (staging / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, output_root)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cleaned", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split-threshold", type=int, default=405)
    args = parser.parse_args()
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(prepare(args.raw, args.cleaned, args.output_root, args.split_threshold), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
