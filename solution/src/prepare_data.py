#!/usr/bin/env python3
"""Create deterministic, query-disjoint smoke train/dev JSONL splits."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

WS = re.compile(r"\s+")


def norm_query(value: str) -> str:
    return WS.sub(" ", value.strip()).casefold()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--train-size", type=int, default=5000)
    ap.add_argument("--dev-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260714)
    args = ap.parse_args()

    # Keep only the first record for each normalized query, storing offsets
    # instead of full 3.88GB records in memory.
    offsets: dict[str, tuple[int, int]] = {}
    with args.input.open("rb") as fh:
        while True:
            start = fh.tell()
            line = fh.readline()
            if not line:
                break
            try:
                row = json.loads(line)
                key = norm_query(str(row.get("query") or ""))
            except Exception:
                continue
            if key and key not in offsets:
                offsets[key] = (start, len(line))

    keys = list(offsets)
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    needed = args.train_size + args.dev_size
    if needed > len(keys):
        raise SystemExit(f"requested {needed} records, only {len(keys)} unique queries available")
    dev_keys = set(keys[: args.dev_size])
    train_keys = set(keys[args.dev_size : needed])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    dev_path = args.output_dir / "dev.jsonl"
    with args.input.open("rb") as src, train_path.open("wb") as train, dev_path.open("wb") as dev:
        for key, (offset, length) in offsets.items():
            if key not in train_keys and key not in dev_keys:
                continue
            src.seek(offset)
            line = src.read(length)
            (dev if key in dev_keys else train).write(line)
    print(f"unique_queries={len(keys)}")
    print(f"train={len(train_keys)} -> {train_path}")
    print(f"dev={len(dev_keys)} -> {dev_path}")
    print(f"seed={args.seed}; normalized query overlap=0")


if __name__ == "__main__":
    main()
