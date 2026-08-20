#!/usr/bin/env python3
"""Prepare a bounded, maximum-length paper-training smoke fixture.

The fixture has exactly one global microbatch for the default two-GPU paper
profile: 32 queries, each with one positive and nine negatives. Text is long
enough to hit the configured 512-token query and passage caps. The output must
stay in the research-only smoke tree and is never competition training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RESEARCH_SMOKE_ROOT = Path(
    "/root/data/LRAT/ccir/research/paper_flywheel_v1/smoke"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--rows",
        type=int,
        default=32,
        help="must be a positive multiple of the exact global microbatch (32)",
    )
    args = parser.parse_args()
    if args.rows < 32 or args.rows % 32:
        raise ValueError("--rows must be a positive multiple of 32")
    output = args.output.resolve()
    if RESEARCH_SMOKE_ROOT.resolve() not in output.parents:
        raise ValueError("output must stay under the research smoke root")
    if output.exists():
        raise FileExistsError(output)

    # Repetition deliberately exceeds 512 tokens after the retrieval
    # instruction is added, so the trainer exercises both configured caps.
    long_query = " ".join(f"querytoken{i % 97}" for i in range(700))
    long_passage = " ".join(f"passagetoken{i % 101}" for i in range(700))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for index in range(args.rows):
            row = {
                "query": f"smoke query {index} {long_query}",
                "pos": [f"positive {index} {long_passage}"],
                "neg": [
                    f"negative {index}-{negative} {long_passage}"
                    for negative in range(9)
                ],
                "reweight_rate": 0.5 + index / max(args.rows - 1, 1),
                "reasoning_len": 64 + index,
                "satisfied": True,
                "source_trajectory": f"synthetic_smoke_{index}.json",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output": str(output),
                "rows": args.rows,
                "positives_per_row": 1,
                "negatives_per_row": 9,
                "research_only": True,
                "competition_submission_eligible": False,
                "sha256": sha256(output),
                "bytes": output.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
