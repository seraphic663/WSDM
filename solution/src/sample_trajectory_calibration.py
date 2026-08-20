#!/usr/bin/env python3
"""Create deterministic, stratified manual-calibration samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from solution.src.build_trajectory_provenance import file_sha256


def stable_strata(record: dict[str, Any], long_steps: int, low_rank: int) -> list[str]:
    if record.get("bucket") != "stable":
        return []
    event = record["event"]
    values = []
    if not event.get("answer_token_subset"):
        values.append("answer_unmatched")
    if event.get("trajectory_steps", 0) >= long_steps:
        values.append("long_trajectory")
    if event.get("next_tool_name") == "search":
        values.append("continue_search")
    rank = event.get("retrieved_rank")
    if isinstance(rank, int) and rank >= low_rank:
        values.append("low_rank")
    if event.get("negative_cues"):
        values.append("explicit_negative_reasoning")
    if event.get("reasoning_len") == 0:
        values.append("empty_reasoning")
    if (
        event.get("negative_cues")
        and (
            not event.get("answer_token_subset")
            or (isinstance(rank, int) and rank >= low_rank)
        )
    ):
        values.append("strict_negative_combo")
    return values


def deterministic_key(seed: int, stratum: str, row_index: int) -> str:
    return hashlib.sha256(f"{seed}\0{stratum}\0{row_index}".encode()).hexdigest()


def select_samples(
    records: list[dict[str, Any]],
    *,
    samples_per_stratum: int,
    seed: int,
    long_steps: int,
    low_rank: int,
    only_strata: set[str] | None = None,
) -> list[dict[str, Any]]:
    predicates: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("explicit_negative_reasoning", lambda record: "explicit_negative_reasoning" in stable_strata(record, long_steps, low_rank)),
        ("empty_reasoning", lambda record: "empty_reasoning" in stable_strata(record, long_steps, low_rank)),
        ("answer_unmatched", lambda record: "answer_unmatched" in stable_strata(record, long_steps, low_rank)),
        ("long_trajectory", lambda record: "long_trajectory" in stable_strata(record, long_steps, low_rank)),
        ("continue_search", lambda record: "continue_search" in stable_strata(record, long_steps, low_rank)),
        ("low_rank", lambda record: "low_rank" in stable_strata(record, long_steps, low_rank)),
        ("strict_negative_combo", lambda record: "strict_negative_combo" in stable_strata(record, long_steps, low_rank)),
        ("control", lambda record: record.get("bucket") == "stable" and not stable_strata(record, long_steps, low_rank)),
    ]
    known_strata = {name for name, _ in predicates}
    if only_strata is not None:
        unknown = only_strata - known_strata
        if unknown:
            raise ValueError(f"unknown calibration strata: {sorted(unknown)}")
        predicates = [
            (name, predicate)
            for name, predicate in predicates
            if name in only_strata
        ]
    selected_rows: set[int] = set()
    selected = []
    for stratum, predicate in predicates:
        candidates = [
            record
            for record in records
            if record["row_index"] not in selected_rows and predicate(record)
        ]
        candidates.sort(
            key=lambda record: deterministic_key(seed, stratum, record["row_index"])
        )
        for record in candidates[:samples_per_stratum]:
            selected_rows.add(record["row_index"])
            selected.append(
                {
                    "sample_id": f"{stratum}-{record['row_index']:06d}",
                    "stratum": stratum,
                    "all_candidate_strata": stable_strata(record, long_steps, low_rank),
                    "row_index": record["row_index"],
                    "source_line": record["source_line"],
                    "query": record["query"],
                    "pos_id": record["pos_id"],
                    "pair_reasoning_len": record["reasoning_len"],
                    "pair_reweight_rate": record["reweight_rate"],
                    "negative_count": record["negative_count"],
                    "event": record["event"],
                    "manual_label": None,
                    "manual_reason": "",
                }
            )
    return selected


def add_pair_excerpts(samples: list[dict[str, Any]], pairs_path: Path) -> None:
    by_index = {sample["row_index"]: sample for sample in samples}
    seen = set()
    with pairs_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            sample = by_index.get(row_index)
            if sample is None:
                continue
            row = json.loads(line)
            sample["positive_excerpt"] = str(row["pos"][0])[:1200]
            sample["negative_id_sample"] = [str(value) for value in row.get("neg_id", [])[:5]]
            sample["negative_excerpt_sample"] = [
                str(value)[:400] for value in row.get("neg", [])[:2]
            ]
            seen.add(row_index)
    missing = set(by_index) - seen
    if missing:
        raise ValueError(f"selected row indexes missing from pair data: {sorted(missing)[:10]}")


def build(
    provenance_path: Path,
    pairs_path: Path,
    output_root: Path,
    *,
    samples_per_stratum: int,
    seed: int,
    long_steps: int,
    low_rank: int,
    only_strata: set[str] | None = None,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    if samples_per_stratum < 1:
        raise ValueError("samples_per_stratum must be positive")
    with provenance_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    selected = select_samples(
        records,
        samples_per_stratum=samples_per_stratum,
        seed=seed,
        long_steps=long_steps,
        low_rank=low_rank,
        only_strata=only_strata,
    )
    add_pair_excerpts(selected, pairs_path)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging.", dir=output_root.parent)
    )
    try:
        sample_path = staging / "calibration_samples.jsonl"
        with sample_path.open("x", encoding="utf-8") as handle:
            for sample in selected:
                handle.write(
                    json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        markdown_path = staging / "REVIEW_TEMPLATE.md"
        lines = [
            "# Trajectory quality manual calibration",
            "",
            "Allowed labels: `keep`, `bad_positive`, `uncertain`.",
            "",
            "| sample_id | stratum | row | manual_label | manual_reason |",
            "|---|---|---:|---|---|",
        ]
        for sample in selected:
            lines.append(
                f"| {sample['sample_id']} | {sample['stratum']} | {sample['row_index']} |  |  |"
            )
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts: dict[str, int] = {}
        for sample in selected:
            counts[sample["stratum"]] = counts.get(sample["stratum"], 0) + 1
        manifest = {
            "created_at": datetime.now().astimezone().isoformat(),
            "inputs": {
                "provenance": {
                    "path": str(provenance_path.resolve()),
                    "sha256": file_sha256(provenance_path),
                },
                "pairs": {
                    "path": str(pairs_path.resolve()),
                    "sha256": file_sha256(pairs_path),
                },
            },
            "contract": {
                "seed": seed,
                "samples_per_stratum": samples_per_stratum,
                "long_trajectory_min_steps": long_steps,
                "low_rank_min": low_rank,
                "labels": ["keep", "bad_positive", "uncertain"],
                "only_strata": sorted(only_strata) if only_strata else None,
                "locked_test_used": False,
            },
            "stratum_counts": counts,
            "sample_rows": len(selected),
            "outputs": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in (sample_path, markdown_path)
            },
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
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--samples-per-stratum", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--long-steps", type=int, default=126)
    parser.add_argument("--low-rank", type=int, default=8)
    parser.add_argument(
        "--only-strata",
        help="comma-separated subset of strata, for targeted supplemental review",
    )
    args = parser.parse_args()
    only_strata = (
        {value.strip() for value in args.only_strata.split(",") if value.strip()}
        if args.only_strata
        else None
    )
    result = build(
        args.provenance,
        args.pairs,
        args.output_root,
        samples_per_stratum=args.samples_per_stratum,
        seed=args.seed,
        long_steps=args.long_steps,
        low_rank=args.low_rank,
        only_strata=only_strata,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
