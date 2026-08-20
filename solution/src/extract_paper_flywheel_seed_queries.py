#!/usr/bin/env python3
"""Prepare and deterministically sample InfoSeekQA queries for paper flywheel loops.

The public LRAT trajectory archive contains multiple retriever-specific runs
over the same query IDs. This tool takes the union of those IDs, requires the
question text to agree wherever an ID appears more than once, and can use that
union to validate a complete InfoSeekQA source file. The paper states that 10K
queries are sampled at each loop but does not disclose its RNG or overlap
policy, so loop samples use a recorded SHA-256 ordering instead of claiming an
undisclosed exact split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tarfile
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(value: str) -> str:
    return " ".join(value.strip().split())


def extract_question(value: dict[str, Any]) -> str:
    direct = value.get("question") or value.get("query")
    if isinstance(direct, str) and direct.strip():
        return normalize_question(direct)
    raw_messages = value.get("raw_messages")
    if isinstance(raw_messages, list):
        for message in raw_messages:
            if (
                isinstance(message, dict)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
                and message["content"].strip()
            ):
                return normalize_question(message["content"])
    result = value.get("result")
    if isinstance(result, list):
        for step in result:
            if (
                isinstance(step, dict)
                and step.get("type") == "input_text"
                and isinstance(step.get("output"), str)
                and step["output"].strip()
            ):
                return normalize_question(step["output"])
    raise ValueError("trajectory lacks recoverable question text")


def query_sort_key(query_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(query_id))
    except ValueError:
        return (1, query_id)


def _atomic_write_tsv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def _read_source_questions(path: Path) -> list[str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"invalid source QA JSONL: {path}")
    rows: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(
                value.get("question"), str
            ):
                raise ValueError(f"invalid source QA row at line {line_number}")
            question = normalize_question(value["question"])
            if not question:
                raise ValueError(f"empty source question at line {line_number}")
            rows.append(question)
    return rows


def _read_archive_questions(
    archive: Path,
) -> tuple[dict[str, str], Counter[str], int, int]:
    if archive.is_symlink() or not archive.is_file():
        raise ValueError(f"invalid trajectory archive: {archive}")
    questions: dict[str, str] = {}
    group_counts: Counter[str] = Counter()
    duplicate_observations = 0
    json_members = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            json_members += 1
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            value = json.load(stream)
            if not isinstance(value, dict):
                raise ValueError(f"trajectory is not an object: {member.name}")
            query_id = value.get("query_id")
            if query_id is None:
                raise ValueError(f"trajectory lacks query_id: {member.name}")
            query_id = str(query_id)
            question = extract_question(value)
            group = member.name.split("/")[1] if "/" in member.name else "(root)"
            group_counts[group] += 1
            previous = questions.get(query_id)
            if previous is not None:
                duplicate_observations += 1
                if previous != question:
                    raise ValueError(
                        f"query text conflict for id {query_id}: "
                        f"{previous!r} != {question!r}"
                    )
            else:
                questions[query_id] = question
    return questions, group_counts, duplicate_observations, json_members


def extract(
    archive: Path,
    output_tsv: Path,
    manifest_path: Path,
    *,
    expected_count: int | None = 10_000,
    source_qa_jsonl: Path | None = None,
    source_revision: str | None = None,
) -> dict[str, Any]:
    if output_tsv.exists() or manifest_path.exists():
        raise FileExistsError("output TSV and manifest must not already exist")
    if archive.is_symlink() or not archive.is_file():
        raise ValueError(f"invalid trajectory archive: {archive}")

    (
        questions,
        group_counts,
        duplicate_observations,
        json_members,
    ) = _read_archive_questions(archive)

    source_details = None
    if source_qa_jsonl is not None:
        source_rows = _read_source_questions(source_qa_jsonl)
        if expected_count is None:
            raise ValueError("expected_count is required with source QA JSONL")
        if len(source_rows) < expected_count:
            raise ValueError(
                f"source QA has only {len(source_rows)} rows; need {expected_count}"
            )
        for query_id, question in questions.items():
            try:
                index = int(query_id)
            except ValueError as exc:
                raise ValueError(
                    f"non-numeric archive query ID cannot map to source: {query_id}"
                ) from exc
            if index < 0 or index >= expected_count:
                raise ValueError(
                    f"archive query ID is outside selected seed range: {query_id}"
                )
            if source_rows[index] != question:
                raise ValueError(
                    f"archive/source question mismatch for id {query_id}"
                )
        rows = [(str(index), source_rows[index]) for index in range(expected_count)]
        source_details = {
            "path": str(source_qa_jsonl.resolve()),
            "bytes": source_qa_jsonl.stat().st_size,
            "sha256": sha256_file(source_qa_jsonl),
            "revision": source_revision,
            "rows": len(source_rows),
            "selection": f"zero-based rows 0 through {expected_count - 1}",
            "archive_subset_validated": len(questions),
        }
    else:
        if expected_count is not None and len(questions) != expected_count:
            raise ValueError(
                f"unexpected unique query count: {len(questions)} != {expected_count}"
            )
        rows = sorted(questions.items(), key=lambda item: query_sort_key(item[0]))
    _atomic_write_tsv(output_tsv, rows)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "method": "union official LRAT trajectory query IDs and require identical recovered question text",
        "archive": str(archive.resolve()),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "archive_json_members": json_members,
        "group_counts": dict(sorted(group_counts.items())),
        "duplicate_observations": duplicate_observations,
        "archive_unique_query_count": len(questions),
        "unique_query_count": len(rows),
        "expected_query_count": expected_count,
        "output_tsv": str(output_tsv.resolve()),
        "output_bytes": output_tsv.stat().st_size,
        "output_sha256": sha256_file(output_tsv),
        "source_qa": source_details,
        "external_data_used": source_qa_jsonl is not None,
        "competition_submission_eligible": False,
        "research_purpose": "paper data-flywheel reproduction only",
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def build_source_pool(
    archive: Path,
    source_qa_jsonl: Path,
    output_tsv: Path,
    manifest_path: Path,
    *,
    source_revision: str,
    minimum_count: int = 10_000,
) -> dict[str, Any]:
    if output_tsv.exists() or manifest_path.exists():
        raise FileExistsError("output TSV and manifest must not already exist")
    source_rows = _read_source_questions(source_qa_jsonl)
    if len(source_rows) < minimum_count:
        raise ValueError(
            f"source QA has only {len(source_rows)} rows; need {minimum_count}"
        )
    (
        archive_questions,
        group_counts,
        duplicate_observations,
        json_members,
    ) = _read_archive_questions(archive)
    for query_id, question in archive_questions.items():
        try:
            index = int(query_id)
        except ValueError as exc:
            raise ValueError(
                f"non-numeric archive query ID cannot map to source: {query_id}"
            ) from exc
        if index < 0 or index >= len(source_rows):
            raise ValueError(f"archive query ID is outside source: {query_id}")
        if source_rows[index] != question:
            raise ValueError(
                f"archive/source question mismatch for id {query_id}"
            )
    rows = [(str(index), question) for index, question in enumerate(source_rows)]
    _atomic_write_tsv(output_tsv, rows)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "method": "full InfoSeekQA source pool validated against all recoverable official LRAT trajectory queries",
        "archive": str(archive.resolve()),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "archive_json_members": json_members,
        "archive_unique_query_count": len(archive_questions),
        "group_counts": dict(sorted(group_counts.items())),
        "duplicate_observations": duplicate_observations,
        "source_qa": {
            "path": str(source_qa_jsonl.resolve()),
            "bytes": source_qa_jsonl.stat().st_size,
            "sha256": sha256_file(source_qa_jsonl),
            "revision": source_revision,
            "rows": len(source_rows),
        },
        "pool_rows": len(rows),
        "output_tsv": str(output_tsv.resolve()),
        "output_bytes": output_tsv.stat().st_size,
        "output_sha256": sha256_file(output_tsv),
        "external_data_used": True,
        "competition_submission_eligible": False,
        "research_purpose": "paper data-flywheel reproduction only",
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def sample_loop_queries(
    pool_tsv: Path,
    output_tsv: Path,
    manifest_path: Path,
    *,
    count: int,
    seed: int,
    loop_number: int,
) -> dict[str, Any]:
    if output_tsv.exists() or manifest_path.exists():
        raise FileExistsError("loop TSV and manifest must not already exist")
    if count <= 0 or loop_number < 0:
        raise ValueError("count must be positive and loop_number non-negative")
    rows: list[tuple[str, str]] = []
    with pool_tsv.open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if len(row) < 2 or not row[0].strip() or not row[1].strip():
                raise ValueError(f"malformed pool TSV row {line_number}")
            rows.append((row[0].strip(), normalize_question(row[1])))
    if len(rows) < count:
        raise ValueError(f"query pool has only {len(rows)} rows; need {count}")

    def order_key(item: tuple[str, str]) -> tuple[str, str]:
        query_id, _ = item
        digest = hashlib.sha256(
            f"paper-flywheel-v1:{seed}:{loop_number}:{query_id}".encode()
        ).hexdigest()
        return digest, query_id

    selected = sorted(rows, key=order_key)[:count]
    selected.sort(key=lambda item: query_sort_key(item[0]))
    _atomic_write_tsv(output_tsv, selected)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "method": "take the lowest SHA-256 order keys for (seed, loop, query_id), without replacement within the loop",
        "paper_disclosure": "paper states 10K queries are sampled at each step but does not disclose RNG or cross-loop overlap policy",
        "pool_tsv": str(pool_tsv.resolve()),
        "pool_bytes": pool_tsv.stat().st_size,
        "pool_sha256": sha256_file(pool_tsv),
        "pool_rows": len(rows),
        "sample_count": count,
        "seed": seed,
        "loop_number": loop_number,
        "output_tsv": str(output_tsv.resolve()),
        "output_bytes": output_tsv.stat().st_size,
        "output_sha256": sha256_file(output_tsv),
        "competition_submission_eligible": False,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["archive-union", "source-pool", "sample-loop"],
        default="archive-union",
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=10_000)
    parser.add_argument("--source-qa-jsonl", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--pool-tsv", type=Path)
    parser.add_argument("--sample-count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--loop-number", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "source-pool":
        if (
            args.archive is None
            or args.source_qa_jsonl is None
            or not args.source_revision
        ):
            parser.error(
                "--archive, --source-qa-jsonl and --source-revision are required for source-pool"
            )
        result = build_source_pool(
            args.archive,
            args.source_qa_jsonl,
            args.output_tsv,
            args.manifest,
            source_revision=args.source_revision,
            minimum_count=args.expected_count,
        )
    elif args.mode == "sample-loop":
        if args.pool_tsv is None:
            parser.error("--pool-tsv is required for sample-loop")
        result = sample_loop_queries(
            args.pool_tsv,
            args.output_tsv,
            args.manifest,
            count=args.sample_count,
            seed=args.seed,
            loop_number=args.loop_number,
        )
    else:
        if args.archive is None:
            parser.error("--archive is required for archive-union")
        result = extract(
            args.archive,
            args.output_tsv,
            args.manifest,
            expected_count=args.expected_count,
            source_qa_jsonl=args.source_qa_jsonl,
            source_revision=args.source_revision,
        )
    print(
        json.dumps(result, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
