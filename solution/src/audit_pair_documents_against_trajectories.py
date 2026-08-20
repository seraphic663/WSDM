#!/usr/bin/env python3
"""Trace every pair document id/text to official trajectory tool output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DOCUMENT_OUTPUT = re.compile(r"^Document\s+([^:\n]+):\n(.*)\Z", re.DOTALL)
BROWSE_TOOLS = {"get_document", "visit"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_document_text(text: str) -> str:
    """Remove transport-only trailing whitespace without changing content."""
    return text.rstrip()


def parse_arguments(step: dict[str, Any]) -> dict[str, Any]:
    value = step.get("arguments")
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def argument_doc_id(step: dict[str, Any]) -> str | None:
    arguments = parse_arguments(step)
    value = arguments.get("docid", arguments.get("id"))
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    return str(value).split(":")[-1]


def parse_document_output(step: dict[str, Any]) -> tuple[str, str] | None:
    if step.get("type") != "tool_call" or step.get("tool_name") not in BROWSE_TOOLS:
        return None
    output = step.get("output")
    if not isinstance(output, str):
        return None
    match = DOCUMENT_OUTPUT.fullmatch(output.rstrip("\n"))
    if match is None:
        return None
    header_id = match.group(1).strip().split(":")[-1]
    argument_id = argument_doc_id(step)
    if argument_id is not None and header_id != argument_id:
        raise ValueError(f"document id differs between arguments and output: {argument_id} != {header_id}")
    return header_id, match.group(2)


def iter_trajectory_documents(
    archive: Path,
    issue_counts: Counter[str] | None = None,
    issue_samples: list[dict[str, Any]] | None = None,
) -> Iterable[tuple[str, str, str, int]]:
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            extracted = bundle.extractfile(member)
            if extracted is None:
                continue
            value = json.load(extracted)
            if not isinstance(value, dict):
                continue
            steps = value.get("result") or value.get("turns") or []
            if not isinstance(steps, list):
                continue
            for step_index, step in enumerate(steps):
                if not isinstance(step, dict):
                    continue
                try:
                    parsed = parse_document_output(step)
                except ValueError as error:
                    if issue_counts is not None:
                        issue_counts["argument_output_doc_id_mismatch"] += 1
                    if issue_samples is not None and len(issue_samples) < 100:
                        issue_samples.append(
                            {
                                "trajectory_path": member.name,
                                "step_index": step_index,
                                "error": str(error),
                            }
                        )
                    continue
                if parsed is not None:
                    yield parsed[0], parsed[1], member.name, step_index


def validate_pair_row(row: dict[str, Any], line_number: int) -> None:
    for text_field, id_field in (("pos", "pos_id"), ("neg", "neg_id")):
        texts = row.get(text_field)
        identifiers = row.get(id_field)
        if not isinstance(texts, list) or not isinstance(identifiers, list) or len(texts) != len(identifiers):
            raise ValueError(f"line {line_number}: {text_field}/{id_field} are not aligned lists")
        if not all(isinstance(text, str) for text in texts):
            raise ValueError(f"line {line_number}: {text_field} contains non-string text")


def audit(
    archive: Path,
    pairs: Path,
    *,
    expected_rows: int | None = None,
    expected_archive_sha256: str | None = None,
    expected_pairs_sha256: str | None = None,
) -> dict[str, Any]:
    archive_sha = sha256_file(archive)
    pairs_sha = sha256_file(pairs)
    if expected_archive_sha256 and archive_sha != expected_archive_sha256:
        raise ValueError("trajectory archive SHA-256 mismatch")
    if expected_pairs_sha256 and pairs_sha != expected_pairs_sha256:
        raise ValueError("pair data SHA-256 mismatch")

    document_hashes: dict[str, set[str]] = defaultdict(set)
    document_examples: dict[tuple[str, str], dict[str, Any]] = {}
    parse_issue_counts: Counter[str] = Counter()
    parse_issue_samples: list[dict[str, Any]] = []
    parsed_events = 0
    for doc_id, text, trajectory_path, step_index in iter_trajectory_documents(
        archive,
        parse_issue_counts,
        parse_issue_samples,
    ):
        text = canonical_document_text(text)
        text_sha = hashlib.sha256(text.encode()).hexdigest()
        document_hashes[doc_id].add(text_sha)
        document_examples.setdefault(
            (doc_id, text_sha),
            {
                "trajectory_path": trajectory_path,
                "step_index": step_index,
                "text_chars": len(text),
            },
        )
        parsed_events += 1

    rows = 0
    references = Counter()
    matched_references = Counter()
    referenced_ids: set[str] = set()
    matched_ids: set[str] = set()
    unresolved_counts = Counter()
    unresolved_samples: list[dict[str, Any]] = []
    with pairs.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: pair row is not an object")
            validate_pair_row(row, line_number)
            rows += 1
            for role, text_field, id_field in (
                ("positive", "pos", "pos_id"),
                ("negative", "neg", "neg_id"),
            ):
                for text, raw_doc_id in zip(row[text_field], row[id_field]):
                    doc_id = str(raw_doc_id)
                    canonical_text = canonical_document_text(text)
                    text_sha = hashlib.sha256(canonical_text.encode()).hexdigest()
                    references[role] += 1
                    referenced_ids.add(doc_id)
                    hashes = document_hashes.get(doc_id)
                    if hashes and text_sha in hashes:
                        matched_references[role] += 1
                        matched_ids.add(doc_id)
                        continue
                    reason = "doc_id_absent_from_trajectory_outputs" if not hashes else "doc_id_present_but_text_differs"
                    unresolved_counts[(role, reason)] += 1
                    if len(unresolved_samples) < 100:
                        unresolved_samples.append(
                            {
                                "line_number": line_number,
                                "role": role,
                                "doc_id": doc_id,
                                "text_sha256": text_sha,
                                "text_chars": len(canonical_text),
                                "reason": reason,
                                "known_text_sha256": sorted(hashes or [])[:10],
                            }
                        )
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {rows}")

    total_references = sum(references.values())
    total_matched = sum(matched_references.values())
    unresolved = total_references - total_matched
    variant_counts = Counter(len(value) for value in document_hashes.values())
    return {
        "schema_version": 1,
        "contract": {
            "document_identity": "exact doc_id plus SHA-256 after removing transport-only trailing whitespace",
            "trajectory_source": "Document <id> tool output from get_document or visit",
            "external_data_used": False,
            "external_api_used": False,
            "locked_test_used": False,
        },
        "inputs": {
            "trajectory_archive": {
                "path": str(archive.resolve()),
                "bytes": archive.stat().st_size,
                "sha256": archive_sha,
            },
            "pairs": {
                "path": str(pairs.resolve()),
                "bytes": pairs.stat().st_size,
                "sha256": pairs_sha,
                "rows": rows,
            },
        },
        "trajectory_documents": {
            "parsed_tool_outputs": parsed_events,
            "unique_doc_ids": len(document_hashes),
            "unique_doc_id_text_pairs": len(document_examples),
            "text_variants_per_doc_id": {
                str(variants): count for variants, count in sorted(variant_counts.items())
            },
            "parse_issue_counts": dict(parse_issue_counts),
            "parse_issue_samples": parse_issue_samples,
        },
        "pair_references": {
            "total": total_references,
            "by_role": dict(references),
            "matched_total": total_matched,
            "matched_by_role": dict(matched_references),
            "unresolved_total": unresolved,
            "unique_doc_ids": len(referenced_ids),
            "matched_unique_doc_ids": len(matched_ids),
            "unresolved_unique_doc_ids": len(referenced_ids - matched_ids),
        },
        "unresolved": {
            "counts": {
                f"{role}:{reason}": count
                for (role, reason), count in sorted(unresolved_counts.items())
            },
            "samples": unresolved_samples,
        },
        "full_document_traceability_verified": unresolved == 0,
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-archive", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--expected-pairs-sha256")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    result = audit(
        args.trajectory_archive,
        args.pairs,
        expected_rows=args.expected_rows,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_pairs_sha256=args.expected_pairs_sha256,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_all and not result["full_document_traceability_verified"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
