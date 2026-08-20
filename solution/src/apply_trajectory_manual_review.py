#!/usr/bin/env python3
"""Validate manual trajectory-quality labels and summarize each stratum."""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from solution.src.build_trajectory_provenance import file_sha256


ALLOWED_LABELS = {"keep", "bad_positive", "uncertain"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def apply_review(
    samples_path: Path, review_path: Path, output_root: Path
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    samples = load_jsonl(samples_path)
    decisions = load_jsonl(review_path)
    by_id = {}
    for decision in decisions:
        sample_id = decision.get("sample_id")
        label = decision.get("manual_label")
        reason = decision.get("manual_reason")
        if not isinstance(sample_id, str) or sample_id in by_id:
            raise ValueError("review sample ids must be unique strings")
        if label not in ALLOWED_LABELS:
            raise ValueError(f"{sample_id}: invalid manual label")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{sample_id}: manual reason is required")
        by_id[sample_id] = decision
    expected = {sample["sample_id"] for sample in samples}
    if set(by_id) != expected:
        raise ValueError(
            f"review coverage mismatch: missing={sorted(expected-set(by_id))[:5]} "
            f"extra={sorted(set(by_id)-expected)[:5]}"
        )

    reviewed = []
    counts: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for sample in samples:
        decision = by_id[sample["sample_id"]]
        sample["manual_label"] = decision["manual_label"]
        sample["manual_reason"] = decision["manual_reason"].strip()
        reviewed.append(sample)
        counts[sample["stratum"]][sample["manual_label"]] += 1
    per_stratum = {}
    for stratum, counter in sorted(counts.items()):
        decided = counter["keep"] + counter["bad_positive"]
        per_stratum[stratum] = {
            **dict(counter),
            "decided_rows": decided,
            "bad_positive_precision": (
                counter["bad_positive"] / decided if decided else None
            ),
        }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging.", dir=output_root.parent)
    )
    try:
        reviewed_path = staging / "reviewed_samples.jsonl"
        with reviewed_path.open("x", encoding="utf-8") as handle:
            for sample in reviewed:
                handle.write(
                    json.dumps(sample, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
        report = {
            "created_at": datetime.now().astimezone().isoformat(),
            "inputs": {
                "samples": {
                    "path": str(samples_path.resolve()),
                    "sha256": file_sha256(samples_path),
                },
                "manual_review": {
                    "path": str(review_path.resolve()),
                    "sha256": file_sha256(review_path),
                },
            },
            "labels": sorted(ALLOWED_LABELS),
            "sample_rows": len(reviewed),
            "overall": dict(
                collections.Counter(sample["manual_label"] for sample in reviewed)
            ),
            "per_stratum": per_stratum,
            "locked_test_used": False,
        }
        report_path = staging / "review_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "created_at": datetime.now().astimezone().isoformat(),
            "review_complete": True,
            "sample_rows": len(reviewed),
            "outputs": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in (reviewed_path, report_path)
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output_root)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--manual-review", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = apply_review(args.samples, args.manual_review, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
