#!/usr/bin/env python3
"""Build controlled LRAT negative-filtering arms from trajectory provenance.

The method arm removes negatives that are visited later in the same trajectory.
The heuristic arm removes repeatedly surfaced negatives.  The random arm drops
the exact same number of negatives per row as the method arm.  All arms retain
at least ``minimum_negatives`` and change only aligned ``neg``/``neg_id`` lists.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable

from build_trajectory_provenance import browse_document_id, parse_search


BROWSE_TOOLS = {"get_document", "visit"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_features(steps: list[dict[str, Any]]) -> list[dict[str, set[str]]]:
    """Return candidate-risk sets for each browse event in source event order."""
    browse_docs = [
        browse_document_id(step)
        for step in steps
        if step.get("type") == "tool_call" and step.get("tool_name") in BROWSE_TOOLS
    ]
    browse_docs = [doc for doc in browse_docs if doc]
    exposure_counts: collections.Counter[str] = collections.Counter()
    features: list[dict[str, set[str]]] = []
    browse_index = 0
    for step in steps:
        if step.get("type") == "tool_call" and step.get("tool_name") == "search":
            _, documents = parse_search(step)
            exposure_counts.update(str(doc) for doc in documents)
            continue
        if not (step.get("type") == "tool_call" and step.get("tool_name") in BROWSE_TOOLS):
            continue
        later = set(browse_docs[browse_index + 1 :])
        repeated = {doc for doc, count in exposure_counts.items() if count >= 2}
        features.append({"later_visited": later, "repeated": repeated})
        browse_index += 1
    return features


def load_feature_map(archive: Path) -> dict[tuple[str, int], dict[str, set[str]]]:
    result: dict[tuple[str, int], dict[str, set[str]]] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            extracted = bundle.extractfile(member)
            if extracted is None:
                continue
            value = json.load(extracted)
            raw_steps = value.get("result") if isinstance(value, dict) else []
            steps = [step if isinstance(step, dict) else {} for step in raw_steps or []]
            for event_idx, features in enumerate(event_features(steps)):
                result[(member.name, event_idx)] = features
    return result


def deterministic_random_indices(length: int, count: int, *, seed: int, source_line: int) -> set[int]:
    digest = hashlib.sha256(f"{seed}\0{source_line}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return set(rng.sample(range(length), count)) if count else set()


def filtered_row(row: dict[str, Any], removed_indices: set[int]) -> dict[str, Any]:
    output = dict(row)
    output["neg"] = [value for index, value in enumerate(row["neg"]) if index not in removed_indices]
    output["neg_id"] = [value for index, value in enumerate(row["neg_id"]) if index not in removed_indices]
    return output


def choose_risk_indices(row: dict[str, Any], risk_ids: set[str], minimum_negatives: int) -> tuple[set[int], bool]:
    candidates = [index for index, doc_id in enumerate(row["neg_id"]) if str(doc_id) in risk_ids]
    capacity = max(0, len(row["neg_id"]) - minimum_negatives)
    return set(candidates[:capacity]), len(candidates) > capacity


def transform_row(
    row: dict[str, Any],
    provenance: dict[str, Any],
    features: dict[str, set[str]] | None,
    *,
    minimum_negatives: int,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    negatives = row.get("neg")
    negative_ids = row.get("neg_id")
    if not isinstance(negatives, list) or not isinstance(negative_ids, list) or len(negatives) != len(negative_ids):
        raise ValueError(f"line {provenance.get('source_line')}: neg/neg_id are not aligned lists")
    if len(negative_ids) < minimum_negatives:
        raise ValueError(f"line {provenance.get('source_line')}: fewer than {minimum_negatives} source negatives")
    if provenance.get("bucket") != "stable" or features is None:
        empty: set[int] = set()
        return {
            "random": filtered_row(row, empty),
            "exposure": filtered_row(row, empty),
            "later_visit": filtered_row(row, empty),
        }, {"stable": False, "later_candidates": 0, "exposure_candidates": 0, "later_removed": 0, "exposure_removed": 0, "capped": False}

    later_indices, later_capped = choose_risk_indices(row, features["later_visited"], minimum_negatives)
    exposure_risk_indices, exposure_capped = choose_risk_indices(row, features["repeated"], minimum_negatives)
    target = len(later_indices)
    exposure_indices = set(sorted(exposure_risk_indices)[:target])
    if len(exposure_indices) < target:
        remaining = [index for index in range(len(negative_ids)) if index not in exposure_indices]
        fill_count = target - len(exposure_indices)
        fill_seed = int.from_bytes(
            hashlib.sha256(f"exposure-fill\0{seed}\0{provenance['source_line']}".encode()).digest()[:8], "big"
        )
        exposure_indices.update(random.Random(fill_seed).sample(remaining, fill_count))
    random_indices = deterministic_random_indices(
        len(negative_ids), len(later_indices), seed=seed, source_line=int(provenance["source_line"])
    )
    return {
        "random": filtered_row(row, random_indices),
        "exposure": filtered_row(row, exposure_indices),
        "later_visit": filtered_row(row, later_indices),
    }, {
        "stable": True,
        "later_candidates": sum(str(doc_id) in features["later_visited"] for doc_id in negative_ids),
        "exposure_candidates": sum(str(doc_id) in features["repeated"] for doc_id in negative_ids),
        "later_removed": len(later_indices),
        "exposure_removed": len(exposure_indices),
        "capped": later_capped or exposure_capped,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    feature_map = load_feature_map(args.trajectory_archive)
    excluded_query_hashes: set[str] = set()
    split_manifest: dict[str, Any] | None = None
    if args.split_manifest is not None:
        split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        for split_name in ("dev", "test"):
            excluded_query_hashes.update(
                str(item["normalized_query_sha256"])
                for item in split_manifest[split_name].get("provenance", [])
            )
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_root.name}.staging.", dir=args.output_root.parent))
    counters: dict[str, collections.Counter[str]] = {
        name: collections.Counter() for name in ("random", "exposure", "later_visit")
    }
    global_counts = collections.Counter()
    try:
        handles = {name: (staging / f"{name}.jsonl").open("x", encoding="utf-8") for name in counters}
        try:
            with args.pairs.open(encoding="utf-8") as pair_handle, args.provenance.open(encoding="utf-8") as prov_handle:
                for source_line, (pair_line, prov_line) in enumerate(zip(pair_handle, prov_handle, strict=True), 1):
                    row = json.loads(pair_line)
                    provenance = json.loads(prov_line)
                    if int(provenance.get("source_line", -1)) != source_line:
                        raise ValueError(f"provenance/source line mismatch at {source_line}")
                    if str(provenance.get("normalized_query_sha256")) in excluded_query_hashes:
                        global_counts["heldout_source_rows_excluded"] += 1
                        continue
                    feature = None
                    if provenance.get("bucket") == "stable":
                        event = provenance.get("event") or {}
                        feature = feature_map.get((str(event.get("traj_path")), int(event.get("event_idx", -1))))
                        if feature is None:
                            raise ValueError(f"missing trajectory event features at source line {source_line}")
                    arms, details = transform_row(
                        row, provenance, feature,
                        minimum_negatives=args.minimum_negatives,
                        seed=args.seed,
                    )
                    global_counts["rows"] += 1
                    global_counts["stable_rows"] += int(details["stable"])
                    global_counts["later_risk_occurrences"] += details["later_candidates"]
                    global_counts["exposure_risk_occurrences"] += details["exposure_candidates"]
                    global_counts["capped_rows"] += int(details["capped"])
                    for name, output_row in arms.items():
                        handles[name].write(json.dumps(output_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        removed = len(row["neg_id"]) - len(output_row["neg_id"])
                        counters[name]["rows"] += 1
                        counters[name]["changed_rows"] += int(removed > 0)
                        counters[name]["negatives_removed"] += removed
                        counters[name]["rows_below_minimum"] += int(len(output_row["neg_id"]) < args.minimum_negatives)
        finally:
            for handle in handles.values():
                handle.close()

        outputs = {
            name: {
                "path": f"{name}.jsonl",
                "bytes": (staging / f"{name}.jsonl").stat().st_size,
                "sha256": file_sha256(staging / f"{name}.jsonl"),
                **dict(counters[name]),
            }
            for name in counters
        }
        if not (
            outputs["random"]["negatives_removed"]
            == outputs["exposure"]["negatives_removed"]
            == outputs["later_visit"]["negatives_removed"]
        ):
            raise AssertionError("control arms are not removal-count matched")
        if any(value["rows_below_minimum"] for value in outputs.values()):
            raise AssertionError("an arm fell below the minimum negative count")
        if split_manifest is not None:
            expected_train_rows = int(split_manifest["train"]["rows"])
            if any(value["rows"] != expected_train_rows for value in outputs.values()):
                raise AssertionError("arm row count does not match the split-manifest training count")
        manifest = {
            "schema_version": "lrat_reliability_arms.v1",
            "contract": {
                "random": "drop the same number per row as later_visit with deterministic random selection",
                "exposure": "prioritize negatives surfaced at least twice before the mapped positive browse event, then deterministically fill if needed to match later_visit removals per row",
                "later_visit": "drop negatives visited after the mapped positive browse event",
                "minimum_negatives": args.minimum_negatives,
                "only_changed_fields": ["neg", "neg_id"],
                "unstable_provenance_rows": "unchanged",
                "locked_test_used": False,
            },
            "seed": args.seed,
            "inputs": {
                "pairs": {"path": str(args.pairs), "bytes": args.pairs.stat().st_size, "sha256": file_sha256(args.pairs)},
                "provenance": {"path": str(args.provenance), "bytes": args.provenance.stat().st_size, "sha256": file_sha256(args.provenance)},
                "trajectory_archive": {"path": str(args.trajectory_archive), "bytes": args.trajectory_archive.stat().st_size, "sha256": file_sha256(args.trajectory_archive)},
                "split_manifest": (
                    {"path": str(args.split_manifest), "bytes": args.split_manifest.stat().st_size, "sha256": file_sha256(args.split_manifest)}
                    if args.split_manifest is not None
                    else None
                ),
            },
            "global": dict(global_counts),
            "outputs": outputs,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, args.output_root)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-archive", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split-manifest", type=Path, help="exclude the manifest's dev/test query groups from arm outputs")
    parser.add_argument("--minimum-negatives", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
