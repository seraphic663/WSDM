#!/usr/bin/env python3
"""Audit paper-style LRAT flywheel inputs and one completed iteration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from solution.src.build_paper_flywheel_pairs import collect_browse_candidates
except ModuleNotFoundError:
    from build_paper_flywheel_pairs import collect_browse_candidates


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def load_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("research_only") is not True:
        raise ValueError("paper flywheel config must be research_only")
    if config.get("competition_submission_eligible") is not False:
        raise ValueError("paper flywheel must be competition-ineligible")
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("config lacks paths")
    research_root = (repo_root / paths["research_root"]).resolve()
    if not _inside(research_root, (repo_root / "ccir/research").resolve()):
        raise ValueError("research root must stay under ccir/research")
    return config


def read_queries(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if len(row) < 2:
                raise ValueError(f"malformed query TSV row {row_number}")
            query_id, question = row[0].strip(), row[1].strip()
            if not query_id or not question:
                raise ValueError(f"empty query TSV field at row {row_number}")
            if query_id in result:
                raise ValueError(f"duplicate query ID: {query_id}")
            result[query_id] = question
    return result


def audit_preflight(config_path: Path, repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = load_config(config_path, repo_root)
    paths = config["paths"]
    resolved = {name: (repo_root / value).resolve() for name, value in paths.items()}
    required_code = [
        repo_root / "search_agent/tongyi_client.py",
        repo_root / "src/data_builder.py",
        repo_root / "FlagEmbedding",
        repo_root / "solution/experiments/train.sh",
    ]
    presence = {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in resolved.items()
    }
    presence["required_code"] = {
        str(path.relative_to(repo_root)): path.exists() for path in required_code
    }
    query_count = None
    if resolved["seed_pool"].is_file():
        query_count = len(read_queries(resolved["seed_pool"]))
    expected_count = int(config["paper_contract"]["seed_queries"])
    blockers = []
    for name in ("trajectory_archive", "corpus", "initial_retriever"):
        if not resolved[name].exists():
            blockers.append(f"missing {name}: {resolved[name]}")
    for path in required_code:
        if not path.exists():
            blockers.append(f"missing required code: {path}")
    if query_count is None or query_count < expected_count:
        blockers.append(
            f"seed query pool count is {query_count}, need at least {expected_count}"
        )
        if not resolved["source_qa"].is_file():
            blockers.append(
                f"missing full InfoSeekQA research source: {resolved['source_qa']}"
            )
    seed_manifest = resolved["seed_manifest"]
    if not seed_manifest.is_file():
        blockers.append(f"missing seed pool manifest: {seed_manifest}")
    else:
        seed_value = json.loads(seed_manifest.read_text(encoding="utf-8"))
        if seed_value.get("external_data_used") is not True:
            blockers.append("seed pool is not marked as external research data")
        if seed_value.get("competition_submission_eligible") is not False:
            blockers.append("seed pool is not marked competition-ineligible")
    runtime = config["runtime"]
    model_availability = {}
    for key in ("agent_model", "judge_model"):
        value = str(runtime[key])
        local = Path(value)
        available = local.exists() if local.is_absolute() else False
        model_availability[key] = {
            "configured": value,
            "local_path_available": available,
            "hub_download_would_be_required": not available,
        }
        if not available:
            blockers.append(f"{key} is not locally available: {value}")
    profile_name = runtime["training_profile"]
    profile = runtime["training_profiles"][profile_name]
    paper_contract = config["paper_contract"]
    training_contract = {
        "profile": profile_name,
        "world_size": profile["world_size"],
        "per_device_train_batch_size": profile["per_device_train_batch_size"],
        "gradient_accumulation_steps": profile["gradient_accumulation_steps"],
        "global_microbatch_queries": profile["global_microbatch_queries"],
        "exact_in_batch_negative_pool": profile["exact_in_batch_negative_pool"],
        "train_group_size": paper_contract["train_group_size"],
        "query_max_tokens": paper_contract["query_max_tokens"],
        "passage_max_tokens": paper_contract["passage_max_tokens"],
        "epochs": paper_contract["train_epochs_per_loop"],
        "temperature": paper_contract["temperature"],
        "weighted_loss_reduction": paper_contract[
            "weighted_loss_reduction"
        ],
    }
    return {
        "mode": "preflight",
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "research_only": True,
        "competition_submission_eligible": False,
        "presence": presence,
        "query_count": query_count,
        "expected_query_count": expected_count,
        "model_availability": model_availability,
        "training_contract": training_contract,
        "disclosed_unknowns": config.get("disclosed_unknowns", []),
        "ready_for_full_collection": not blockers,
        "blockers": blockers,
    }


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield line_number, value


def audit_iteration(
    config_path: Path,
    repo_root: Path,
    *,
    loop_dir: Path,
    current_model: Path,
    require_complete_queries: bool = True,
    require_completed_training: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = load_config(config_path, repo_root)
    research_root = (repo_root / config["paths"]["research_root"]).resolve()
    loop_dir = loop_dir.resolve()
    if not _inside(loop_dir, research_root):
        raise ValueError("loop directory is outside research root")
    if not current_model.is_dir() or not (current_model / "model.safetensors").is_file():
        raise ValueError(f"invalid current retriever: {current_model}")

    query_path = loop_dir / "queries.tsv"
    query_manifest_path = loop_dir / "queries.manifest.json"
    if not query_manifest_path.is_file():
        raise ValueError("iteration lacks loop query manifest")
    query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
    queries = read_queries(query_path)
    if query_manifest.get("output_sha256") != sha256_file(query_path):
        raise ValueError("loop query manifest SHA mismatch")
    if query_manifest.get("sample_count") != len(queries):
        raise ValueError("loop query manifest count mismatch")
    trajectory_dir = loop_dir / "trajectories"
    pairs_path = loop_dir / "training_pairs.jsonl"
    pairs_summary_path = loop_dir / "training_pairs.summary.json"
    index_manifest = loop_dir / "index" / "manifest.json"
    collection_manifest = trajectory_dir / "COLLECTION_MANIFEST.json"
    if (
        not trajectory_dir.is_dir()
        or not (trajectory_dir / "COMPLETED").is_file()
        or not collection_manifest.is_file()
        or not pairs_path.is_file()
        or not pairs_summary_path.is_file()
        or not (loop_dir / "PAIRS_COMPLETED").is_file()
        or not index_manifest.is_file()
    ):
        raise ValueError("iteration lacks trajectories, pairs, or index manifest")
    collection_value = json.loads(
        collection_manifest.read_text(encoding="utf-8")
    )
    if collection_value.get("complete") is not True:
        raise ValueError("trajectory collection manifest is incomplete")
    pairs_summary = json.loads(
        pairs_summary_path.read_text(encoding="utf-8")
    )
    pair_contract = pairs_summary.get("paper_contract", {})
    if not all(
        pair_contract.get(key) is True
        for key in (
            "irrelevant_browses_removed_from_positives",
            "negatives_only_from_corresponding_search_candidate_set",
            "immediate_post_browse_reasoning_used",
            "global_eq3_weights",
        )
    ):
        raise ValueError("pair builder summary does not satisfy paper contract")

    trajectory_ids: set[str] = set()
    trajectory_status = Counter()
    search_steps = browse_steps = search_to_browse = 0
    pair_origins: dict[tuple[str, str, str], list[tuple[str, ...]]] = {}
    for path in sorted(trajectory_dir.glob("run_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        query_id = str(value.get("query_id"))
        if query_id not in queries:
            raise ValueError(f"unknown trajectory query ID: {query_id}")
        if query_id in trajectory_ids:
            raise ValueError(f"duplicate trajectory query ID: {query_id}")
        trajectory_ids.add(query_id)
        trajectory_status[str(value.get("status", "unknown"))] += 1
        steps = value.get("result")
        if not isinstance(steps, list):
            raise ValueError(f"trajectory lacks result list: {path}")
        previous_search = False
        for step in steps:
            is_search = (
                isinstance(step, dict)
                and step.get("type") == "tool_call"
                and step.get("tool_name") == "search"
            )
            is_browse = (
                isinstance(step, dict)
                and step.get("type") == "tool_call"
                and step.get("tool_name") in {"get_document", "visit"}
            )
            search_steps += int(is_search)
            browse_steps += int(is_browse)
            search_to_browse += int(previous_search and is_browse)
            if is_search:
                previous_search = True
            elif isinstance(step, dict) and step.get("type") != "reasoning":
                previous_search = False
        for candidate in collect_browse_candidates(
            value, trajectory_path=path.name
        ):
            key = (
                candidate.trajectory_path,
                candidate.query,
                candidate.positive_id,
            )
            expected_negatives = tuple(
                doc_id
                for doc_id in candidate.candidate_ids
                if doc_id != candidate.positive_id
            )
            pair_origins.setdefault(key, []).append(expected_negatives)
    if require_complete_queries and trajectory_ids != set(queries):
        missing = sorted(set(queries) - trajectory_ids)[:20]
        raise ValueError(
            f"trajectory query coverage incomplete: {len(trajectory_ids)}/"
            f"{len(queries)}, first missing={missing}"
        )

    pairs = []
    for line_number, row in iter_jsonl(pairs_path):
        required = {
            "query",
            "pos",
            "neg",
            "pos_id",
            "neg_id",
            "reasoning_len",
            "satisfied",
            "reweight_rate",
            "source_trajectory",
        }
        if not required.issubset(row):
            raise ValueError(f"training pair {line_number} lacks required fields")
        if not row["pos"] or not row["neg"]:
            raise ValueError(f"training pair {line_number} has empty pos/neg")
        if len(row["pos"]) != len(row["pos_id"]) or len(row["neg"]) != len(row["neg_id"]):
            raise ValueError(f"training pair {line_number} has ID/text misalignment")
        reasoning_len = row["reasoning_len"]
        weight = float(row["reweight_rate"])
        if not isinstance(reasoning_len, (int, float)) or reasoning_len <= 0:
            raise ValueError(f"training pair {line_number} has invalid reasoning_len")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"training pair {line_number} has invalid weight")
        if row["satisfied"] is not True:
            raise ValueError(
                f"training pair {line_number} retained an irrelevant positive"
            )
        if len(row["pos_id"]) != 1:
            raise ValueError(
                f"training pair {line_number} must have one browsed positive"
            )
        origin_key = (
            str(row["source_trajectory"]),
            str(row["query"]),
            str(row["pos_id"][0]),
        )
        possible_negatives = pair_origins.get(origin_key, [])
        actual_negatives = tuple(map(str, row["neg_id"]))
        if actual_negatives not in possible_negatives:
            raise ValueError(
                f"training pair {line_number} is not a paper Search->Browse "
                "sample with search-local negatives"
            )
        pairs.append(row)
    if not pairs:
        raise ValueError("training pair file is empty")

    lengths = [float(row["reasoning_len"]) for row in pairs]
    beta = statistics.median(lengths)
    raw = [1.0 - math.exp(-math.log(2.0) * value / beta) for value in lengths]
    mean_raw = statistics.fmean(raw)
    expected_weights = [value / mean_raw for value in raw]
    max_error = max(
        abs(float(row["reweight_rate"]) - expected)
        for row, expected in zip(pairs, expected_weights)
    )
    if max_error > 1e-9:
        raise ValueError(f"paper weight formula mismatch: max error {max_error}")

    index_value = json.loads(index_manifest.read_text(encoding="utf-8"))
    expected_model_sha = sha256_file(current_model / "model.safetensors")
    if index_value.get("complete") is not True:
        raise ValueError("index manifest is incomplete")
    if index_value.get("retriever_model_sha256") != expected_model_sha:
        raise ValueError("index was not built from the declared current retriever")
    shards = index_value.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("index manifest lacks shards")
    for shard in shards:
        shard_path = Path(shard["path"])
        if (
            not shard_path.is_file()
            or shard_path.stat().st_size != shard["bytes"]
            or sha256_file(shard_path) != shard["sha256"]
        ):
            raise ValueError(f"index shard identity mismatch: {shard_path}")

    training_audit = None
    next_model = loop_dir / "next_retriever"
    if require_completed_training or next_model.exists():
        completed = next_model / "COMPLETED"
        contract_path = next_model / "TRAINING_CONTRACT.json"
        next_weight = next_model / "model.safetensors"
        if not completed.is_file() or (next_model / "RUNNING").exists():
            raise ValueError("next retriever is not cleanly completed")
        if not contract_path.is_file() or not next_weight.is_file():
            raise ValueError("next retriever lacks contract or final weight")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if contract.get("research_only") is not True:
            raise ValueError("training contract is not research-only")
        if contract.get("competition_submission_eligible") is not False:
            raise ValueError("training contract is not competition-ineligible")
        if contract.get("runtime_weight_reduction") != "paper_mean":
            raise ValueError("training did not use the paper mean reduction")
        if contract.get("current_retriever_sha256") != expected_model_sha:
            raise ValueError("training did not start from the declared retriever")
        if contract.get("training_pairs_sha256") != sha256_file(pairs_path):
            raise ValueError("training pair SHA does not match training contract")
        paper = contract.get("paper_contract", {})
        expected_paper = config["paper_contract"]
        for key in (
            "train_epochs_per_loop",
            "learning_rate",
            "temperature",
            "weighted_loss_reduction",
            "reported_train_batch_size",
            "train_group_size",
            "query_max_tokens",
            "passage_max_tokens",
        ):
            if paper.get(key) != expected_paper[key]:
                raise ValueError(f"training contract mismatch for {key}")
        training_audit = {
            "completed": True,
            "profile": contract["profile"],
            "profile_contract": contract["profile_contract"],
            "next_retriever": str(next_model),
            "next_retriever_sha256": sha256_file(next_weight),
            "training_contract_sha256": sha256_file(contract_path),
        }

    return {
        "mode": "iteration",
        "loop_dir": str(loop_dir),
        "research_only": True,
        "competition_submission_eligible": False,
        "current_model": str(current_model.resolve()),
        "current_model_sha256": expected_model_sha,
        "query_count": len(queries),
        "trajectory_count": len(trajectory_ids),
        "trajectory_status": dict(trajectory_status),
        "search_steps": search_steps,
        "browse_steps": browse_steps,
        "search_to_browse_transitions": search_to_browse,
        "training_pair_count": len(pairs),
        "global_median_reasoning_len": beta,
        "mean_raw_weight": mean_raw,
        "maximum_weight_formula_error": max_error,
        "pairs_sha256": sha256_file(pairs_path),
        "training": training_audit,
        "paper_contract_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--loop-dir", type=Path)
    parser.add_argument("--current-model", type=Path)
    parser.add_argument("--allow-partial-query-coverage", action="store_true")
    parser.add_argument("--require-completed-training", action="store_true")
    args = parser.parse_args()
    if args.loop_dir is None:
        result = audit_preflight(args.config, args.repo_root)
    else:
        if args.current_model is None:
            parser.error("--current-model is required with --loop-dir")
        result = audit_iteration(
            args.config,
            args.repo_root,
            loop_dir=args.loop_dir,
            current_model=args.current_model,
            require_complete_queries=not args.allow_partial_query_coverage,
            require_completed_training=args.require_completed_training,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
