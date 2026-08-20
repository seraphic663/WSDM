#!/usr/bin/env python3
"""Build one competition-compliant offline flywheel training shard.

The current retriever refreshes supervision only by selecting its hardest
negatives from each official LRAT pair. No text, label, or trajectory is
generated. A uniform control shard and model-mined candidate shard are emitted
from the same deterministic normalized-query bucket.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_query(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def query_hash(query: str) -> str:
    return hashlib.sha256(normalize_query(query).encode("utf-8")).hexdigest()


def query_bucket(query: str, *, salt: str, modulus: int) -> int:
    if modulus <= 1:
        raise ValueError("modulus must be greater than one")
    payload = f"{salt}\0{normalize_query(query)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulus


def validate_pair(row: dict[str, Any], line_number: int) -> None:
    query = row.get("query")
    positives = row.get("pos")
    positive_ids = row.get("pos_id")
    negatives = row.get("neg")
    negative_ids = row.get("neg_id")
    if not isinstance(query, str):
        raise ValueError(f"line {line_number}: query must be a string")
    for name, value in (
        ("pos", positives),
        ("pos_id", positive_ids),
        ("neg", negatives),
        ("neg_id", negative_ids),
    ):
        if not isinstance(value, list) or not value:
            raise ValueError(f"line {line_number}: {name} must be a non-empty list")
    if len(positives) != len(positive_ids):
        raise ValueError(f"line {line_number}: pos/pos_id length mismatch")
    if len(negatives) != len(negative_ids):
        raise ValueError(f"line {line_number}: neg/neg_id length mismatch")
    if len(negatives) < 5:
        raise ValueError(f"line {line_number}: at least five negatives are required")
    if not all(isinstance(value, str) for value in positives + negatives):
        raise ValueError(f"line {line_number}: passage values must be strings")
    weight = row.get("reweight_rate", 1.0)
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"line {line_number}: invalid reweight_rate")


def negative_pool_indices(
    row: dict[str, Any],
    *,
    pool_size: int,
    normalized_query_hash: str,
) -> list[int]:
    if pool_size < 5:
        raise ValueError("pool_size must be at least five")
    negative_ids = row["neg_id"]
    if len(negative_ids) <= pool_size:
        return list(range(len(negative_ids)))
    ranked = sorted(
        range(len(negative_ids)),
        key=lambda index: hashlib.sha256(
            f"{normalized_query_hash}\0{negative_ids[index]}\0{index}".encode("utf-8")
        ).digest(),
    )
    return ranked[:pool_size]


def candidate_row(row: dict[str, Any], selected_negative_indices: list[int]) -> dict[str, Any]:
    if len(set(selected_negative_indices)) != len(selected_negative_indices):
        raise ValueError("selected negative indexes are duplicated")
    result = dict(row)
    result["neg"] = [row["neg"][index] for index in selected_negative_indices]
    result["neg_id"] = [row["neg_id"][index] for index in selected_negative_indices]
    return result


def _atomic_json(path: Path, value: Any) -> None:
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_shard(
    source: Path,
    *,
    salt: str,
    modulus: int,
    bucket: int,
    pool_size: int,
) -> list[dict[str, Any]]:
    if not 0 <= bucket < modulus:
        raise ValueError("bucket must be within modulus")
    selected = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected object")
            validate_pair(row, line_number)
            if query_bucket(row["query"], salt=salt, modulus=modulus) != bucket:
                continue
            qhash = query_hash(row["query"])
            selected.append(
                {
                    "source_line": line_number,
                    "query_hash": qhash,
                    "row": row,
                    "pool_indices": negative_pool_indices(
                        row,
                        pool_size=pool_size,
                        normalized_query_hash=qhash,
                    ),
                }
            )
    if not selected:
        raise ValueError("selected shard is empty")
    return selected


def mine(
    *,
    source: Path,
    model_path: Path,
    output_dir: Path,
    salt: str,
    modulus: int,
    bucket: int,
    pool_size: int,
    hard_k: int,
    batch_size: int,
    query_max_length: int,
    passage_max_length: int,
    device_name: str,
    instruction: str,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if hard_k < 5 or hard_k > pool_size:
        raise ValueError("hard_k must be between five and pool_size")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    try:
        from solution.src.evaluate_qwen3_pairs import encode
    except ModuleNotFoundError:
        from evaluate_qwen3_pairs import encode
    import torch
    from transformers import AutoModel, AutoTokenizer

    started = time.perf_counter()
    selected = load_shard(
        source,
        salt=salt,
        modulus=modulus,
        bucket=bucket,
        pool_size=pool_size,
    )
    control_rows = [item["row"] for item in selected]
    query_texts = [
        f"Instruct: {instruction}\nQuery:{item['row']['query']}" for item in selected
    ]
    passage_texts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    for item in selected:
        row = item["row"]
        start = len(passage_texts)
        passage_texts.extend(row["pos"])
        passage_texts.extend(row["neg"][index] for index in item["pool_indices"])
        spans.append((start, len(passage_texts), len(row["pos"])))

    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
    loaded_at = time.perf_counter()
    query_embeddings = encode(
        query_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=query_max_length,
    )
    passage_embeddings = encode(
        passage_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=passage_max_length,
    )
    encoded_at = time.perf_counter()

    candidate_rows = []
    metadata = []
    rank_counts: Counter[str] = Counter()
    margins = []
    for item, (start, end, positive_count), query_embedding in zip(
        selected, spans, query_embeddings
    ):
        scores = query_embedding @ passage_embeddings[start:end].T
        positive_scores = scores[:positive_count]
        negative_scores = scores[positive_count:]
        best_positive = float(positive_scores.max().item())
        ranked_negative_offsets = sorted(
            range(len(item["pool_indices"])),
            key=lambda index: (-float(negative_scores[index].item()), index),
        )
        selected_offsets = ranked_negative_offsets[:hard_k]
        selected_source_indices = [item["pool_indices"][offset] for offset in selected_offsets]
        best_negative = float(negative_scores[ranked_negative_offsets[0]].item())
        positive_rank = 1 + sum(
            float(score.item()) > best_positive for score in negative_scores
        )
        margin = best_positive - best_negative
        margins.append(margin)
        if positive_rank == 1:
            rank_counts["rank1"] += 1
        elif positive_rank <= 5:
            rank_counts["rank2_5"] += 1
        else:
            rank_counts["rank6plus"] += 1
        candidate_rows.append(candidate_row(item["row"], selected_source_indices))
        metadata.append(
            {
                "source_line": item["source_line"],
                "query_sha256": item["query_hash"],
                "original_negative_count": len(item["row"]["neg"]),
                "candidate_pool_count": len(item["pool_indices"]),
                "selected_negative_count": len(selected_source_indices),
                "positive_rank_in_pool": positive_rank,
                "best_positive_score": best_positive,
                "best_negative_score": best_negative,
                "positive_negative_margin": margin,
                "candidate_pool_source_indices": item["pool_indices"],
                "selected_negative_source_indices": selected_source_indices,
                "selected_negative_ids": [
                    item["row"]["neg_id"][index] for index in selected_source_indices
                ],
                "selected_negative_scores": [
                    float(negative_scores[offset].item()) for offset in selected_offsets
                ],
            }
        )

    control_path = output_dir / "control.jsonl"
    candidate_path = output_dir / "candidate.jsonl"
    metadata_path = output_dir / "mining.jsonl"
    _write_jsonl(control_path, control_rows)
    _write_jsonl(candidate_path, candidate_rows)
    _write_jsonl(metadata_path, metadata)
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "method": "offline retriever-in-the-loop hard-negative refresh over official pairs",
        "hypothesis": "the current retriever can improve its next update by concentrating official negative sampling on its own highest-scoring mistakes",
        "inputs": {
            "source": {"path": str(source.resolve()), "sha256": sha256_file(source)},
            "model": {
                "path": str(model_path.resolve()),
                "model_safetensors_sha256": sha256_file(model_path / "model.safetensors"),
            },
        },
        "shard": {
            "salt": salt,
            "modulus": modulus,
            "bucket": bucket,
            "rows": len(selected),
            "unique_normalized_queries": len({item["query_hash"] for item in selected}),
        },
        "mining": {
            "candidate_pool_size_cap": pool_size,
            "hard_negative_count": hard_k,
            "batch_size": batch_size,
            "query_max_length": query_max_length,
            "passage_max_length": passage_max_length,
            "instruction": instruction,
            "dtype": str(dtype),
            "device": str(device),
            "model_load_seconds": loaded_at - started,
            "encoding_seconds": encoded_at - loaded_at,
            "total_seconds": time.perf_counter() - started,
            "passages_encoded": len(passage_texts),
            "positive_rank_buckets": dict(rank_counts),
            "margin": {
                "min": min(margins),
                "median": median(margins),
                "max": max(margins),
                "mean": sum(margins) / len(margins),
            },
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        },
        "outputs": {
            "control": {"path": str(control_path), "sha256": sha256_file(control_path)},
            "candidate": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "metadata": {"path": str(metadata_path), "sha256": sha256_file(metadata_path)},
        },
        "contract": {
            "external_data_used": False,
            "external_model_used": False,
            "generated_text_used": False,
            "locked_test_used": False,
            "all_training_text_and_ids_trace_to_source_pairs": True,
            "control_rows_identical_to_source": True,
            "candidate_changes_only_neg_and_neg_id": True,
            "model_inference_used_only_to_select_source_negatives": True,
        },
    }
    _atomic_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--salt", default="offline-flywheel-v1")
    parser.add_argument("--modulus", type=int, default=16)
    parser.add_argument("--bucket", type=int, required=True)
    parser.add_argument("--pool-size", type=int, default=32)
    parser.add_argument("--hard-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-max-length", type=int, default=128)
    parser.add_argument("--passage-max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    args = parser.parse_args()
    result = mine(
        source=args.source,
        model_path=args.model,
        output_dir=args.output_dir,
        salt=args.salt,
        modulus=args.modulus,
        bucket=args.bucket,
        pool_size=args.pool_size,
        hard_k=args.hard_k,
        batch_size=args.batch_size,
        query_max_length=args.query_max_length,
        passage_max_length=args.passage_max_length,
        device_name=args.device,
        instruction=args.instruction,
    )
    print(
        json.dumps(
            {
                "rows": result["shard"]["rows"],
                "queries": result["shard"]["unique_normalized_queries"],
                "rank_buckets": result["mining"]["positive_rank_buckets"],
                "outputs": result["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
