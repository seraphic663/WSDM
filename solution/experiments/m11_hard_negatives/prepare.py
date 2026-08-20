#!/usr/bin/env python3
"""M11: In-Training-Set Hard Negative Mining.

Model: M11
Source: M00 (Qwen3-Embedding-0.6B, SHA 0437e45c...e23fd)
Data: early_stop_v1 train 94,113 rows, augmented with hard negatives
Method: Use M01 to rank all candidate passages for each query.
        Top-ranked passages that are currently labeled as negatives
        are "hard negatives" — added to the training data's negative set.

This prepares augmented training data. No external data is used.
"""

import json
from pathlib import Path
from collections import defaultdict
from common import (
    TRAIN_94K, M01_PATH, DEV_1500,
    load_jsonl, save_jsonl, sha256_file,
)

OUTPUT_DIR = Path("/root/data/LRAT/ccir/data/experiments/m11_hard_negatives")
OUTPUT_TRAIN = OUTPUT_DIR / "train_m11.jsonl"
OUTPUT_CONFIG = OUTPUT_DIR / "m11_config.json"

# How many hard negatives to add per query
MAX_HARD_NEGS = 3
# Use top-K from M01 to find hard negatives
M01_TOP_K = 50


def mine_hard_negatives_with_model():
    """Use M01 to find hard negatives in the training set.

    For each query:
    1. Encode query with M01
    2. Encode all candidate passages with M01
    3. Find top-K passages
    4. Those in top-K that are currently labeled negative → hard negatives

    Returns: dict query → list of hard negative doc_ids
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading M01 from: {M01_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(M01_PATH))
    model = AutoModel.from_pretrained(str(M01_PATH), trust_remote_code=True)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Collect all unique (query, positive_id, negative_ids) from training data
    train_rows = load_jsonl(TRAIN_94K)
    print(f"Loaded {len(train_rows)} training rows")

    hard_negatives = {}
    batch_size = 16

    for idx, row in enumerate(train_rows):
        query = row.get("query", "")
        pos_docs = row.get("positive_passages") or row.get("positive") or []
        if isinstance(pos_docs, dict):
            pos_docs = [pos_docs]
        neg_docs = row.get("negative_passages") or row.get("negative") or []
        if isinstance(neg_docs, dict):
            neg_docs = [neg_docs]

        if not neg_docs:
            continue

        # Get positive doc_ids to exclude from hard negatives
        pos_ids = set()
        for p in pos_docs:
            pid = p.get("doc_id") or p.get("docid") or p.get("id", "")
            if pid:
                pos_ids.add(str(pid))

        # Collect negative passages with content
        neg_passages = []
        for n in neg_docs:
            ndoc_id = n.get("doc_id") or n.get("docid") or n.get("id", "")
            ncontent = n.get("content") or n.get("text") or n.get("passage", "")
            if ndoc_id and ncontent:
                neg_passages.append((str(ndoc_id), ncontent))

        if len(neg_passages) <= MAX_HARD_NEGS:
            continue  # Already few negatives, no need to mine

        # Encode query and negatives
        with torch.no_grad():
            # Query encoding
            q_inputs = tokenizer(
                [query], padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(device)
            q_emb = model(**q_inputs).last_hidden_state[:, -1, :]
            q_emb = torch.nn.functional.normalize(q_emb, dim=-1)

            # Negative encoding (batch)
            neg_texts = [n[1] for n in neg_passages]
            all_sims = []

            for i in range(0, len(neg_texts), batch_size):
                batch = neg_texts[i:i + batch_size]
                n_inputs = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=512, return_tensors="pt"
                ).to(device)
                n_emb = model(**n_inputs).last_hidden_state[:, -1, :]
                n_emb = torch.nn.functional.normalize(n_emb, dim=-1)
                sims = (q_emb @ n_emb.T).squeeze(0)
                all_sims.extend(sims.cpu().tolist())

        # Find hard negatives: high similarity, but labeled negative
        neg_sims = list(zip(neg_passages, all_sims))
        neg_sims.sort(key=lambda x: x[1], reverse=True)

        # Top MAX_HARD_NEGS that are NOT positives
        hard = []
        for (ndoc_id, ncontent), sim in neg_sims[:M01_TOP_K]:
            if ndoc_id not in pos_ids and len(hard) < MAX_HARD_NEGS:
                hard.append({"doc_id": ndoc_id, "content": ncontent, "similarity": float(sim)})

        if hard:
            hard_negatives[idx] = hard

        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{len(train_rows)}... found {len(hard_negatives)} with hard negatives")

    print(f"Total queries with hard negatives: {len(hard_negatives)}")
    return hard_negatives


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Mine hard negatives
    print("=== Mining Hard Negatives ===")
    hard_neg_map = mine_hard_negatives_with_model()

    # Augment training data
    print("\n=== Augmenting Training Data ===")
    train_rows = load_jsonl(TRAIN_94K)

    augmented = 0
    for idx, row in enumerate(train_rows):
        if idx in hard_neg_map:
            hard_negs = hard_neg_map[idx]
            # Add hard negatives to the negative list
            existing_negs = row.get("negative_passages") or row.get("negative") or []
            if isinstance(existing_negs, dict):
                existing_negs = [existing_negs]

            for hn in hard_negs:
                existing_negs.append({
                    "doc_id": hn["doc_id"],
                    "content": hn["content"],
                    "_hard_negative": True,
                    "_m01_similarity": hn["similarity"],
                })

            if "negative_passages" in row:
                row["negative_passages"] = existing_negs
            else:
                row["negative"] = existing_negs
            augmented += 1

    print(f"Rows augmented with hard negatives: {augmented}")

    # Save
    save_jsonl(OUTPUT_TRAIN, train_rows)
    print(f"\nSaved: {OUTPUT_TRAIN}")
    print(f"  SHA-256: {sha256_file(OUTPUT_TRAIN)}")

    # Save config
    config = {
        "model_id": "M11",
        "base_model": "M00",
        "source_data": str(TRAIN_94K),
        "method": "in-training-set hard negative mining using M01",
        "m01_model": str(M01_PATH),
        "m01_sha": "b25b3b08a3199a788ea5e8bc005ee20ab831ada36cb4eccd7ba915e0d6e02501",
        "max_hard_negatives_per_query": MAX_HARD_NEGS,
        "m01_top_k": M01_TOP_K,
        "rows_augmented": augmented,
        "total_rows": len(train_rows),
        "output_sha": sha256_file(OUTPUT_TRAIN),
    }
    with open(OUTPUT_CONFIG, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {OUTPUT_CONFIG}")


if __name__ == "__main__":
    main()
