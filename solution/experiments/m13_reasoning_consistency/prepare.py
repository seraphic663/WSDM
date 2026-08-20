#!/usr/bin/env python3
"""M13: Reasoning-Content Semantic Consistency Filtering.

Model: M13
Source: M00 (Qwen3-Embedding-0.6B, SHA 0437e45c...e23fd)
Data: early_stop_v1 train 94,113 rows, with reweight_rate modified by
      reasoning-document semantic consistency
Method: Use M01 to encode (query + document) and (query + agent reasoning).
        If the two embeddings are semantically far apart,
        the agent's reasoning does NOT support this document as a good positive.
        Lower the weight for inconsistent pairs, raise for consistent ones.

This is a fine-grained signal: it directly measures whether the Agent
"agreed" with the document selection.
"""

import json
from pathlib import Path
from collections import defaultdict
from common import (
    TRAIN_94K, M01_PATH, TRAJECTORY_TAR, DATA_RAW,
    load_jsonl, save_jsonl, sha256_file,
)

OUTPUT_DIR = Path("/root/data/LRAT/ccir/data/experiments/m13_reasoning_consistency")
OUTPUT_TRAIN = OUTPUT_DIR / "train_m13.jsonl"
OUTPUT_CONFIG = OUTPUT_DIR / "m13_config.json"

# Weight multipliers based on consistency score
# High consistency: agent reasoning strongly supports this doc → weight up
# Low consistency: agent reasoning contradicts or ignores this doc → weight down
CONSISTENCY_THRESHOLDS = {
    "high": (0.75, 1.3),      # cosine > 0.75 → ×1.3
    "medium": (0.50, 1.0),    # cosine 0.5-0.75 → ×1.0 (keep)
    "low": (0.25, 0.7),       # cosine 0.25-0.5 → ×0.7
    "very_low": (0.0, 0.5),   # cosine < 0.25 → ×0.5
}


def compute_consistency_scores():
    """Compute reasoning-document consistency using M01 embeddings.

    For each training pair that has agent reasoning in the trajectory:
    1. Encode (query + document content) → doc_emb
    2. Encode (query + agent reasoning) → reasoning_emb
    3. Cosine similarity = consistency score

    Returns: dict index → consistency_score
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading M01 from: {M01_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(M01_PATH))
    model = AutoModel.from_pretrained(str(M01_PATH), trust_remote_code=True)
    model = model.to(device)
    model.eval()

    # Encode function
    def encode_texts(texts, max_length=512):
        """Encode a list of texts and return normalized embeddings."""
        embeddings = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, -1, :]
                emb = torch.nn.functional.normalize(emb, dim=-1)
            embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0)

    # Load training data
    train_rows = load_jsonl(TRAIN_94K)
    print(f"Loaded {len(train_rows)} training rows")

    # For each row, construct (query+doc) and (query+reasoning) pairs
    doc_texts = []
    reasoning_texts = []
    valid_indices = []

    for idx, row in enumerate(train_rows):
        query = row.get("query", "")
        pos_docs = row.get("positive_passages") or row.get("positive") or []
        if isinstance(pos_docs, dict):
            pos_docs = [pos_docs]

        # Get document content
        doc_content = ""
        for pos in pos_docs:
            content = pos.get("content") or pos.get("text") or pos.get("passage", "")
            if content:
                doc_content = content[:500]  # Truncate
                break

        # Get reasoning (may be in trajectory, not directly in pair)
        # Use any reasoning-like field in the pair or trajectory metadata
        reasoning = row.get("reasoning") or row.get("agent_reasoning") or ""
        if not reasoning:
            # Check trajectory metadata
            traj_meta = row.get("trajectory_metadata") or {}
            reasoning = traj_meta.get("reasoning") or ""

        if doc_content and reasoning:
            doc_texts.append(f"Instruct: {query}\nQuery:{doc_content}")
            reasoning_texts.append(f"Instruct: {query}\nQuery:{reasoning[:500]}")
            valid_indices.append(idx)

    print(f"Rows with both doc content and reasoning: {len(valid_indices)}")

    if len(valid_indices) == 0:
        print("WARNING: No rows have reasoning data. Skipping consistency computation.")
        return {}

    # Encode
    print("Encoding doc texts...")
    doc_embs = encode_texts(doc_texts)
    print("Encoding reasoning texts...")
    reasoning_embs = encode_texts(reasoning_texts)

    # Compute cosine similarities
    doc_embs = doc_embs.to(device)
    reasoning_embs = reasoning_embs.to(device)
    similarities = torch.nn.functional.cosine_similarity(doc_embs, reasoning_embs, dim=-1)
    similarities = similarities.cpu().tolist()

    # Build result
    consistency = {}
    for idx, sim in zip(valid_indices, similarities):
        consistency[idx] = float(sim)

    # Distribution stats
    high = sum(1 for s in consistency.values() if s > 0.75)
    mid = sum(1 for s in consistency.values() if 0.5 <= s <= 0.75)
    low = sum(1 for s in consistency.values() if 0.25 <= s < 0.5)
    very_low = sum(1 for s in consistency.values() if s < 0.25)
    print(f"Consistency distribution: high={high}, mid={mid}, low={low}, very_low={very_low}")
    print(f"Mean consistency: {sum(consistency.values())/len(consistency):.4f}")

    return consistency


def get_weight(score: float) -> float:
    """Get weight multiplier based on consistency score."""
    if score > 0.75:
        return 1.3
    elif score > 0.5:
        return 1.0
    elif score > 0.25:
        return 0.7
    else:
        return 0.5


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Compute consistency scores
    print("=== Computing Reasoning-Document Consistency ===")
    consistency = compute_consistency_scores()

    if not consistency:
        print("No consistency data available. Check that training data has reasoning fields.")
        # Even without reasoning, we can still save an identity mapping
        train_rows = load_jsonl(TRAIN_94K)
        save_jsonl(OUTPUT_TRAIN, train_rows)
        return

    # Apply weights
    print("\n=== Applying Consistency Weights ===")
    train_rows = load_jsonl(TRAIN_94K)
    weighted = 0
    weight_stats = defaultdict(int)

    for idx, row in enumerate(train_rows):
        if idx in consistency:
            score = consistency[idx]
            weight = get_weight(score)
            original_rr = row.get("reweight_rate", 1.0)
            row["reweight_rate"] = original_rr * weight
            row["_consistency_score"] = score
            row["_consistency_weight"] = weight
            weighted += 1
            if score > 0.75:
                weight_stats["high"] += 1
            elif score > 0.5:
                weight_stats["medium"] += 1
            elif score > 0.25:
                weight_stats["low"] += 1
            else:
                weight_stats["very_low"] += 1

    print(f"Weighted rows: {weighted}/{len(train_rows)}")
    print(f"Weight distribution: {dict(weight_stats)}")

    # Save
    save_jsonl(OUTPUT_TRAIN, train_rows)
    print(f"\nSaved: {OUTPUT_TRAIN}")
    print(f"  SHA-256: {sha256_file(OUTPUT_TRAIN)}")

    config = {
        "model_id": "M13",
        "base_model": "M00",
        "base_model_sha": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
        "source_data": str(TRAIN_94K),
        "method": "reasoning-document semantic consistency weighting",
        "encoder_model": str(M01_PATH),
        "consistency_thresholds": CONSISTENCY_THRESHOLDS,
        "weighted_rows": weighted,
        "weight_distribution": dict(weight_stats),
        "output_sha": sha256_file(OUTPUT_TRAIN),
    }
    with open(OUTPUT_CONFIG, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {OUTPUT_CONFIG}")


if __name__ == "__main__":
    main()
