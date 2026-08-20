#!/usr/bin/env python3
"""M12: Sequence-Level Training Signals.

Model: M12
Source: M00 (Qwen3-Embedding-0.6B, SHA 0437e45c...e23fd)
Data: early_stop_v1 train 94,113 rows, with reweight_rate modified by turn position
Method: Later turns in a trajectory get higher weights
        Turn 1: ×1.0, Turn 2-3: ×1.1, Turn ≥4: ×1.2

Rationale: Agent经过多轮筛选后选择的文档更可能与任务核心相关。
"""

import json
from pathlib import Path
from common import (
    TRAIN_94K, PROVENANCE_PATH, TRAJECTORY_TAR, DATA_RAW,
    load_jsonl, save_jsonl, sha256_file,
)
import tarfile

OUTPUT_DIR = Path("/root/data/LRAT/ccir/data/experiments/m12_sequence_weight")
OUTPUT_TRAIN = OUTPUT_DIR / "train_m12.jsonl"
OUTPUT_CONFIG = OUTPUT_DIR / "m12_config.json"

TURN_WEIGHTS = {
    1: 1.0,
    2: 1.1,
    3: 1.1,
    # >= 4: 1.2
}


def get_turn_weight(turn_number: int) -> float:
    if turn_number >= 4:
        return 1.2
    return TURN_WEIGHTS.get(turn_number, 1.0)


def load_provenance_with_turns():
    """Load provenance and extract turn numbers.

    Returns dict: (query_norm, positive_doc_id) → {"trajectory_id": str, "turn_number": int}
    """
    mapping = {}
    if not PROVENANCE_PATH.exists():
        print(f"ERROR: {PROVENANCE_PATH} not found")
        return mapping

    for row in load_jsonl(PROVENANCE_PATH):
        key = (row.get("query_normalized", row.get("query", "")),
               row.get("positive_doc_id", ""))
        tid = row.get("trajectory_id", "")
        turn = row.get("turn_number") or row.get("turn") or row.get("step_number") or 1
        try:
            turn = int(turn)
        except (ValueError, TypeError):
            turn = 1

        if key[0] and key[1]:
            mapping[key] = {"trajectory_id": tid, "turn_number": turn}
    return mapping


def count_turns_in_trajectory(traj_id: str, trajectories: dict) -> int:
    """Get the total number of turns in a trajectory."""
    if traj_id in trajectories:
        traj = trajectories[traj_id]
        turns = traj.get("turns") or traj.get("steps") or []
        return len(turns)
    return 0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load provenance with turn info
    print("Loading provenance with turn numbers...")
    prov_map = load_provenance_with_turns()
    print(f"  Provenance entries: {len(prov_map)}")

    # Count turns
    from collections import Counter
    turn_counts = Counter(v["turn_number"] for v in prov_map.values())
    print(f"  Turn distribution: {dict(sorted(turn_counts.items()))}")

    # 2. Load training data
    print(f"Loading training data: {TRAIN_94K}")
    train_rows = load_jsonl(TRAIN_94K)
    print(f"  Training rows: {len(train_rows)}")

    # 3. Modify reweight_rate by turn
    modified = 0
    unmatched = 0
    turn_stats = Counter()

    for row in train_rows:
        query = row.get("query", "")
        query_norm = " ".join(query.strip().lower().split())
        pos_docs = row.get("positive_passages") or row.get("positive") or []
        if isinstance(pos_docs, dict):
            pos_docs = [pos_docs]
        if not pos_docs:
            continue

        found = False
        for pos in pos_docs:
            doc_id = pos.get("doc_id") or pos.get("docid") or pos.get("id", "")
            key = (query_norm, str(doc_id))
            key_orig = (row.get("query", ""), str(doc_id))

            prov = prov_map.get(key) or prov_map.get(key_orig)
            if prov:
                turn = prov["turn_number"]
                weight = get_turn_weight(turn)
                original_rr = row.get("reweight_rate", 1.0)
                row["reweight_rate"] = original_rr * weight
                modified += 1
                turn_stats[turn] += 1
                found = True
                break

        if not found:
            unmatched += 1

    print(f"  Modified: {modified}, Unmatched: {unmatched}")
    print(f"  Turn weight distribution: {dict(sorted(turn_stats.items()))}")

    # 4. Save
    save_jsonl(OUTPUT_TRAIN, train_rows)
    print(f"Saved: {OUTPUT_TRAIN}")
    print(f"  SHA-256: {sha256_file(OUTPUT_TRAIN)}")

    # 5. Save config
    config = {
        "model_id": "M12",
        "base_model": "M00",
        "base_model_sha": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
        "source_data": str(TRAIN_94K),
        "method": "sequence-level turn weighting",
        "turn_weights": TURN_WEIGHTS,
        "turn_4plus_weight": 1.2,
        "rows_modified": modified,
        "rows_unmatched": unmatched,
        "turn_distribution": dict(sorted(turn_stats.items())),
        "output_sha": sha256_file(OUTPUT_TRAIN),
    }
    with open(OUTPUT_CONFIG, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {OUTPUT_CONFIG}")


if __name__ == "__main__":
    main()
