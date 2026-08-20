#!/usr/bin/env python3
"""M08: Trajectory-level success/failure weighting.

Model: M08
Source: M00 (Qwen3-Embedding-0.6B, SHA 0437e45c...e23fd)
Data: early_stop_v1 train 94,113 rows, with reweight_rate modified by trajectory final_status
Method: Success trajectory positives ×1.2, Failed trajectory positives ×0.8
Baseline: Same-config unit weight training (equivalent to M07 A@1000 reference)

This script prepares the training data by:
1. Loading provenance mapping (pair → trajectory)
2. Loading trajectory data to get final_status
3. Modifying reweight_rate based on final_status
4. Saving new train_m08.jsonl
"""

import json
import sys
import tarfile
from pathlib import Path
from common import (
    TRAIN_94K, PROVENANCE_PATH, TRAJECTORY_TAR, DATA_RAW,
    load_jsonl, save_jsonl, sha256_file, sha256_str,
)

OUTPUT_DIR = Path("/root/data/LRAT/ccir/data/experiments/m08_trajectory_weighting")
OUTPUT_TRAIN = OUTPUT_DIR / "train_m08.jsonl"
OUTPUT_CONFIG = OUTPUT_DIR / "m08_config.json"


def load_provenance_mapping():
    """Load provenance: maps (query, positive_doc_id) → trajectory_id"""
    mapping = {}
    if not PROVENANCE_PATH.exists():
        print(f"ERROR: {PROVENANCE_PATH} not found")
        return mapping

    for row in load_jsonl(PROVENANCE_PATH):
        key = (row.get("query_normalized", row.get("query", "")),
               row.get("positive_doc_id", ""))
        tid = row.get("trajectory_id", "")
        if key[0] and key[1] and tid:
            mapping[key] = tid
    return mapping


def load_trajectory_status():
    """Load trajectory_id → final_status from tar.gz"""
    status = {}
    extracted_dir = DATA_RAW / "LRAT-trajectories"

    def process_rows(rows):
        for row in rows:
            tid = row.get("trajectory_id") or row.get("task_id") or row.get("id")
            fs = row.get("final_status") or row.get("status") or ""
            if tid:
                status[tid] = fs.lower()

    if extracted_dir.exists():
        for f in sorted(extracted_dir.glob("*.jsonl")):
            process_rows(load_jsonl(f))
    elif TRAJECTORY_TAR.exists():
        with tarfile.open(TRAJECTORY_TAR, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".jsonl"):
                    f = tar.extractfile(member)
                    if f:
                        rows = [json.loads(line) for line in f.read().decode("utf-8").strip().split("\n") if line.strip()]
                        process_rows(rows)
    else:
        print(f"ERROR: No trajectory data")

    return status


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load provenance mapping
    print("Loading provenance mapping...")
    prov_map = load_provenance_mapping()
    print(f"  Provenance entries: {len(prov_map)}")

    # 2. Load trajectory status
    print("Loading trajectory statuses...")
    traj_status = load_trajectory_status()
    print(f"  Trajectories with status: {len(traj_status)}")

    # Count statuses
    successes = sum(1 for v in traj_status.values() if v in ("success", "completed", "true", "1"))
    failures = sum(1 for v in traj_status.values() if v in ("failed", "failure", "false", "0", "error"))
    print(f"  Success: {successes}, Failed: {failures}, Other: {len(traj_status) - successes - failures}")

    # 3. Load training data
    print(f"Loading training data: {TRAIN_94K}")
    train_rows = load_jsonl(TRAIN_94K)
    print(f"  Training rows: {len(train_rows)}")

    # 4. Modify reweight_rate
    modified = 0
    unmatched = 0
    success_weight = 1.2
    failure_weight = 0.8

    for row in train_rows:
        query = row.get("query", "")
        # Normalize query for matching (strip, lowercase)
        query_norm = " ".join(query.strip().lower().split())
        pos_docs = row.get("positive_passages") or row.get("positive") or []
        if isinstance(pos_docs, dict):
            pos_docs = [pos_docs]
        if not pos_docs:
            continue

        # Try to find trajectory for this pair
        found = False
        for pos in pos_docs:
            doc_id = pos.get("doc_id") or pos.get("docid") or pos.get("id", "")
            key = (query_norm, str(doc_id))

            # Also try with original query
            key_orig = (row.get("query", ""), str(doc_id))

            tid = prov_map.get(key) or prov_map.get(key_orig)
            if tid and tid in traj_status:
                status = traj_status[tid]
                original_rr = row.get("reweight_rate", 1.0)

                if status in ("success", "completed", "true", "1"):
                    row["reweight_rate"] = original_rr * success_weight
                    modified += 1
                elif status in ("failed", "failure", "false", "0", "error"):
                    row["reweight_rate"] = original_rr * failure_weight
                    modified += 1
                found = True
                break

        if not found:
            unmatched += 1

    print(f"  Modified: {modified}, Unmatched: {unmatched}")
    print(f"  Coverage: {modified}/{len(train_rows)} = {100*modified/len(train_rows):.1f}%")

    # 5. Save
    save_jsonl(OUTPUT_TRAIN, train_rows)
    print(f"Saved: {OUTPUT_TRAIN}")
    print(f"  SHA-256: {sha256_file(OUTPUT_TRAIN)}")

    # 6. Save config
    config = {
        "model_id": "M08",
        "base_model": "M00",
        "base_model_sha": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
        "source_data": str(TRAIN_94K),
        "source_data_sha": "158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9",
        "method": "trajectory success/failure weighting",
        "success_weight": success_weight,
        "failure_weight": failure_weight,
        "rows_modified": modified,
        "rows_unmatched": unmatched,
        "total_rows": len(train_rows),
        "output_sha": sha256_file(OUTPUT_TRAIN),
    }
    with open(OUTPUT_CONFIG, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {OUTPUT_CONFIG}")


if __name__ == "__main__":
    main()
