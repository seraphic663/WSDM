#!/usr/bin/env python3
"""LRAT Experiment Shared Utilities.

Common functions for all six experiment directions (M08–M13).
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Server paths
SERVER_ROOT = Path("/root/data/LRAT")
CCIR = SERVER_ROOT / "ccir"
DATA_RAW = CCIR / "data" / "raw"
DATA_EXP = CCIR / "data" / "experiments"
EARLY_STOP_V1 = DATA_EXP / "early_stop_v1"
OUTPUTS = CCIR / "outputs"
CHECKPOINTS = OUTPUTS / "checkpoints"
EXPERIMENTS = OUTPUTS / "experiments"
MODELS = CCIR / "models"
VENV_PYTHON = SERVER_ROOT / ".venv" / "bin" / "python"

# Key model paths
M00_PATH = MODELS / "Qwen3-Embedding-0.6B"
M01_PATH = MODELS / "Qwen3-Embedding-0.6B-LRAT-1epoch-20260716"

# Key data paths
TRAIN_94K = EARLY_STOP_V1 / "train.jsonl"
DEV_1500 = EARLY_STOP_V1 / "dev.jsonl"
TRAJECTORY_TAR = DATA_RAW / "LRAT-trajectories.tar.gz"

# SHA constants
M00_SHA = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
M01_SHA = "b25b3b08a3199a788ea5e8bc005ee20ab831ada36cb4eccd7ba915e0d6e02501"
M07_TRAIN_SHA = "158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9"
PROVENANCE_SHA = "924d373cffbc8f229c326c1de02e7e223c87109e57c32c1f2be29ea2e65f0b15"
TRAJECTORY_TAR_SHA = "fb8ca29a7807e334fa0eab2d22fd3c3d52852c2f42f534969c4b1605578617a9"

# Provenance path (from trajectory quality experiment)
PROVENANCE_PATH = EXPERIMENTS / "trajectory_quality_v1_20260724" / "trajectory_provenance_v1.jsonl"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    """Compute SHA-256 of a string."""
    return hashlib.sha256(s.encode()).hexdigest()


def load_jsonl(path: Path) -> List[dict]:
    """Load a JSONL file into a list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: List[dict]):
    """Save a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_provenance() -> Dict[str, dict]:
    """Load trajectory provenance mapping.

    Returns dict mapping (query, positive_doc_id) -> provenance record.
    """
    prov = {}
    if not PROVENANCE_PATH.exists():
        print(f"ERROR: Provenance file not found: {PROVENANCE_PATH}")
        return prov

    for row in load_jsonl(PROVENANCE_PATH):
        key = (row.get("query", ""), row.get("positive_doc_id", ""))
        prov[key] = row

    return prov


def load_trajectories() -> Dict[str, dict]:
    """Load trajectory data from the tar.gz archive.

    Returns dict mapping trajectory_id -> trajectory record.

    Note: This reads from within the tar.gz. If already extracted,
    it reads from the extracted directory.
    """
    import tarfile

    extracted_dir = DATA_RAW / "LRAT-trajectories"
    trajectories = {}

    if extracted_dir.exists():
        # Read from extracted directory
        for f in extracted_dir.glob("*.jsonl"):
            for row in load_jsonl(f):
                tid = row.get("trajectory_id") or row.get("task_id")
                if tid:
                    trajectories[tid] = row
    elif TRAJECTORY_TAR.exists():
        # Read from tar.gz
        with tarfile.open(TRAJECTORY_TAR, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".jsonl"):
                    f = tar.extractfile(member)
                    if f:
                        for line in f:
                            line = line.decode("utf-8").strip()
                            if line:
                                row = json.loads(line)
                                tid = row.get("trajectory_id") or row.get("task_id")
                                if tid:
                                    trajectories[tid] = row
    else:
        print(f"ERROR: No trajectory data found at {TRAJECTORY_TAR} or {extracted_dir}")

    return trajectories


def compute_dev1500_metrics(eval_json_path: Path) -> dict:
    """Parse evaluation JSON output and extract metrics.

    Expected format:
    {
        "recall@1": float, "recall@5": float, "recall@10": float,
        "mrr": float, ...
    }
    """
    with open(eval_json_path) as f:
        data = json.load(f)
    return {
        "R@1": data.get("recall@1", data.get("Recall@1", 0)),
        "R@5": data.get("recall@5", data.get("Recall@5", 0)),
        "R@10": data.get("recall@10", data.get("Recall@10", 0)),
        "MRR": data.get("mrr", data.get("MRR", 0)),
    }


def compare_metrics(baseline: dict, candidate: dict) -> dict:
    """Compare candidate metrics against baseline.

    Returns dict with deltas and a 'recommend' boolean (>=2/4 metrics improved).
    """
    deltas = {}
    improvements = 0
    for key in ["R@1", "R@5", "R@10", "MRR"]:
        b = baseline.get(key, 0)
        c = candidate.get(key, 0)
        deltas[f"Δ{key}"] = c - b
        if c > b:
            improvements += 1

    deltas["improvements"] = improvements
    deltas["recommend"] = improvements >= 2
    return deltas


def write_run_config(output_dir: Path, config: dict):
    """Write run_config.env and run_config.json for a training run."""
    # JSON version
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # ENV version (for shell sourcing)
    with open(output_dir / "run_config.env", "w") as f:
        for k, v in config.items():
            f.write(f'{k}="{v}"\n')


def mark_completed(output_dir: Path, completed_at: str = None):
    """Write COMPLETED marker file."""
    from datetime import datetime
    if completed_at is None:
        completed_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00")
    with open(output_dir / "COMPLETED", "w") as f:
        f.write(f"completed_at={completed_at}\n")


def mark_failed(output_dir: Path, reason: str):
    """Write FAILED marker file."""
    with open(output_dir / "FAILED", "w") as f:
        f.write(f"reason={reason}\n")


if __name__ == "__main__":
    print(f"Server root: {SERVER_ROOT}")
    print(f"M00 path: {M00_PATH}")
    print(f"Train data: {TRAIN_94K}")
    print(f"Dev data: {DEV_1500}")
    print(f"Provenance: {PROVENANCE_PATH}")
