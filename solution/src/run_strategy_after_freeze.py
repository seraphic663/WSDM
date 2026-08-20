#!/usr/bin/env python3
"""Wait for formal training/eval, freeze M06, then launch the isolated GPU suite."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT = Path("/root/data/LRAT")
RUN_ID = "qwen3_dual_epoch3_from_epoch2_pathfix_20260718"
SOURCE = PROJECT / f"ccir/outputs/checkpoints/{RUN_ID}"
TARGET = PROJECT / "ccir/models/Qwen3-Embedding-0.6B-LRAT-3epoch-20260718"
EVAL_OUTPUT = PROJECT / f"ccir/outputs/eval/{RUN_ID}_dev500.json"
AUTO_EVAL_PID = 62947
SUITE_ROOT = PROJECT / "ccir/outputs/simulations/strategy_suite_20260718"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def gpu_snapshot() -> list[dict]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [
        {"index": int(parts[0]), "memory_mib": int(parts[1]), "utilization": int(parts[2])}
        for line in output.strip().splitlines()
        for parts in ([item.strip() for item in line.split(",")],)
    ]


def matching_processes() -> list[str]:
    output = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    return [
        line for line in output.splitlines()
        if RUN_ID in line and any(token in line for token in ("torchrun", "decoder_only.base", "evaluate_qwen3_pairs"))
    ]


def wait_for_release() -> dict:
    started = now()
    idle_streak = 0
    while True:
        if (SOURCE / "FAILED").exists():
            raise RuntimeError("formal 3epoch run failed")
        completed = (SOURCE / "COMPLETED").is_file() and not (SOURCE / "RUNNING").exists()
        eval_done = EVAL_OUTPUT.is_file()
        processes = matching_processes()
        watcher_exited = not Path(f"/proc/{AUTO_EVAL_PID}").exists()
        snapshot = gpu_snapshot()
        idle = len(snapshot) == 2 and all(row["memory_mib"] <= 2048 and row["utilization"] <= 20 for row in snapshot)
        idle_streak = idle_streak + 1 if idle else 0
        print(json.dumps({"at": now(), "completed": completed, "eval_done": eval_done, "auto_eval_exited": watcher_exited, "matching_processes": processes, "idle_streak": idle_streak, "gpu": snapshot}), flush=True)
        if completed and eval_done and watcher_exited and not processes and idle_streak >= 3:
            return {"wait_started_at": started, "released_at": now(), "gpu": snapshot}
        time.sleep(20)


def main() -> None:
    if TARGET.exists() or TARGET.is_symlink():
        raise FileExistsError(f"freeze target already exists: {TARGET}")
    release = wait_for_release()
    print(json.dumps({"phase": "freeze_start", "at": now(), "source": str(SOURCE), "target": str(TARGET)}), flush=True)
    freeze = subprocess.run(
        [sys.executable, "solution/src/freeze_inference_model.py", "--source", str(SOURCE), "--target", str(TARGET)],
        cwd=PROJECT,
        text=True,
        capture_output=True,
    )
    print(freeze.stdout, end="", flush=True)
    if freeze.returncode != 0:
        print(freeze.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"freeze failed with code {freeze.returncode}")
    if not TARGET.is_dir() or TARGET.stat().st_mode & 0o777 != 0o555:
        raise RuntimeError("freeze postcondition failed")
    handoff = {"formal_release": release, "freeze_manifest": str(TARGET / "FREEZE_MANIFEST.json"), "gpu_suite_started_at": now()}
    handoff_path = SUITE_ROOT / "freeze_then_gpu_handoff.json"
    if handoff_path.exists():
        raise FileExistsError(handoff_path)
    handoff_path.write_text(json.dumps(handoff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"phase": "gpu_suite_start", **handoff}), flush=True)
    result = subprocess.run([sys.executable, "solution/src/run_strategy_gpu_suite.py"], cwd=PROJECT)
    if result.returncode != 0:
        raise RuntimeError(f"GPU suite failed with code {result.returncode}")
    print(json.dumps({"phase": "all_complete", "at": now()}), flush=True)


if __name__ == "__main__":
    main()
