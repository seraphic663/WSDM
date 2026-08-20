#!/usr/bin/env python3
"""Wait for the formal run, then execute isolated short GPU simulations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
from safetensors import safe_open


PROJECT = Path("/root/data/LRAT")
DATA = PROJECT / "ccir/data/simulations/strategy_suite_20260718"
OUT = PROJECT / "ccir/outputs/simulations/strategy_suite_20260718"
FORMAL_RUN = "qwen3_dual_epoch3_from_epoch2_pathfix_20260718"
M00 = PROJECT / "ccir/models/Qwen3-Embedding-0.6B"
M00_SHA = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_rows() -> list[tuple[int, int, int]]:
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    return [tuple(map(int, line.split(","))) for line in text.strip().splitlines()]


def relevant_processes() -> str:
    text = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    return "\n".join(line for line in text.splitlines() if any(token in line for token in ("torchrun", "evaluate_qwen3", "wait_then_eval_qwen3_run")))


def wait_for_formal_release() -> dict:
    run = PROJECT / f"ccir/outputs/checkpoints/{FORMAL_RUN}"
    started = time.time()
    idle_streak = 0
    while True:
        if (run / "FAILED").exists():
            raise RuntimeError("formal 3epoch run failed; GPU suite is blocked")
        completed = (run / "COMPLETED").exists() and not (run / "RUNNING").exists()
        eval_done = (PROJECT / f"ccir/outputs/eval/{FORMAL_RUN}_dev500.json").exists()
        processes = relevant_processes()
        formal_busy = FORMAL_RUN in processes
        rows = gpu_rows()
        idle = all(memory <= 2048 and util <= 20 for _, memory, util in rows)
        idle_streak = idle_streak + 1 if idle else 0
        if completed and eval_done and not formal_busy and idle_streak >= 3:
            return {"wait_started_at": datetime.fromtimestamp(started).astimezone().isoformat(), "released_at": now(), "gpu_snapshot": rows}
        time.sleep(60)


def monitor_process(process: subprocess.Popen, log_handle) -> tuple[int, list[int]]:
    peaks = [0, 0]
    while process.poll() is None:
        for index, memory, _ in gpu_rows():
            if index < len(peaks):
                peaks[index] = max(peaks[index], memory)
        time.sleep(1)
    return process.returncode, peaks


def parse_train_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    matches = re.findall(r"\{'train_runtime': ([0-9.]+),.*?'train_loss': ([0-9.eE+-]+)", text)
    steps = [int(value) for value in re.findall(r"([0-9]+)/10", text)]
    return {
        "runtime_seconds": float(matches[-1][0]) if matches else None,
        "train_loss": float(matches[-1][1]) if matches else None,
        "max_step": max(steps) if steps else None,
        "traceback_count": text.count("Traceback"),
        "oom_count": text.count("CUDA out of memory"),
        "nccl_error_count": len(re.findall(r"NCCL.*(?:error|failed)", text, re.I)),
    }


def tensor_delta(model: Path) -> dict:
    base_path = M00 / "model.safetensors"
    model_path = model / "model.safetensors"
    squared = 0.0
    base_squared = 0.0
    count = 0
    with safe_open(base_path, framework="pt", device="cpu") as base, safe_open(model_path, framework="pt", device="cpu") as candidate:
        if list(base.keys()) != list(candidate.keys()):
            raise ValueError("delta tensor keys differ")
        for key in base.keys():
            a = base.get_tensor(key).float()
            b = candidate.get_tensor(key).float()
            if a.shape != b.shape:
                raise ValueError(f"delta shape differs: {key}")
            squared += torch.sum((b - a) ** 2).item()
            base_squared += torch.sum(a ** 2).item()
            count += a.numel()
    return {"l2": squared ** 0.5, "relative_l2": (squared / base_squared) ** 0.5, "rms": (squared / count) ** 0.5, "elements": count}


def run_training(name: str, data: Path, group: int, lr: str) -> dict:
    run_id = f"strategy_suite_20260718_{name}"
    run_dir = OUT / f"runs/{run_id}"
    cache = DATA / f"cache/{run_id}"
    main_log = OUT / f"logs/{run_id}.log"
    supervisor_log = OUT / f"logs/{run_id}.supervisor.log"
    for path in (run_dir, main_log, supervisor_log):
        if path.exists():
            raise FileExistsError(path)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    main_log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "RUN_ID": run_id, "MODEL_PATH": str(M00), "MODEL_SHA256": M00_SHA,
        "TRAIN_DATA": str(data), "DATA_SHA256": sha256(data),
        "OUTPUT_DIR": str(run_dir), "LOG_FILE": str(main_log), "CACHE_PATH": str(cache),
        "NUM_EPOCHS": "1", "MAX_STEPS": "10", "TRAIN_GROUP_SIZE": str(group),
        "PER_DEVICE_BATCH": "1", "GRADIENT_ACCUMULATION": "4",
        "QUERY_MAX_LEN": "128", "PASSAGE_MAX_LEN": "512", "LEARNING_RATE": lr,
        "SAVE_STEPS": "100", "SAVE_TOTAL_LIMIT": "1", "RESUME_MODE": "auto",
        "CUDA_VISIBLE_DEVICES": "0,1", "MASTER_PORT": str(29600 + len(list((OUT / "runs").glob("*")))),
        "MAX_ATTEMPTS": "3", "RETRY_DELAY_SECONDS": "10",
    })
    started = now()
    with supervisor_log.open("x") as log:
        process = subprocess.Popen(["bash", "solution/scripts/run_qwen3_dual_supervised.sh"], cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
        rc, peaks = monitor_process(process, log)
    result = {"name": name, "run_id": run_id, "started_at": started, "completed_at": now(), "return_code": rc, "data": str(data), "data_sha256": sha256(data), "group_size": group, "learning_rate": lr, "peak_gpu_mib": peaks}
    result.update(parse_train_log(main_log))
    if rc == 0 and (run_dir / "COMPLETED").exists():
        result["model_sha256"] = sha256(run_dir / "model.safetensors")
        result["weight_delta_from_m00"] = tensor_delta(run_dir)
        result["status"] = "PASS"
    else:
        result["status"] = "FAILED"
    return result


def run_evaluation(name: str, model: Path, data: Path) -> dict:
    result_path = OUT / f"eval/{name}.json"
    log_path = OUT / f"logs/{name}.eval.log"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists() or log_path.exists():
        raise FileExistsError(name)
    command = [str(PROJECT / ".venv/bin/python"), "solution/src/evaluate_qwen3_pairs.py", "--model", str(model), "--input", str(data), "--output", str(result_path), "--batch-size", "16", "--max-length", "512", "--device", "cuda:0"]
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = "0"
    started = now()
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
        rc, peaks = monitor_process(process, log)
    result = {"name": name, "started_at": started, "completed_at": now(), "return_code": rc, "model": str(model), "model_sha256": sha256(model / "model.safetensors"), "data": str(data), "data_sha256": sha256(data), "peak_gpu_mib": peaks}
    if rc == 0:
        metrics = json.loads(result_path.read_text())
        result["metrics"] = metrics.get("metrics", metrics)
        result["output_sha256"] = sha256(result_path)
        result["status"] = "PASS"
    else:
        result["status"] = "FAILED"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "solution/configs/strategy_suite_20260718.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result_file = OUT / "gpu_results.json"
    if result_file.exists():
        raise FileExistsError(result_file)
    release = wait_for_formal_release()
    experiments = [
        (item["name"], DATA / item["data"], int(item["group_size"]), str(item["learning_rate"]))
        for item in config["training"]
    ]
    training = [run_training(*experiment) for experiment in experiments]
    model_aliases = {"M00": M00, "G/tail_avg_2": OUT / "G/tail_avg_2", "G/tail_avg_3": OUT / "G/tail_avg_3"}
    evaluations = [
        run_evaluation(item["name"], model_aliases[item["model"]], DATA / item["data"])
        for item in config["evaluations"]
    ]
    payload = {"suite_id": "strategy_suite_20260718", "simulation_only": True, "formal_release": release, "training": training, "evaluations": evaluations, "completed_at": now(), "final_gpu_snapshot": gpu_rows()}
    result_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
