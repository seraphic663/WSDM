#!/usr/bin/env python3
"""Run the pre-registered LRAT reliability arms serially on the two-GPU server."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path


PROJECT = Path("/root/data/LRAT")
BASE_MODEL = PROJECT / "ccir/models/Qwen3-Embedding-0.6B"
DEV1500 = PROJECT / "ccir/data/experiments/early_stop_v1/dev.jsonl"
MODEL_SHA = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
DEV_SHA = "6b9cac0b1dc351062b4b23999bdd463b6df096be60c483734a1579256eef9f9e"
EXPECTED_MODEL_BYTES = 2_383_139_480
EXPECTED_ROWS = 94_113


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def file_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def gpu_rows() -> list[dict[str, int]]:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, cwd=PROJECT,
    ).stdout
    rows = []
    for line in output.splitlines():
        index, memory, utilization = [int(value.strip()) for value in line.split(",")]
        rows.append({"index": index, "memory_mib": memory, "utilization_percent": utilization})
    if len(rows) != 2:
        raise RuntimeError(f"expected two GPUs, found {rows}")
    return rows


def require_idle() -> None:
    for attempt in range(12):
        rows = gpu_rows()
        if all(row["memory_mib"] <= 2048 and row["utilization_percent"] <= 20 for row in rows):
            return
        if attempt < 11:
            time.sleep(5)
    raise RuntimeError(f"GPUs did not settle: {rows}")


def load_arm_inputs(manifest_path: Path, names: list[str]) -> tuple[dict, dict[str, dict]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = {}
    for name in names:
        record = manifest["outputs"][name]
        path = manifest_path.parent / record["path"]
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing regular arm file: {path}")
        actual_sha = sha256(path)
        if actual_sha != record["sha256"]:
            raise RuntimeError(f"arm SHA mismatch: {name}")
        if int(record["rows"]) != EXPECTED_ROWS or file_rows(path) != EXPECTED_ROWS:
            raise RuntimeError(f"arm row count mismatch: {name}")
        if int(record.get("rows_below_minimum", 1)) != 0:
            raise RuntimeError(f"arm has rows below the negative floor: {name}")
        inputs[name] = {"path": path, "sha256": actual_sha, "manifest": record}
    if manifest["contract"].get("locked_test_used") is not False:
        raise RuntimeError("arm manifest does not assert locked-test isolation")
    return manifest, inputs


def evaluate(model: Path, output: Path, log: Path) -> dict:
    command = [
        str(PROJECT / ".venv/bin/python"), "solution/src/evaluate_qwen3_pairs.py",
        "--model", str(model), "--input", str(DEV1500), "--output", str(output),
        "--batch-size", "16", "--max-length", "512", "--device", "cuda:0",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=PROJECT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
    value = json.loads(output.read_text(encoding="utf-8"))
    metrics = value["metrics"]
    for key in ("mrr", "recall_at_1", "recall_at_5", "recall_at_10"):
        if not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"non-finite {key}: {metrics[key]}")
    return value


def healthy(record: dict) -> bool:
    try:
        model_path = Path(record["model_path"])
        evaluation = Path(record["evaluation"])
        return (
            record["status"] == "completed"
            and record["locked_test_used"] is False
            and (model_path / "COMPLETED").is_file()
            and not (model_path / "RUNNING").exists()
            and not (model_path / "FAILED").exists()
            and (model_path / "model.safetensors").stat().st_size == EXPECTED_MODEL_BYTES
            and evaluation.is_file()
        )
    except (KeyError, OSError, TypeError):
        return False


def run_one(item: dict, args: argparse.Namespace, arm_inputs: dict[str, dict], roots: dict[str, Path]) -> dict:
    arm = item["arm"]
    seed = item["seed"]
    run_id = item["run_id"]
    train = arm_inputs[arm]["path"]
    train_sha = arm_inputs[arm]["sha256"]
    model_root = roots["models"] / run_id
    evaluation = roots["eval"] / f"{run_id}.json"
    train_log = roots["logs"] / f"{run_id}.train.log"
    eval_log = roots["logs"] / f"{run_id}.eval.log"
    cache = roots["cache"] / f"{arm}_seed{seed}"
    for path in (roots["models"], roots["eval"], roots["logs"], roots["cache"]):
        path.mkdir(parents=True, exist_ok=True)
    if model_root.exists() and not (model_root / "COMPLETED").is_file():
        raise RuntimeError(f"refusing incomplete output: {model_root}")
    if not model_root.exists():
        require_idle()
        environment = os.environ.copy()
        environment.update(
            {
                "RUN_ID": run_id,
                "MODEL_PATH": str(BASE_MODEL),
                "MODEL_SHA256": MODEL_SHA,
                "TRAIN_DATA": str(train),
                "DATA_SHA256": train_sha,
                "OUTPUT_DIR": str(model_root),
                "LOG_FILE": str(train_log),
                "CACHE_PATH": str(cache),
                "NUM_EPOCHS": "1",
                "MAX_STEPS": str(args.max_steps),
                "SAVE_STRATEGY": "no",
                "SAVE_STEPS": "1000000",
                "SAVE_TOTAL_LIMIT": "1",
                "TRAIN_GROUP_SIZE": str(args.group_size),
                "QUERY_MAX_LEN": "128",
                "PASSAGE_MAX_LEN": "512",
                "PER_DEVICE_BATCH": "1",
                "GRADIENT_ACCUMULATION": "4",
                "LEARNING_RATE": args.learning_rate,
                "SEED": str(seed),
                "MASTER_PORT": str(args.master_port),
                "RESUME_MODE": "never",
                "COLLISION_FREE_QUERIES": "0",
                "CUDA_VISIBLE_DEVICES": "0,1",
                "OMP_NUM_THREADS": "4",
            }
        )
        with train_log.open("w", encoding="utf-8") as handle:
            subprocess.run(
                ["bash", "solution/scripts/run_qwen3_dual_train.sh"], cwd=PROJECT, env=environment,
                stdout=handle, stderr=subprocess.STDOUT, check=True,
            )
        if not (model_root / "COMPLETED").is_file():
            raise RuntimeError(f"training returned without COMPLETED: {run_id}")
    model = model_root / "model.safetensors"
    if model.stat().st_size != EXPECTED_MODEL_BYTES:
        raise RuntimeError(f"unexpected final model size: {model}")
    if not evaluation.exists():
        require_idle()
        evaluation_value = evaluate(model_root, evaluation, eval_log)
    else:
        evaluation_value = json.loads(evaluation.read_text(encoding="utf-8"))
    require_idle()
    return {
        **item,
        "status": "completed",
        "train_data": str(train),
        "train_data_sha256": train_sha,
        "model_path": str(model_root),
        "model_sha256": sha256(model),
        "evaluation": str(evaluation),
        "evaluation_sha256": sha256(evaluation),
        "metrics": evaluation_value["metrics"],
        "sample1500": str(DEV1500),
        "sample1500_sha256": DEV_SHA,
        "locked_test_used": False,
        "completed_at": now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--arm-manifest", required=True, type=Path)
    parser.add_argument("--arms", nargs="+", default=["random", "exposure", "later_visit"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260716, 20260815, 20260816])
    parser.add_argument("--learning-rate", default="6e-6")
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--master-port", type=int, default=29682)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if float(args.learning_rate) <= 0 or args.group_size <= 1 or args.max_steps <= 0:
        raise SystemExit("invalid training hyperparameters")
    for path, expected in ((BASE_MODEL / "model.safetensors", MODEL_SHA), (DEV1500, DEV_SHA)):
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise RuntimeError(f"preflight identity failed: {path}")
    manifest, arm_inputs = load_arm_inputs(args.arm_manifest, args.arms)
    items = [
        {
            "run_id": f"{args.batch_id}_{arm}_s{seed}_st{args.max_steps}",
            "arm": arm,
            "seed": seed,
            "learning_rate": args.learning_rate,
            "group_size": args.group_size,
            "steps": args.max_steps,
        }
        for seed in args.seeds
        for arm in args.arms
    ]
    preflight = {
        "batch_id": args.batch_id,
        "items": items,
        "arm_manifest": str(args.arm_manifest),
        "arm_manifest_sha256": sha256(args.arm_manifest),
        "model_sha256": MODEL_SHA,
        "dev1500_sha256": DEV_SHA,
        "locked_test_read": False,
        "available_bytes": shutil.disk_usage(PROJECT).free,
    }
    if preflight["available_bytes"] < 100 * 1024**3:
        raise RuntimeError("less than 100 GiB available")
    if args.dry_run:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return

    root = PROJECT / "ccir/outputs" / args.batch_id
    roots = {
        "batch": root,
        "models": PROJECT / "ccir/outputs/checkpoints" / args.batch_id,
        "eval": PROJECT / "ccir/outputs/eval" / args.batch_id,
        "logs": root / "logs",
        "cache": PROJECT / "ccir/data/cache" / args.batch_id,
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "PREFLIGHT.json", {**preflight, "created_at": now(), "arm_contract": manifest["contract"]})
    results_path = root / "RESULTS.json"
    existing = json.loads(results_path.read_text(encoding="utf-8")).get("runs", []) if results_path.exists() else []
    records = {record["run_id"]: record for record in existing if healthy(record)}
    with (root / "batch.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for item in items:
            if item["run_id"] in records:
                continue
            atomic_json(root / "PROGRESS.json", {"status": "running", "current": item, "completed": len(records), "total": len(items), "updated_at": now()})
            record = run_one(item, args, arm_inputs, roots)
            records[item["run_id"]] = record
            atomic_json(results_path, {"batch_id": args.batch_id, "runs": list(records.values()), "locked_test_used": False})
        atomic_json(root / "PROGRESS.json", {"status": "completed", "completed": len(records), "total": len(items), "updated_at": now(), "locked_test_used": False})
    print(json.dumps({"batch_id": args.batch_id, "runs": len(records), "status": "completed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
