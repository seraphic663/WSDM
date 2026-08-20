#!/usr/bin/env python3
"""Parameterized, serial dual-GPU LRAT experiment runner.

This runner is deliberately conservative:

* it accepts lists of learning rates, group sizes, and reweight modes;
* it skips only healthy, configuration-matching completed runs;
* it records immutable input hashes automatically;
* it runs one dual-GPU training job at a time;
* it evaluates only the approved sample1500 file;
* it never opens the locked-test file.

The script is kept separate from the historical GRID32 runner so that the
completed 32-condition batch remains reproducible and untouched.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


PROJECT = Path("/root/data/LRAT")
BASE_MODEL = PROJECT / "ccir/models/Qwen3-Embedding-0.6B"
TRAIN_ON = PROJECT / "ccir/data/experiments/early_stop_v1/train.jsonl"
TRAIN_OFF = PROJECT / "ccir/data/experiments/grid32_20260812/unit_weight/train_unit_weight.jsonl"
DEV1500 = PROJECT / "ccir/data/experiments/early_stop_v1/dev.jsonl"
SPLIT_MANIFEST = PROJECT / "ccir/data/experiments/early_stop_v1/manifest.json"
UNIT_MANIFEST = PROJECT / "ccir/data/experiments/grid32_20260812/unit_weight/manifest.json"

MODEL_SHA = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
TRAIN_ON_SHA = "158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9"
TRAIN_OFF_SHA = "dc4191962b2acee40a2bea1315e76a7fd6036bdabbbb535124c0ecfc6d364d32"
DEV1500_SHA = "6b9cac0b1dc351062b4b23999bdd463b6df096be60c483734a1579256eef9f9e"
EXPECTED_MODEL_BYTES = 2_383_139_480
EXPECTED_TRAIN_ROWS = 94_113
EXPECTED_DEV_ROWS = 1_500
MIN_AVAILABLE_BYTES = 150 * 1024**3
MAX_BATCH_CACHE_BYTES = 100 * 1024**3

DEFAULT_EXISTING_BATCH = PROJECT / "ccir/outputs/grid32_20260812"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def canonical_lr(value: str | float) -> str:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"learning rate must be finite and positive: {value}")
    token = format(number, ".12g")
    token = token.replace("e-0", "e-").replace("e+0", "e+")
    return token


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_query(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def require_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise RuntimeError(f"required regular file is missing or invalid: {path}")


def check_sha(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def log(batch_log: Path, message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    batch_log.parent.mkdir(parents=True, exist_ok=True)
    with batch_log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_output(command: list[str], *, cwd: Path = PROJECT) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout


def gpu_rows() -> list[dict[str, int]]:
    output = run_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, int]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise RuntimeError(f"unexpected nvidia-smi output: {line}")
        index, memory, utilization = map(int, fields)
        rows.append({"index": index, "memory_mib": memory, "utilization_percent": utilization})
    if len(rows) != 2:
        raise RuntimeError(f"expected two A40 GPUs, got {rows}")
    return rows


def check_gpu_idle(*, attempts: int = 12, wait_seconds: float = 5.0) -> None:
    last = []
    for attempt in range(1, attempts + 1):
        last = gpu_rows()
        if all(row["memory_mib"] <= 2048 and row["utilization_percent"] <= 20 for row in last):
            return
        if attempt < attempts:
            time.sleep(wait_seconds)
    raise RuntimeError(f"GPUs did not settle: {last}")


def disk_status(cache_root: Path) -> dict[str, int]:
    usage = shutil.disk_usage(PROJECT)
    cache_bytes = 0
    if cache_root.exists():
        cache_bytes = sum(
            path.stat().st_size
            for path in cache_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
    if usage.free < MIN_AVAILABLE_BYTES:
        raise RuntimeError(f"available disk below guard: {usage.free} bytes")
    if cache_bytes > MAX_BATCH_CACHE_BYTES:
        raise RuntimeError(f"batch cache above guard: {cache_bytes} bytes")
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "available_bytes": usage.free,
        "batch_cache_bytes": cache_bytes,
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_query_partition() -> dict[str, object]:
    manifest = load_json(SPLIT_MANIFEST)
    if manifest["train"]["rows"] != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("unexpected train row count in split manifest")
    if manifest["dev"]["query_groups"] != EXPECTED_DEV_ROWS:
        raise RuntimeError("unexpected sample1500 query count in split manifest")
    if manifest["train"]["normalized_query_overlap_with_dev_or_test"] != 0:
        raise RuntimeError("split manifest reports train overlap with held-out data")

    train_queries: set[str] = set()
    with TRAIN_ON.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            query = row.get("query")
            if not isinstance(query, str):
                raise RuntimeError(f"train line {line_number} has no string query")
            train_queries.add(norm_query(query))

    dev_queries: set[str] = set()
    with DEV1500.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            query = row.get("query")
            if not isinstance(query, str):
                raise RuntimeError(f"sample1500 line {line_number} has no string query")
            dev_queries.add(norm_query(query))

    overlap = train_queries & dev_queries
    if overlap:
        raise RuntimeError(f"train/sample1500 query overlap detected: {len(overlap)}")
    if len(dev_queries) != EXPECTED_DEV_ROWS:
        raise RuntimeError(f"sample1500 has {len(dev_queries)} unique queries")

    unit_manifest = load_json(UNIT_MANIFEST)
    if unit_manifest["input"]["sha256"] != TRAIN_ON_SHA:
        raise RuntimeError("unit-weight input is not the locked train partition")
    if unit_manifest["output"]["sha256"] != TRAIN_OFF_SHA:
        raise RuntimeError("unit-weight output SHA differs from the locked artifact")

    return {
        "train_rows": EXPECTED_TRAIN_ROWS,
        "train_unique_normalized_queries": len(train_queries),
        "sample1500_rows": EXPECTED_DEV_ROWS,
        "sample1500_unique_normalized_queries": len(dev_queries),
        "train_sample1500_overlap": 0,
        "locked_test_read": False,
        "split_manifest": str(SPLIT_MANIFEST),
        "unit_weight_manifest": str(UNIT_MANIFEST),
    }


def parse_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def condition_key(
    *,
    learning_rate: str,
    group_size: int,
    reweight: str,
    max_steps: int,
    seed: int,
    train_sha: str,
) -> tuple[str, int, str, int, int, str]:
    return (canonical_lr(learning_rate), group_size, reweight, max_steps, seed, train_sha)


def condition_from_record(record: dict[str, object]) -> tuple[str, int, str, int, int, str] | None:
    try:
        model_path = Path(str(record["model_path"]))
        env = parse_env(model_path / "run_config.env")
        max_steps = int(env.get("max_steps", record.get("steps", 0)))
        seed = int(env.get("seed", 20260716))
        return condition_key(
            learning_rate=str(record["learning_rate"]),
            group_size=int(record["group_size"]),
            reweight=str(record["reweight"]),
            max_steps=max_steps,
            seed=seed,
            train_sha=str(record["train_data_sha256"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def healthy_record(record: dict[str, object]) -> bool:
    try:
        model_path = Path(str(record["model_path"]))
        evaluation = Path(str(record["evaluation"]))
        model = model_path / "model.safetensors"
        if record.get("status") not in {"completed", "already_complete"}:
            return False
        if record.get("locked_test_used") is not False:
            return False
        if (model_path / "RUNNING").exists() or (model_path / "FAILED").exists():
            return False
        if not (model_path / "COMPLETED").is_file():
            return False
        if not model.is_file() or model.is_symlink() or model.stat().st_size != EXPECTED_MODEL_BYTES:
            return False
        if not evaluation.is_file():
            return False
        metrics = load_json(evaluation).get("metrics", {})
        required = {"mrr", "recall_at_1", "recall_at_5", "recall_at_10"}
        return required.issubset(metrics) and all(math.isfinite(float(metrics[key])) for key in required)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def load_existing_records(batch_roots: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[tuple[str, int, str, int, int, str]] = set()
    for root in batch_roots:
        result_path = root / "RESULTS.json"
        if not result_path.is_file():
            continue
        try:
            values = load_json(result_path).get("runs", [])
        except (OSError, json.JSONDecodeError):
            continue
        for record in values:
            if not isinstance(record, dict) or not healthy_record(record):
                continue
            key = condition_from_record(record)
            if key is None or key in seen:
                continue
            seen.add(key)
            records.append(record)
    return records


def combinations(args: argparse.Namespace) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[tuple[str, int, str, int, int, str]] = set()
    for learning_rate in args.learning_rates:
        lr = canonical_lr(learning_rate)
        for group_size in args.group_sizes:
            if group_size <= 1:
                raise ValueError(f"group size must be >1: {group_size}")
            for reweight in args.reweight:
                train_sha = TRAIN_ON_SHA if reweight == "on" else TRAIN_OFF_SHA
                key = condition_key(
                    learning_rate=lr,
                    group_size=group_size,
                    reweight=reweight,
                    max_steps=args.max_steps,
                    seed=args.seed,
                    train_sha=train_sha,
                )
                if key in seen:
                    continue
                seen.add(key)
                result.append(
                    {
                        "learning_rate": lr,
                        "group_size": group_size,
                        "reweight": reweight,
                        "max_steps": args.max_steps,
                        "seed": args.seed,
                        "run_id": (
                            f"{args.batch_id}_lr{lr}_g{group_size}_rw{reweight}"
                            f"_s{args.seed}_st{args.max_steps}"
                        ),
                    }
                )
    if not result:
        raise ValueError("the requested grid is empty")
    return result


def preflight_inputs() -> dict[str, object]:
    if not (PROJECT / ".venv/bin/python").is_file():
        raise RuntimeError("server virtual environment is missing")
    for path in (BASE_MODEL / "model.safetensors", TRAIN_ON, TRAIN_OFF, DEV1500):
        require_file(path)
    check_sha(BASE_MODEL / "model.safetensors", MODEL_SHA)
    check_sha(TRAIN_ON, TRAIN_ON_SHA)
    check_sha(TRAIN_OFF, TRAIN_OFF_SHA)
    check_sha(DEV1500, DEV1500_SHA)
    return {
        "model": {"path": str(BASE_MODEL), "sha256": MODEL_SHA},
        "train_on": {"path": str(TRAIN_ON), "sha256": TRAIN_ON_SHA},
        "train_off": {"path": str(TRAIN_OFF), "sha256": TRAIN_OFF_SHA},
        "sample1500": {"path": str(DEV1500), "sha256": DEV1500_SHA},
        "locked_test": {"read": False, "used_for_training": False, "used_for_selection": False},
        "partition": check_query_partition(),
    }


def directory_bytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file() and not entry.is_symlink())


def cleanup_intermediate_checkpoints(output: Path, batch_log: Path) -> None:
    checkpoints = sorted(output.glob("checkpoint-*"))
    unexpected = [path for path in checkpoints if path.name != "checkpoint-500"]
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint directories require inspection: {unexpected}")
    for checkpoint in checkpoints:
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise RuntimeError(f"checkpoint is not a regular directory: {checkpoint}")
        size = directory_bytes(checkpoint)
        shutil.rmtree(checkpoint)
        log(batch_log, f"removed expected intermediate checkpoint {checkpoint} ({size} bytes)")


def evaluate(model_path: Path, output_path: Path, log_path: Path) -> dict:
    command = [
        str(PROJECT / ".venv/bin/python"),
        "solution/src/evaluate_qwen3_pairs.py",
        "--model",
        str(model_path),
        "--input",
        str(DEV1500),
        "--output",
        str(output_path),
        "--batch-size",
        "16",
        "--max-length",
        "512",
        "--device",
        "cuda:0",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=PROJECT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return load_json(output_path)


def run_one(
    item: dict[str, object],
    *,
    args: argparse.Namespace,
    batch_root: Path,
    run_root: Path,
    eval_root: Path,
    cache_root: Path,
    batch_log: Path,
) -> dict[str, object]:
    run_id = str(item["run_id"])
    learning_rate = str(item["learning_rate"])
    group_size = int(item["group_size"])
    reweight = str(item["reweight"])
    train_data = TRAIN_ON if reweight == "on" else TRAIN_OFF
    train_sha = TRAIN_ON_SHA if reweight == "on" else TRAIN_OFF_SHA
    output = run_root / run_id
    evaluation = eval_root / f"{run_id}.json"
    train_log = batch_root / "logs" / f"{run_id}.train.log"
    eval_log = batch_root / "logs" / f"{run_id}.eval.log"
    cache = cache_root / f"rw_{reweight}"
    # The first failed launch happened before torchrun: opening the per-run
    # log file failed because the batch-level logs directory did not exist.
    # Create it at the call boundary so both fresh and resumed runs are safe.
    train_log.parent.mkdir(parents=True, exist_ok=True)
    eval_log.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not (output / "COMPLETED").exists():
        raise RuntimeError(f"refusing to reuse incomplete output: {output}")
    if output.exists() and (output / "COMPLETED").exists():
        cleanup_intermediate_checkpoints(output, batch_log)
        if not evaluation.exists():
            log(batch_log, f"{run_id}: completed output found; evaluating it")
        else:
            log(batch_log, f"{run_id}: completed output and evaluation already present")
    else:
        check_gpu_idle()
        disk = disk_status(cache_root)
        log(batch_log, f"{run_id}: start; disk_available={disk['available_bytes']} cache={disk['batch_cache_bytes']}")
        environment = os.environ.copy()
        environment.update(
            {
                "RUN_ID": run_id,
                "MODEL_PATH": str(BASE_MODEL),
                "MODEL_SHA256": MODEL_SHA,
                "TRAIN_DATA": str(train_data),
                "DATA_SHA256": train_sha,
                "OUTPUT_DIR": str(output),
                "LOG_FILE": str(train_log),
                "CACHE_PATH": str(cache),
                "NUM_EPOCHS": "1",
                "MAX_STEPS": str(args.max_steps),
                "SAVE_STRATEGY": "no",
                "SAVE_STEPS": "1000000",
                "SAVE_TOTAL_LIMIT": "1",
                "TRAIN_GROUP_SIZE": str(group_size),
                "QUERY_MAX_LEN": "128",
                "PASSAGE_MAX_LEN": "512",
                "PER_DEVICE_BATCH": "1",
                "GRADIENT_ACCUMULATION": "4",
                "LEARNING_RATE": learning_rate,
                "SEED": str(args.seed),
                "MASTER_PORT": str(args.master_port),
                "RESUME_MODE": "never",
                "COLLISION_FREE_QUERIES": "0",
                "CUDA_VISIBLE_DEVICES": "0,1",
                "OMP_NUM_THREADS": "4",
            }
        )
        with train_log.open("w", encoding="utf-8") as handle:
            subprocess.run(
                ["bash", "solution/scripts/run_qwen3_dual_train.sh"],
                cwd=PROJECT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not (output / "COMPLETED").is_file():
            raise RuntimeError(f"training returned but COMPLETED is missing: {output}")
        cleanup_intermediate_checkpoints(output, batch_log)
        check_gpu_idle()
        log(batch_log, f"{run_id}: training complete; final model ready")

    model = output / "model.safetensors"
    if not model.is_file() or model.stat().st_size != EXPECTED_MODEL_BYTES:
        raise RuntimeError(f"final model is missing or has unexpected size: {model}")
    if not evaluation.exists():
        log(batch_log, f"{run_id}: evaluate sample1500")
        evaluation_result = evaluate(output, evaluation, eval_log)
    else:
        evaluation_result = load_json(evaluation)
    check_gpu_idle()
    metrics = evaluation_result["metrics"]
    required = ("mrr", "recall_at_1", "recall_at_5", "recall_at_10")
    if not all(math.isfinite(float(metrics[key])) for key in required):
        raise RuntimeError(f"non-finite evaluation metrics for {run_id}: {metrics}")
    model_sha = sha256_file(model)
    record = {
        "run_id": run_id,
        "learning_rate": learning_rate,
        "group_size": group_size,
        "reweight": reweight,
        "status": "completed",
        "steps": args.max_steps,
        "seed": args.seed,
        "train_data": str(train_data),
        "train_data_sha256": train_sha,
        "sample1500": str(DEV1500),
        "sample1500_sha256": DEV1500_SHA,
        "locked_test_used": False,
        "model_path": str(output),
        "model_sha256": model_sha,
        "evaluation": str(evaluation),
        "metrics": metrics,
        "completed_at": now(),
    }
    log(
        batch_log,
        f"{run_id}: MRR={metrics['mrr']:.9f} R@1={metrics['recall_at_1']:.6f} "
        f"R@5={metrics['recall_at_5']:.6f} R@10={metrics['recall_at_10']:.6f}",
    )
    return record


def rank(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        records,
        key=lambda item: (
            float(item["metrics"]["mrr"]),
            float(item["metrics"]["recall_at_1"]),
            float(item["metrics"]["recall_at_5"]),
            float(item["metrics"]["recall_at_10"]),
        ),
        reverse=True,
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--learning-rates", nargs="+", required=True)
    parser.add_argument("--group-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--reweight", nargs="+", choices=("on", "off"), required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--master-port", type=int, default=29642)
    parser.add_argument("--existing-batch-root", action="append", default=[str(DEFAULT_EXISTING_BATCH)])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    args.learning_rates = [canonical_lr(value) for value in args.learning_rates]
    items = combinations(args)
    existing_roots = [Path(value) for value in args.existing_batch_root]
    existing_records = load_existing_records(existing_roots)
    existing_by_key = {condition_from_record(record): record for record in existing_records}
    preflight = preflight_inputs()
    to_skip: list[dict[str, object]] = []
    to_run: list[dict[str, object]] = []
    for item in items:
        train_sha = TRAIN_ON_SHA if item["reweight"] == "on" else TRAIN_OFF_SHA
        key = condition_key(
            learning_rate=str(item["learning_rate"]),
            group_size=int(item["group_size"]),
            reweight=str(item["reweight"]),
            max_steps=int(item["max_steps"]),
            seed=int(item["seed"]),
            train_sha=train_sha,
        )
        if args.skip_completed and key in existing_by_key:
            to_skip.append({"requested": item, "existing": existing_by_key[key]})
        else:
            to_run.append(item)

    summary = {
        "batch_id": args.batch_id,
        "requested": len(items),
        "skip_completed": len(to_skip),
        "to_run": len(to_run),
        "locked_test_read": False,
        "existing_roots": [str(path) for path in existing_roots],
        "requested_conditions": items,
        "skipped_run_ids": [str(entry["requested"]["run_id"]) for entry in to_skip],
        "to_run_ids": [str(item["run_id"]) for item in to_run],
        "inputs": preflight,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    batch_root = PROJECT / "ccir/outputs" / args.batch_id
    run_root = PROJECT / "ccir/outputs/checkpoints" / args.batch_id
    eval_root = PROJECT / "ccir/outputs/eval" / args.batch_id
    cache_root = PROJECT / "ccir/data/cache" / args.batch_id
    batch_log = batch_root / "batch.log"
    lock_path = batch_root / "batch.lock"
    batch_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"batch lock is already held: {lock_path}")
        atomic_json(
            batch_root / "PREFLIGHT.json",
            {
                **summary,
                "created_at": now(),
                "git_commit": run_output(["git", "rev-parse", "HEAD"]).strip(),
                "grid": {
                    "learning_rates": args.learning_rates,
                    "group_sizes": args.group_sizes,
                    "reweight": args.reweight,
                    "max_steps": args.max_steps,
                    "seed": args.seed,
                    "world_size": 2,
                },
                "resource_guard": {
                    "one_run_at_a_time": True,
                    "min_available_bytes": MIN_AVAILABLE_BYTES,
                    "max_batch_cache_bytes": MAX_BATCH_CACHE_BYTES,
                    "initial_disk": disk_status(cache_root),
                },
            },
        )
        results: list[dict[str, object]] = []
        if (batch_root / "RESULTS.json").is_file():
            results = load_json(batch_root / "RESULTS.json").get("runs", [])
        results_by_key = {condition_from_record(record): record for record in results if condition_from_record(record)}
        log(batch_log, f"batch {args.batch_id}: requested={len(items)} skip={len(to_skip)} run={len(to_run)}")
        atomic_json(batch_root / "PROGRESS.json", {"batch_id": args.batch_id, "status": "running", "completed": len(results), "total": len(items), "updated_at": now()})

        for index, item in enumerate(to_run, 1):
            run_id = str(item["run_id"])
            log(batch_log, f"{run_id}: planned run {index}/{len(to_run)}")
            try:
                record = run_one(
                    item,
                    args=args,
                    batch_root=batch_root,
                    run_root=run_root,
                    eval_root=eval_root,
                    cache_root=cache_root,
                    batch_log=batch_log,
                )
            except BaseException as error:
                atomic_json(
                    batch_root / "PROGRESS.json",
                    {
                        "batch_id": args.batch_id,
                        "status": "failed",
                        "completed": len(results),
                        "total": len(items),
                        "current": item,
                        "error": repr(error),
                        "updated_at": now(),
                    },
                )
                log(batch_log, f"{run_id}: FAILED; stopping batch: {error!r}")
                raise
            key = condition_from_record(record)
            if key is None:
                raise RuntimeError(f"new result cannot be assigned a condition key: {record}")
            results_by_key[key] = record
            results = list(results_by_key.values())
            atomic_json(batch_root / "RESULTS.json", {"batch_id": args.batch_id, "created_at": now(), "runs": results})
            atomic_json(
                batch_root / "PROGRESS.json",
                {
                    "batch_id": args.batch_id,
                    "status": "running",
                    "completed": len(results),
                    "total": len(items),
                    "current": item,
                    "updated_at": now(),
                    "disk": disk_status(cache_root),
                },
            )

        combined = list(existing_records)
        combined_by_key = {condition_from_record(record): record for record in combined if condition_from_record(record)}
        for record in results:
            key = condition_from_record(record)
            if key is not None:
                combined_by_key[key] = record
        combined = list(combined_by_key.values())
        atomic_json(batch_root / "MERGED_RESULTS.json", {"locked_test_used": False, "runs": combined})
        atomic_json(batch_root / "RANKING.json", {"locked_test_used": False, "ranked_runs": rank(results)})
        atomic_json(batch_root / "MERGED_RANKING.json", {"locked_test_used": False, "ranked_runs": rank(combined)})
        atomic_json(
            batch_root / "PROGRESS.json",
            {
                "batch_id": args.batch_id,
                "status": "completed",
                "completed": len(results),
                "total": len(items),
                "skipped_existing": len(to_skip),
                "updated_at": now(),
                "disk": disk_status(cache_root),
            },
        )
        log(batch_log, f"batch {args.batch_id}: COMPLETED; merged_conditions={len(combined)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
