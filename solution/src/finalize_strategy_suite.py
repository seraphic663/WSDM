#!/usr/bin/env python3
"""Validate a completed strategy suite and write its immutable final manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from datetime import datetime
from pathlib import Path


EXPECTED_TRAINING = {
    "b_original_g6_lr1e6",
    "b_cleaned_g6_lr1e6",
    "shared_weighted_g6_lr1e6",
    "c_unit_g6_lr1e6",
    "e_group10_lr1e6",
    "f_group6_lr5e7",
    "f_group6_lr3e7",
}
EXPECTED_EVALUATIONS = {
    "a_m00_query_isolated_diag32",
    "d_m00_shielded_existing_neg16",
    "g_tail_avg2_diag32",
    "g_tail_avg3_diag32",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_and_classify(cpu: dict, validation: dict, gpu: dict) -> dict[str, dict[str, str]]:
    training = {item["name"]: item for item in gpu["training"]}
    evaluations = {item["name"]: item for item in gpu["evaluations"]}
    if set(training) != EXPECTED_TRAINING or set(evaluations) != EXPECTED_EVALUATIONS:
        raise ValueError("strategy matrix does not match the expected A-H suite")
    for item in training.values():
        if item["status"] != "PASS" or item["return_code"] != 0 or item["max_step"] != 10:
            raise ValueError(f"incomplete short training: {item['name']}")
        if any(item[key] for key in ("traceback_count", "oom_count", "nccl_error_count")):
            raise ValueError(f"training errors recorded: {item['name']}")
    for item in evaluations.values():
        if item["status"] != "PASS" or item["return_code"] != 0:
            raise ValueError(f"incomplete evaluation: {item['name']}")
    if gpu.get("simulation_only") is not True:
        raise ValueError("GPU result is not marked simulation_only")
    if cpu["outputs"]["a_train64"]["rows"] != 64 or cpu["outputs"]["a_diag32"]["rows"] != 32:
        raise ValueError("A split sample counts differ")
    if cpu["official_corpus_present"] is not False:
        raise ValueError("D classification assumes the recorded absent official corpus")
    if validation["H_sampling"]["all_rows_expose_every_proven_positive"] is not True:
        raise ValueError("H sampling validation failed")

    return {
        "A": {"status": "PASS", "reason": "64/32 deterministic query-disjoint samples and overlap audit completed; task mapping remains unavailable."},
        "B": {"status": "PASS", "reason": "Line-aligned raw/cleaned data loaded and both matched 10-step runs completed."},
        "C": {"status": "PASS", "reason": "Official trajectory fields, weighted-loss unit check, and matched 10-step runs completed; benefit is not established."},
        "D": {"status": "PARTIAL", "reason": "Official offline corpus is absent; allowed existing-negative ranking with global positive shielding completed."},
        "E": {"status": "PASS", "reason": "Matched group6/group10 10-step runs completed and resource measurements were recorded."},
        "F": {"status": "PASS", "reason": "Matched 1e-6/5e-7/3e-7 10-step runs completed; the short run cannot select a winner."},
        "G": {"status": "PASS", "reason": "Compatible two/three-point averages passed key/shape checks, fresh-process load, and isolated diagnostics."},
        "H": {"status": "PASS", "reason": "Provable official multi-positive groups and sampling simulation completed; no joint multi-positive objective was claimed."},
        "I": {"status": "PASS", "reason": "Unified evidence manifest and comparison report completed."},
    }


def command_for_evaluation(item: dict) -> str:
    return shlex.join([
        ".venv/bin/python",
        "solution/src/evaluate_qwen3_pairs.py",
        "--model", item["model"],
        "--input", item["data"],
        "--output", f"ccir/outputs/simulations/strategy_suite_20260718/eval/{item['name']}.json",
        "--batch-size", "16", "--max-length", "512", "--device", "cuda:0",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--freeze-manifest", required=True, type=Path)
    parser.add_argument("--formal-run", required=True, type=Path)
    parser.add_argument("--git-head", required=True)
    args = parser.parse_args()

    output = args.output_root / "final_manifest.json"
    if output.exists():
        raise FileExistsError(output)
    initial = load_json(args.output_root / "manifest.json")
    cpu = load_json(args.output_root / "cpu_report.json")
    validation = load_json(args.output_root / "cpu_validation.json")
    gpu = load_json(args.output_root / "gpu_results.json")
    freeze = load_json(args.freeze_manifest)
    config = load_json(args.config)
    outcomes = validate_and_classify(cpu, validation, gpu)

    training = []
    for item in gpu["training"]:
        log = args.output_root / "logs" / f"strategy_suite_20260718_{item['name']}.log"
        command = next((line.removeprefix("command: ") for line in log.read_text(errors="replace").splitlines() if line.startswith("command: ")), None)
        if command is None:
            raise ValueError(f"missing training command: {log}")
        training.append({
            "name": item["name"], "simulation_only": True,
            "started_at": item["started_at"], "completed_at": item["completed_at"],
            "input_path": item["data"], "input_sha256": item["data_sha256"],
            "base_model_path": "/root/data/LRAT/ccir/models/Qwen3-Embedding-0.6B",
            "base_model_sha256": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
            "command": command, "result": item,
        })

    evaluations = [{
        "name": item["name"], "simulation_only": True,
        "started_at": item["started_at"], "completed_at": item["completed_at"],
        "input_path": item["data"], "input_sha256": item["data_sha256"],
        "model_path": item["model"], "model_sha256": item["model_sha256"],
        "command": command_for_evaluation(item), "result": item,
    } for item in gpu["evaluations"]]

    payload = {
        "suite_id": config["suite_id"], "simulation_only": True,
        "suite_git_head": initial["git_head"], "finalizer_git_head": args.git_head,
        "started_at": initial["started_at"], "completed_at": datetime.now().astimezone().isoformat(),
        "commands": {
            "prepare": initial["command"],
            "cpu_validate": f".venv/bin/python solution/src/validate_strategy_cpu.py --data-root {args.data_root} --output-root {args.output_root}",
            "gpu_suite": f".venv/bin/python solution/src/run_strategy_gpu_suite.py --config {args.config}",
            "finalize": " ".join(shlex.quote(value) for value in [".venv/bin/python", "solution/src/finalize_strategy_suite.py", "--data-root", str(args.data_root), "--output-root", str(args.output_root), "--config", str(args.config), "--freeze-manifest", str(args.freeze_manifest), "--formal-run", str(args.formal_run), "--git-head", args.git_head]),
        },
        "inputs": {
            **initial["inputs"],
            "config": {"path": str(args.config.resolve()), "sha256": sha256(args.config)},
            "formal_model": {"path": str((args.formal_run / "model.safetensors").resolve()), "sha256": sha256(args.formal_run / "model.safetensors")},
            "freeze_manifest": {"path": str(args.freeze_manifest.resolve()), "sha256": sha256(args.freeze_manifest), "model_sha256": freeze["model_sha256"]},
        },
        "phases": {"training": training, "evaluations": evaluations},
        "outcomes": outcomes,
        "evidence": {
            name: {"path": str((args.output_root / name).resolve()), "sha256": sha256(args.output_root / name)}
            for name in ("manifest.json", "cpu_report.json", "cpu_validation.json", "provenance.json", "gpu_results.json", "freeze_recovery.log")
        },
        "constraints": config["constraints"],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", errors="strict")
    print(json.dumps({"path": str(output), "sha256": sha256(output), "outcomes": outcomes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
