#!/usr/bin/env python3
"""M10: Learning Rate and Temperature Joint Scan.

Model: M10
Source: M00 (Qwen3-Embedding-0.6B, SHA 0437e45c...e23fd)
Data: early_stop_v1 train 94,113 rows (no reweight modification)
Method: Grid search over LR [3e-7, 5e-7, 1e-6, 2e-6, 3e-6] × temp [0.01, 0.02, 0.03, 0.05]
        Phase 1: Fix temp=0.02, scan LR (5 configurations, 500 steps each)
        Phase 2: Fix best LR, scan temp (4 configurations, 500 steps each)
        Phase 3: Best combo → full 1 epoch

This script generates the scan configurations and writes a bash runner.
"""

import json
import itertools
from pathlib import Path
from common import (
    TRAIN_94K, DEV_1500, M00_PATH,
    sha256_str, write_run_config,
)

OUTPUT_DIR = Path("/root/data/LRAT/ccir/outputs/experiments/m10_lr_temp_scan")
TRAIN_SCRIPT = "/root/data/LRAT/solution/experiments/m10_lr_temp_scan/train_one.sh"
EVAL_SCRIPT = "/root/data/LRAT/solution/src/evaluate_qwen3_pairs.py"

LR_VALUES = [3e-7, 5e-7, 1e-6, 2e-6, 3e-6]
TEMP_VALUES = [0.01, 0.02, 0.03, 0.05]

BASE_TRAIN_CONFIG = {
    "model_name_or_path": str(M00_PATH),
    "train_data": str(TRAIN_94K),
    "train_group_size": 6,
    "query_max_len": 128,
    "passage_max_len": 512,
    "pad_to_multiple_of": 8,
    "query_instruction_for_retrieval": "Given a web search query, retrieve relevant passages that answer the query",
    "query_instruction_format": "Instruct: {}\nQuery:{}",
    "knowledge_distillation": False,
    "num_train_epochs": 1,
    "max_steps": 500,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "dataloader_drop_last": True,
    "warmup_ratio": 0.1,
    "gradient_checkpointing": True,
    "logging_steps": 1,
    "save_strategy": "steps",
    "save_steps": 500,
    "save_total_limit": 1,
    "negatives_cross_device": True,
    "sentence_pooling_method": "last_token",
    "normalize_embeddings": True,
    "seed": 20260716,
    "data_seed": 20260716,
    "ddp_find_unused_parameters": False,
    "report_to": "none",
    "bf16": True,
}


def generate_scan_plan():
    """Generate the scan plan: which configs to run and in what order."""
    # Phase 1: LR scan with fixed temp=0.02
    phase1 = []
    for lr in LR_VALUES:
        run_id = f"m10_lr{lr:.0e}_temp0.02"
        phase1.append({
            "phase": 1,
            "run_id": run_id,
            "learning_rate": lr,
            "temperature": 0.02,
            "max_steps": 500,
            "output_dir": str(OUTPUT_DIR / "checkpoints" / run_id),
            "cache_path": str(Path("/root/data/LRAT/ccir/data/cache") / run_id),
        })

    # Phase 2: Temp scan with best LR (placeholder, will be filled after phase 1)
    phase2 = []
    for temp in TEMP_VALUES:
        run_id = f"m10_lrBEST_temp{temp}"
        phase2.append({
            "phase": 2,
            "run_id": run_id,
            "learning_rate": "BEST",  # Will be replaced
            "temperature": temp,
            "max_steps": 500,
            "output_dir": str(OUTPUT_DIR / "checkpoints" / run_id),
            "cache_path": str(Path("/root/data/LRAT/ccir/data/cache") / run_id),
        })

    # Phase 3: Full epoch with best combo
    best_run_id = "m10_best_full_epoch"
    phase3 = {
        "phase": 3,
        "run_id": best_run_id,
        "learning_rate": "BEST",
        "temperature": "BEST",
        "max_steps": -1,  # Full epoch
        "output_dir": str(OUTPUT_DIR / "checkpoints" / best_run_id),
        "cache_path": str(Path("/root/data/LRAT/ccir/data/cache") / best_run_id),
    }

    return phase1, phase2, phase3


def write_runner_script(phase1, phase2, phase3):
    """Write a bash script that executes the entire scan."""
    runner_path = OUTPUT_DIR / "run_scan.sh"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "#!/bin/bash",
        "# M10 LR + Temperature Scan Runner",
        "# Generated automatically - do not edit manually",
        "",
        "set -euo pipefail",
        "",
        'cd /root/data/LRAT',
        'source .venv/bin/activate',
        "",
        "SCAN_DIR=\"" + str(OUTPUT_DIR) + "\"",
        'EVAL_SCRIPT="' + EVAL_SCRIPT + '"',
        'DEV_DATA="' + str(DEV_1500) + '"',
        'BASE_MODEL="' + str(M00_PATH) + '"',
        "",
        "# Phase 1: LR scan (fixed temp=0.02)",
        "echo '=== PHASE 1: LR Scan ==='",
        "BEST_LR=''",
        "BEST_MRR=0",
        "",
    ]

    for cfg in phase1:
        lines.append(f"echo '--- {cfg['run_id']} ---'")
        lines.append(f"torchrun --standalone --nproc_per_node=2 \\")
        lines.append(f"  -m FlagEmbedding.finetune.embedder.decoder_only.base \\")
        lines.append(f"  --model_name_or_path {BASE_TRAIN_CONFIG['model_name_or_path']} \\")
        lines.append(f"  --train_data {BASE_TRAIN_CONFIG['train_data']} \\")
        lines.append(f"  --cache_path {cfg['cache_path']} \\")
        lines.append(f"  --train_group_size {BASE_TRAIN_CONFIG['train_group_size']} \\")
        lines.append(f"  --query_max_len {BASE_TRAIN_CONFIG['query_max_len']} \\")
        lines.append(f"  --passage_max_len {BASE_TRAIN_CONFIG['passage_max_len']} \\")
        lines.append(f"  --output_dir {cfg['output_dir']} \\")
        lines.append(f"  --learning_rate {cfg['learning_rate']} \\")
        lines.append(f"  --temperature {cfg['temperature']} \\")
        lines.append(f"  --max_steps {cfg['max_steps']} \\")
        lines.append(f"  --per_device_train_batch_size {BASE_TRAIN_CONFIG['per_device_train_batch_size']} \\")
        lines.append(f"  --gradient_accumulation_steps {BASE_TRAIN_CONFIG['gradient_accumulation_steps']} \\")
        lines.append(f"  --bf16 --negatives_cross_device \\")
        lines.append(f"  --sentence_pooling_method {BASE_TRAIN_CONFIG['sentence_pooling_method']} \\")
        lines.append(f"  --normalize_embeddings True \\")
        lines.append(f"  --seed {BASE_TRAIN_CONFIG['seed']} --data_seed {BASE_TRAIN_CONFIG['data_seed']} \\")
        lines.append(f"  --logging_steps 10 --save_strategy steps --save_steps 500 \\")
        lines.append(f"  --warmup_ratio 0.1 --ddp_find_unused_parameters False --report_to none")
        lines.append(f"")
        # Eval
        lines.append(f"echo 'Eval {cfg['run_id']}...'")
        lines.append(f"$VENV_PYTHON $EVAL_SCRIPT \\")
        lines.append(f"  --model_path {cfg['output_dir']} \\")
        lines.append(f"  --eval_data $DEV_DATA \\")
        lines.append(f"  --output_path $SCAN_DIR/eval_{cfg['run_id']}.json")
        lines.append(f"")

    lines.extend([
        "",
        "# Phase 2: Temp scan with best LR",
        "echo '=== PHASE 2: Temperature Scan ==='",
        "# (Best LR determined from phase 1 results)",
        "# Run python select_best.py to pick the best LR, then run phase 2",
        "",
        "# Phase 3: Full epoch with best combo",
        "echo '=== PHASE 3: Best Combo Full Epoch ==='",
        "# Run with best LR and temp, max_steps removed for full epoch",
        "",
        "echo 'Scan complete. Results in: ' $SCAN_DIR",
    ])

    with open(runner_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Make executable
    import os
    os.chmod(runner_path, 0o755)
    print(f"Runner script: {runner_path}")


def main():
    print("M10: LR + Temperature Scan Plan")
    print("=" * 60)

    phase1, phase2, phase3 = generate_scan_plan()

    print(f"Phase 1: {len(phase1)} LR configs × 500 steps")
    for cfg in phase1:
        print(f"  {cfg['run_id']}: LR={cfg['learning_rate']}, temp={cfg['temperature']}")

    print(f"Phase 2: {len(phase2)} temp configs × 500 steps (best LR TBD)")
    print(f"Phase 3: 1 full epoch with best combo")

    # Save scan plan
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = {
        "model_id": "M10",
        "base_model": "M00",
        "base_model_sha": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
        "method": "LR and temperature grid search",
        "lr_values": LR_VALUES,
        "temp_values": TEMP_VALUES,
        "phase1": [{k: str(v) for k, v in cfg.items()} for cfg in phase1],
        "phase2": [{k: str(v) for k, v in cfg.items()} for cfg in phase2],
    }
    with open(OUTPUT_DIR / "scan_plan.json", "w") as f:
        json.dump(plan, f, indent=2, default=str)

    write_runner_script(phase1, phase2, phase3)


if __name__ == "__main__":
    main()
