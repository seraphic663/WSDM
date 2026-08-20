#!/bin/bash
# Master Experiment Runner for M08–M13
#
# Runs all six experiment directions in sequence.
# Each experiment:
#   1. Prepares data (python prepare.py)
#   2. Trains 500-step short run
#   3. Evaluates on dev1500
#   4. Reports metrics vs baseline
#   5. If >=2/4 metrics improve, optionally proceeds to full epoch
#
# Usage: bash run_all.sh [--full-epoch-if-promising]

set -euo pipefail

cd /root/data/LRAT
source .venv/bin/activate

SCRIPT_DIR="solution/experiments"
EVAL_SCRIPT="solution/src/evaluate_qwen3_pairs.py"
DEV_DATA="ccir/data/experiments/early_stop_v1/dev.jsonl"
BASELINE_REF="ccir/outputs/experiments/trajectory_quality_v1_20260724/gpu_results.json"

echo "============================================"
echo "  M08–M13 Experiment Suite Runner"
echo "  Started: $(date -Is)"
echo "============================================"

run_experiment() {
    local exp_id="$1"
    local data_path="$2"
    local extra_args="${3:-}"

    echo ""
    echo "===== ${exp_id} ====="

    # Step 1: Prepare data
    echo "[${exp_id}] Preparing data..."
    if [ -f "${SCRIPT_DIR}/${exp_id}/prepare.py" ]; then
        python "${SCRIPT_DIR}/${exp_id}/prepare.py"
    else
        echo "[${exp_id}] No prepare.py, using existing data: ${data_path}"
    fi

    # Step 2: 500-step training
    echo "[${exp_id}] Training 500 steps..."
    bash "${SCRIPT_DIR}/train.sh" "${exp_id}_500" "${data_path}" --short ${extra_args}

    # Step 3: Evaluate
    echo "[${exp_id}] Evaluating on dev1500..."
    local checkpoint_dir="ccir/outputs/experiments/${exp_id}_500/checkpoints"
    local eval_output="ccir/outputs/experiments/${exp_id}_500/eval_dev1500.json"

    python "${EVAL_SCRIPT}" \
        --model_path "${checkpoint_dir}" \
        --eval_data "${DEV_DATA}" \
        --output_path "${eval_output}" 2>&1 || {
        echo "[${exp_id}] Evaluation failed, skipping"
        return 1
    }

    # Step 4: Quick metrics check
    echo "[${exp_id}] Results:"
    python - << PYEOF
import json
try:
    with open("${eval_output}") as f:
        data = json.load(f)
    r1 = data.get("recall@1", data.get("Recall@1", 0))
    r5 = data.get("recall@5", data.get("Recall@5", 0))
    r10 = data.get("recall@10", data.get("Recall@10", 0))
    mrr = data.get("mrr", data.get("MRR", 0))
    print(f"  R@1={r1:.4f} R@5={r5:.4f} R@10={r10:.4f} MRR={mrr:.6f}")
except Exception as e:
    print(f"  Error reading results: {e}")
PYEOF

    echo "[${exp_id}] Done."
}

# ===== Run all experiments =====

# M08: Trajectory weighting
run_experiment "m08_trajectory_weighting" \
    "ccir/data/experiments/m08_trajectory_weighting/train_m08.jsonl"

# M09: Multi-positive (needs loader patch)
export M09_MULTI_POSITIVE=1
source "${SCRIPT_DIR}/m09_multipositive/patch_loader.sh"
run_experiment "m09_multipositive" \
    "ccir/data/experiments/m09_multipositive/train_m09.jsonl"

# M10: LR/Temp scan (handled by its own runner)
echo ""
echo "===== M10: LR/Temp Scan ====="
bash "${SCRIPT_DIR}/m10_lr_temp_scan/run_scan.sh"

# M11: Hard negative mining
run_experiment "m11_hard_negatives" \
    "ccir/data/experiments/m11_hard_negatives/train_m11.jsonl"

# M12: Sequence weighting
run_experiment "m12_sequence_weight" \
    "ccir/data/experiments/m12_sequence_weight/train_m12.jsonl"

# M13: Reasoning-consistency weighting
run_experiment "m13_reasoning_consistency" \
    "ccir/data/experiments/m13_reasoning_consistency/train_m13.jsonl"

echo ""
echo "============================================"
echo "  Experiment Suite Complete"
echo "  Finished: $(date -Is)"
echo "============================================"
