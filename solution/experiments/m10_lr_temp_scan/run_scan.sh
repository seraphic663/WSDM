#!/bin/bash
# M10: LR + Temperature Scan Runner
# Phase 1: 5 LR values × 500 steps (temp=0.02 fixed)
# Phase 2: 4 temp values × 500 steps (best LR from phase 1)
# Phase 3: Best combo → full 1 epoch

set -euo pipefail

cd /root/data/LRAT
source .venv/bin/activate

TRAIN_SCRIPT="solution/experiments/train.sh"
EVAL_SCRIPT="solution/src/evaluate_qwen3_pairs.py"
DEV_DATA="ccir/data/experiments/early_stop_v1/dev.jsonl"
TRAIN_DATA="ccir/data/experiments/early_stop_v1/train.jsonl"
SCAN_DIR="ccir/outputs/experiments/m10_lr_temp_scan"
RESULTS_FILE="${SCAN_DIR}/scan_results.jsonl"

mkdir -p "${SCAN_DIR}"
echo "" > "${RESULTS_FILE}"

LR_VALUES=(3e-7 5e-7 1e-6 2e-6 3e-6)
TEMP_VALUES=(0.01 0.02 0.03 0.05)

run_and_eval() {
    local name="$1"
    local lr="$2"
    local temp="$3"
    local steps="${4:-500}"

    echo ""
    echo "=== ${name}: LR=${lr} temp=${temp} steps=${steps} ==="

    # Train
    bash "${TRAIN_SCRIPT}" "${name}" "${TRAIN_DATA}" \
        --steps "${steps}" --lr "${lr}" --temp "${temp}"

    # Eval
    local ckpt="ccir/outputs/experiments/${name}/checkpoints"
    local eval_out="${SCAN_DIR}/eval_${name}.json"

    .venv/bin/python "${EVAL_SCRIPT}" \
        --model_path "${ckpt}" \
        --eval_data "${DEV_DATA}" \
        --output_path "${eval_out}" 2>&1 || {
        echo "Eval failed for ${name}"
        return 1
    }

    # Extract metrics
    .venv/bin/python - << PYEOF
import json
with open("${eval_out}") as f:
    data = json.load(f)
r1 = data.get("recall@1", data.get("Recall@1", 0))
r5 = data.get("recall@5", data.get("Recall@5", 0))
r10 = data.get("recall@10", data.get("Recall@10", 0))
mrr = data.get("mrr", data.get("MRR", 0))
result = {
    "name": "${name}", "lr": "${lr}", "temp": "${temp}",
    "steps": ${steps}, "R1": r1, "R5": r5, "R10": r10, "MRR": mrr
}
with open("${RESULTS_FILE}", "a") as f:
    f.write(json.dumps(result) + "\n")
print(f"  R@1={r1:.4f} R@5={r5:.4f} R@10={r10:.4f} MRR={mrr:.6f}")
PYEOF
}

# ===== Phase 1: LR Scan =====
echo "=========================================="
echo "  Phase 1: LR Scan (temp=0.02)"
echo "=========================================="

for lr in "${LR_VALUES[@]}"; do
    lr_str=$(echo "${lr}" | sed 's/e-0/e-/g' | sed 's/e-0/e-/g')
    run_and_eval "m10_phase1_lr${lr_str}" "${lr}" "0.02" 500
done

echo ""
echo "Phase 1 complete. Finding best LR..."

# Find best LR by MRR
BEST_LR=$(.venv/bin/python - << PYEOF
import json
results = []
with open("${RESULTS_FILE}") as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))
phase1 = [r for r in results if "phase1" in r["name"]]
if not phase1:
    print("1e-6")  # default
else:
    best = max(phase1, key=lambda r: r["MRR"])
    print(best["lr"])
PYEOF
)

echo "Best LR from Phase 1: ${BEST_LR}"

# ===== Phase 2: Temperature Scan =====
echo ""
echo "=========================================="
echo "  Phase 2: Temperature Scan (LR=${BEST_LR})"
echo "=========================================="

for temp in "${TEMP_VALUES[@]}"; do
    run_and_eval "m10_phase2_temp${temp}" "${BEST_LR}" "${temp}" 500
done

echo ""
echo "Phase 2 complete. Finding best combo..."

# Find best combo
BEST_COMBO=$(.venv/bin/python - << PYEOF
import json
results = []
with open("${RESULTS_FILE}") as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))
phase2 = [r for r in results if "phase2" in r["name"]]
if not phase2:
    phase2 = [r for r in results if "phase1" in r["name"]]
best = max(phase2, key=lambda r: r["MRR"])
print(f"{best['lr']} {best['temp']}")
PYEOF
)

BEST_LR_FINAL=$(echo "${BEST_COMBO}" | cut -d' ' -f1)
BEST_TEMP_FINAL=$(echo "${BEST_COMBO}" | cut -d' ' -f2)
echo "Best combo: LR=${BEST_LR_FINAL} temp=${BEST_TEMP_FINAL}"

# ===== Phase 3: Full Epoch =====
echo ""
echo "=========================================="
echo "  Phase 3: Full Epoch (LR=${BEST_LR_FINAL} temp=${BEST_TEMP_FINAL})"
echo "=========================================="

bash "${TRAIN_SCRIPT}" "m10_best_full" "${TRAIN_DATA}" \
    --lr "${BEST_LR_FINAL}" --temp "${BEST_TEMP_FINAL}"

echo ""
echo "M10 scan complete. Best: LR=${BEST_LR_FINAL} temp=${BEST_TEMP_FINAL}"
echo "All results: ${RESULTS_FILE}"
