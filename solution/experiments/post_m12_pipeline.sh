#!/bin/bash
# Post-M12 Automation Pipeline
# Runs automatically after M12 full epoch completes
# Handles: M10 remaining evals → M12 checkpoint eval → M13 → M09
set -euo pipefail

cd /root/data/LRAT
source .venv/bin/activate
export PYTHONPATH=/root/data/LRAT/FlagEmbedding:${PYTHONPATH:-}
DEV="ccir/data/experiments/early_stop_v1/dev.jsonl"
EVAL_SCRIPT="solution/src/evaluate_qwen3_pairs.py"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Post-M12 Pipeline Started ==="

# Step 1: Complete M10 evaluations (LR=1e-6, 2e-6)
log "Step 1: M10 remaining evals"
for lr in 1e-6 2e-6; do
    name="m10_p1_lr${lr}"
    ckpt="ccir/outputs/experiments/${name}/checkpoints"
    out="ccir/outputs/experiments/m10_lr_temp_scan/eval_${name}.json"
    if [ -f "$out" ]; then
        log "  ${name}: already done"
    else
        log "  ${name}: evaluating..."
        .venv/bin/python "$EVAL_SCRIPT" --model "$ckpt" --input "$DEV" --output "$out" --device cuda:0 2>/dev/null
        mrr=$(python3 -c "import json;print(json.load(open('$out'))['metrics']['mrr'])" 2>/dev/null)
        r1=$(python3 -c "import json;print(json.load(open('$out'))['metrics']['recall_at_1'])" 2>/dev/null)
        log "  ${name}: R@1=${r1} MRR=${mrr}"
    fi
done

# Print M10 summary
log "M10 Summary:"
python3 -c "
import json,os
SD='ccir/outputs/experiments/m10_lr_temp_scan'
for lr in ['3e-7','5e-7','1e-6','2e-6']:
    p=os.path.join(SD,f'eval_m10_p1_lr{lr}.json')
    if os.path.exists(p):
        m=json.load(open(p))['metrics']
        print(f'  LR={lr}: R@1={m[\"recall_at_1\"]:.4f} MRR={m[\"mrr\"]:.6f}')
"

# Step 2: Evaluate all M12 checkpoints
log "Step 2: M12 checkpoint evaluation"
bash solution/experiments/eval_checkpoints.sh m12_full

# Step 3: M13 500-step
log "Step 3: M13 500-step"
bash solution/experiments/train.sh m13_500 ccir/data/experiments/m13_reasoning_consistency/train_m13.jsonl --short
.venv/bin/python "$EVAL_SCRIPT" --model ccir/outputs/experiments/m13_500/checkpoints --input "$DEV" --output /tmp/eval_m13_500.json --device cuda:0 2>/dev/null
mrr13=$(python3 -c "import json;print(json.load(open('/tmp/eval_m13_500.json'))['metrics']['mrr'])" 2>/dev/null)
log "  M13 500-step MRR: ${mrr13}"

# Step 4: M09 500-step
log "Step 4: M09 500-step"
bash solution/experiments/train.sh m09_500 ccir/data/experiments/m09_multipositive/train_m09_expanded.jsonl --short
.venv/bin/python "$EVAL_SCRIPT" --model ccir/outputs/experiments/m09_500/checkpoints --input "$DEV" --output /tmp/eval_m09_500.json --device cuda:0 2>/dev/null
mrr09=$(python3 -c "import json;print(json.load(open('/tmp/eval_m09_500.json'))['metrics']['mrr'])" 2>/dev/null)
log "  M09 500-step MRR: ${mrr09}"

# Step 5: Print final comparison
log "=== FINAL RESULTS ==="
echo ""
echo "Model           | R@1    | MRR"
echo "----------------|--------|--------"
for exp in m10_p1_lr5e-7 m12_full m13_500 m09_500; do
    case $exp in
        m10_p1_lr5e-7) label="M10 LR=5e-7"; eval_file="ccir/outputs/experiments/m10_lr_temp_scan/eval_m10_p1_lr5e-7.json" ;;
        m12_full) label="M12 best ckpt"; eval_file=$(ls ccir/outputs/experiments/m12_full/eval/eval_checkpoint-*.json 2>/dev/null | head -1) ;;
        m13_500) label="M13 500-step"; eval_file="/tmp/eval_m13_500.json" ;;
        m09_500) label="M09 500-step"; eval_file="/tmp/eval_m09_500.json" ;;
    esac
    if [ -f "$eval_file" ]; then
        r1=$(python3 -c "import json;print(json.load(open('$eval_file'))['metrics']['recall_at_1'])" 2>/dev/null)
        mrr=$(python3 -c "import json;print(json.load(open('$eval_file'))['metrics']['mrr'])" 2>/dev/null)
        printf "%-16s | %.4f | %.6f\n" "$label" "$r1" "$mrr"
    fi
done

log "=== Pipeline Complete ==="
