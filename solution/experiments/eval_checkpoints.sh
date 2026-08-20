#!/bin/bash
# Evaluate all backed-up checkpoints on dev1500
# Finds the optimal checkpoint (not just the final one)
# Usage: bash eval_checkpoints.sh <experiment_id>
# Example: bash eval_checkpoints.sh m12_full

set -euo pipefail
EXP="${1:?Usage: eval_checkpoints.sh <experiment_id>}"
cd /root/data/LRAT
source .venv/bin/activate
export PYTHONPATH=/root/data/LRAT/FlagEmbedding:${PYTHONPATH:-}

CKPT_DIR="ccir/outputs/experiments/${EXP}/checkpoints"
BACKUP_DIR="ccir/outputs/experiments/${EXP}/checkpoint_backups"
EVAL_DIR="ccir/outputs/experiments/${EXP}/eval"
DEV_DATA="ccir/data/experiments/early_stop_v1/dev.jsonl"
EVAL_SCRIPT="solution/src/evaluate_qwen3_pairs.py"
RESULTS="${EVAL_DIR}/all_checkpoints_results.jsonl"

mkdir -p "${EVAL_DIR}"
echo "" > "${RESULTS}"

# Collect all checkpoints (both active and backed up)
ALL_CKPTS=()
for d in "${CKPT_DIR}"/checkpoint-* "${BACKUP_DIR}"/checkpoint-*; do
    [ -d "$d" ] || continue
    [ -f "$d/model.safetensors" ] || continue
    name=$(basename "$d")
    # Avoid duplicates (backup may have same checkpoint as active)
    if [[ ! " ${ALL_CKPTS[*]} " =~ " ${name} " ]]; then
        ALL_CKPTS+=("$name")
    fi
done

# Sort by step number
IFS=$'\n' ALL_CKPTS=($(sort -t'-' -k2 -n <<< "${ALL_CKPTS[*]}")); unset IFS

echo "=== Evaluating ${#ALL_CKPTS[@]} checkpoints for ${EXP} ==="
echo "Checkpoints: ${ALL_CKPTS[*]}"

BEST_MRR=0
BEST_CKPT=""

for ckpt_name in "${ALL_CKPTS[@]}"; do
    # Find the actual path
    if [ -d "${CKPT_DIR}/${ckpt_name}" ]; then
        ckpt_path="${CKPT_DIR}/${ckpt_name}"
    else
        ckpt_path="${BACKUP_DIR}/${ckpt_name}"
    fi

    step=$(echo "$ckpt_name" | grep -oP '\d+')
    eval_out="${EVAL_DIR}/eval_${ckpt_name}.json"

    # Skip if already evaluated
    if [ -f "$eval_out" ]; then
        echo "[${ckpt_name}] Already evaluated, loading..."
    else
        echo "[${ckpt_name}] Evaluating (step ${step})..."
        .venv/bin/python "${EVAL_SCRIPT}" \
            --model "${ckpt_path}" \
            --input "${DEV_DATA}" \
            --output "${eval_out}" \
            --device cuda:0 2>/dev/null || {
            echo "  WARNING: Eval failed for ${ckpt_name}, skipping"
            continue
        }
    fi

    # Extract metrics
    mrr=$(python3 -c "
import json
d=json.load(open('${eval_out}'))
m=d.get('metrics',{})
print(m.get('mrr',0))
" 2>/dev/null)
    r1=$(python3 -c "
import json
d=json.load(open('${eval_out}'))
m=d.get('metrics',{})
print(m.get('recall_at_1',0))
" 2>/dev/null)

    echo "  R@1=${r1} MRR=${mrr}"

    # Track best
    if python3 -c "exit(0 if ${mrr} > ${BEST_MRR} else 1)" 2>/dev/null; then
        BEST_MRR=$mrr
        BEST_CKPT=$ckpt_name
    fi

    # Save to results
    python3 -c "
import json
r={'checkpoint':'${ckpt_name}','step':${step},'R1':${r1},'MRR':${mrr}}
with open('${RESULTS}','a') as f: f.write(json.dumps(r)+'\n')
"
done

echo ""
echo "=== BEST CHECKPOINT: ${BEST_CKPT} (MRR=${BEST_MRR}) ==="

# Copy best checkpoint to a dedicated directory
if [ -n "${BEST_CKPT}" ]; then
    BEST_DIR="ccir/outputs/experiments/${EXP}/best_checkpoint"
    rm -rf "${BEST_DIR}"
    if [ -d "${CKPT_DIR}/${BEST_CKPT}" ]; then
        cp -r "${CKPT_DIR}/${BEST_CKPT}" "${BEST_DIR}"
    else
        cp -r "${BACKUP_DIR}/${BEST_CKPT}" "${BEST_DIR}"
    fi
    echo "Best checkpoint copied to: ${BEST_DIR}"
fi

# Save summary
python3 -c "
import json
results=[]
with open('${RESULTS}') as f:
    for l in f:
        if l.strip(): results.append(json.loads(l))
results.sort(key=lambda r: r['MRR'], reverse=True)
print('\n=== RANKED RESULTS ===')
print(f'{\"Rank\":<6} {\"Checkpoint\":<20} {\"Step\":<8} {\"R@1\":<10} {\"MRR\":<12}')
print('-'*56)
for i,r in enumerate(results):
    marker = ' <-- BEST' if i==0 else ''
    print(f'{i+1:<6} {r[\"checkpoint\"]:<20} {r[\"step\"]:<8} {r[\"R1\"]:<10.4f} {r[\"MRR\"]:<12.6f}{marker}')
" | tee "${EVAL_DIR}/checkpoint_ranking.txt"

echo "Results saved to: ${EVAL_DIR}/"
