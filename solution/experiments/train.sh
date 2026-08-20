#!/bin/bash
# LRAT Experiment Training Wrapper
# Usage: bash train.sh <experiment_id> <train_data_path> [extra_args...]
#
# Handles:
# - Full epoch training with dual-GPU torchrun
# - 500-step short training for diagnostics
# - Automatic checkpoint/resume
# - COMPLETED/FAILED marking
#
# Models use the same training infrastructure as M01/M07.

set -euo pipefail

EXPERIMENT_ID="${1:?Usage: train.sh <experiment_id> <train_data> [--short] [extra_args]}"
TRAIN_DATA="${2:?}"
shift 2 || true

cd /root/data/LRAT
source .venv/bin/activate

# Base configuration (same as M01/M07)
MODEL_PATH="ccir/models/Qwen3-Embedding-0.6B"
BASE_OUTPUT="ccir/outputs/experiments/${EXPERIMENT_ID}"
CHECKPOINT_DIR="${BASE_OUTPUT}/checkpoints"
CACHE_DIR="ccir/data/cache/${EXPERIMENT_ID}"
LOG_DIR="ccir/outputs/logs"

mkdir -p "${CHECKPOINT_DIR}" "${CACHE_DIR}" "${LOG_DIR}"

# Parse mode
MAX_STEPS=""
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --short)
            MAX_STEPS="--max_steps 500"
            shift
            ;;
        --steps)
            MAX_STEPS="--max_steps $2"
            shift 2
            ;;
        --lr)
            EXTRA_ARGS="${EXTRA_ARGS} --learning_rate $2"
            shift 2
            ;;
        --temp)
            EXTRA_ARGS="${EXTRA_ARGS} --temperature $2"
            shift 2
            ;;
        --group-size)
            EXTRA_ARGS="${EXTRA_ARGS} --train_group_size $2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/${EXPERIMENT_ID}_${TIMESTAMP}.log"

echo "=== Experiment: ${EXPERIMENT_ID} ==="
echo "Train data: ${TRAIN_DATA}"
echo "Output: ${CHECKPOINT_DIR}"
echo "Cache: ${CACHE_DIR}"
echo "Log: ${LOG_FILE}"
echo "Max steps: ${MAX_STEPS:-full epoch}"
echo "Extra args: ${EXTRA_ARGS}"

# Check if already completed
if [ -f "${CHECKPOINT_DIR}/COMPLETED" ]; then
    echo "Experiment ${EXPERIMENT_ID} already completed. Skipping."
    exit 0
fi

# Write RUNNING marker
echo "running_at=$(date -Is)" > "${CHECKPOINT_DIR}/RUNNING"

# Write run config
cat > "${CHECKPOINT_DIR}/run_config.env" << EOF
EXPERIMENT_ID="${EXPERIMENT_ID}"
TRAIN_DATA="${TRAIN_DATA}"
TRAIN_DATA_SHA="$(sha256sum ${TRAIN_DATA} | cut -d' ' -f1)"
MODEL_PATH="${MODEL_PATH}"
MODEL_SHA="$(sha256sum ${MODEL_PATH}/model.safetensors | cut -d' ' -f1)"
TIMESTAMP="${TIMESTAMP}"
MAX_STEPS="${MAX_STEPS:-full_epoch}"
EXTRA_ARGS="${EXTRA_ARGS}"
EOF

# Run training
set +e
torchrun --standalone --nproc_per_node=2 \
    -m FlagEmbedding.finetune.embedder.decoder_only.base \
    --model_name_or_path "${MODEL_PATH}" \
    --train_data "${TRAIN_DATA}" \
    --cache_path "${CACHE_DIR}" \
    --train_group_size 6 \
    --query_max_len 128 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval 'Given a web search query, retrieve relevant passages that answer the query' \
    --query_instruction_format $'Instruct: {}\nQuery:{}' \
    --knowledge_distillation False \
    --output_dir "${CHECKPOINT_DIR}" \
    --learning_rate 1e-6 \
    --bf16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method last_token \
    --normalize_embeddings True \
    --seed 20260716 \
    --data_seed 20260716 \
    --ddp_find_unused_parameters False \
    --report_to none \
    ${MAX_STEPS} \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_FILE}"

TRAIN_EXIT=$?

if [ ${TRAIN_EXIT} -eq 0 ]; then
    rm -f "${CHECKPOINT_DIR}/RUNNING"
    echo "completed_at=$(date -Is)" > "${CHECKPOINT_DIR}/COMPLETED"
    echo "train_exit_code=0" >> "${CHECKPOINT_DIR}/COMPLETED"
    echo "=== Training completed successfully ==="
else
    rm -f "${CHECKPOINT_DIR}/RUNNING"
    echo "reason=train_exit_code_${TRAIN_EXIT}" > "${CHECKPOINT_DIR}/FAILED"
    echo "=== Training FAILED with exit code ${TRAIN_EXIT} ==="
fi

exit ${TRAIN_EXIT}
