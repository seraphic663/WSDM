#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
run_id="${RUN_ID:?Set RUN_ID to a completed training run}"
eval_id="${EVAL_ID:-${run_id}_dev500}"
model_path="${MODEL_PATH:-${project_root}/ccir/outputs/checkpoints/${run_id}}"
input_path="${EVAL_INPUT:-${project_root}/ccir/data/smoke/dev.jsonl}"
output_path="${EVAL_OUTPUT:-${project_root}/ccir/outputs/eval/${eval_id}.json}"
log_path="${EVAL_LOG:-${project_root}/ccir/outputs/logs/${eval_id}.log}"
batch_size="${EVAL_BATCH_SIZE:-16}"
max_length="${EVAL_MAX_LENGTH:-512}"
cuda_device="${CUDA_VISIBLE_DEVICES:-0}"

cd "${project_root}"
if [[ ! -f "ccir/outputs/checkpoints/${run_id}/COMPLETED" ]]; then
  echo "Run is not complete: ${run_id}" >&2
  exit 2
fi
if [[ -e "ccir/outputs/checkpoints/${run_id}/RUNNING" ]]; then
  echo "Run still has a RUNNING marker: ${run_id}" >&2
  exit 2
fi
if [[ ! -s "${model_path}/model.safetensors" ]]; then
  echo "Model weight is missing or empty: ${model_path}" >&2
  exit 2
fi
if [[ -e "${output_path}" || -e "${log_path}" ]]; then
  echo "Refusing to overwrite an existing evaluation output or log" >&2
  exit 2
fi

gpu_memory="$(nvidia-smi --id="${cuda_device%%,*}" --query-gpu=memory.used --format=csv,noheader,nounits)"
gpu_util="$(nvidia-smi --id="${cuda_device%%,*}" --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
if (( gpu_memory > 2048 || gpu_util > 20 )); then
  echo "GPU ${cuda_device%%,*} is not idle enough: ${gpu_memory} MiB, ${gpu_util}%" >&2
  exit 3
fi

mkdir -p "$(dirname "${output_path}")" "$(dirname "${log_path}")"
export CUDA_VISIBLE_DEVICES="${cuda_device}"
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "model_path=${model_path}"
  sha256sum "${model_path}/model.safetensors" "${input_path}"
  .venv/bin/python solution/src/evaluate_qwen3_pairs.py \
    --model "${model_path}" \
    --input "${input_path}" \
    --output "${output_path}" \
    --batch-size "${batch_size}" \
    --max-length "${max_length}" \
    --device cuda:0
  echo "completed_at=$(date --iso-8601=seconds)"
  sha256sum "${output_path}"
} > "${log_path}" 2>&1

cat "${output_path}"
