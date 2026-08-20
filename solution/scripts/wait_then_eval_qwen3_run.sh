#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
run_id="${RUN_ID:?Set RUN_ID to the training run to watch}"
poll_seconds="${POLL_SECONDS:-60}"
wait_timeout_seconds="${WAIT_TIMEOUT_SECONDS:-64800}"
gpu_wait_timeout_seconds="${GPU_WAIT_TIMEOUT_SECONDS:-3600}"
eval_script="${EVAL_SCRIPT:-${project_root}/solution/scripts/eval_completed_qwen3_run.sh}"
run_dir="${project_root}/ccir/outputs/checkpoints/${run_id}"

for value_name in poll_seconds wait_timeout_seconds gpu_wait_timeout_seconds; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "${value_name} must be a non-negative integer" >&2
    exit 2
  fi
done
if (( poll_seconds == 0 )); then
  echo "poll_seconds must be positive" >&2
  exit 2
fi
if [[ ! -d "${run_dir}" || ! -f "${eval_script}" ]]; then
  echo "Run directory or evaluation script is missing" >&2
  exit 2
fi

started_epoch="$(date +%s)"
echo "watch_started_at=$(date --iso-8601=seconds) run_id=${run_id}"
while [[ ! -f "${run_dir}/COMPLETED" ]]; do
  now="$(date +%s)"
  if (( now - started_epoch >= wait_timeout_seconds )); then
    echo "Timed out waiting for COMPLETED: ${run_id}" >&2
    exit 124
  fi
  sleep "${poll_seconds}"
done
echo "training_completed_at=$(date --iso-8601=seconds)"

gpu_started_epoch="$(date +%s)"
while true; do
  gpu_memory="$(nvidia-smi --id="${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=memory.used --format=csv,noheader,nounits)"
  gpu_util="$(nvidia-smi --id="${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
  if (( gpu_memory <= 2048 && gpu_util <= 20 )); then
    break
  fi
  now="$(date +%s)"
  if (( now - gpu_started_epoch >= gpu_wait_timeout_seconds )); then
    echo "Timed out waiting for an idle evaluation GPU" >&2
    exit 124
  fi
  sleep "${poll_seconds}"
done

exec env \
  RUN_ID="${run_id}" \
  EVAL_ID="${EVAL_ID:-${run_id}_dev500}" \
  MODEL_PATH="${MODEL_PATH:-${run_dir}}" \
  EVAL_INPUT="${EVAL_INPUT:-${project_root}/ccir/data/smoke/dev.jsonl}" \
  EVAL_OUTPUT="${EVAL_OUTPUT:-${project_root}/ccir/outputs/eval/${run_id}_dev500.json}" \
  EVAL_LOG="${EVAL_LOG:-${project_root}/ccir/outputs/logs/${run_id}_dev500.log}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  bash "${eval_script}"
