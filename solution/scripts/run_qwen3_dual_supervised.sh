#!/usr/bin/env bash
set -uo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
run_id="${RUN_ID:?Set RUN_ID to the same unique name used by the trainer}"
train_script="${TRAIN_SCRIPT:-${project_root}/solution/scripts/run_qwen3_dual_train.sh}"
max_attempts="${MAX_ATTEMPTS:-3}"
retry_delay_seconds="${RETRY_DELAY_SECONDS:-60}"

if ! [[ "${max_attempts}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_ATTEMPTS must be a positive integer" >&2
  exit 2
fi
if ! [[ "${retry_delay_seconds}" =~ ^[0-9]+$ ]]; then
  echo "RETRY_DELAY_SECONDS must be a non-negative integer" >&2
  exit 2
fi
if [[ ! -f "${train_script}" ]]; then
  echo "Training script not found: ${train_script}" >&2
  exit 2
fi

cd "${project_root}"
output_dir="${OUTPUT_DIR:-${project_root}/ccir/outputs/checkpoints/${run_id}}"

echo "=== supervised dual-GPU training ==="
echo "started_at=$(date --iso-8601=seconds)"
echo "run_id=${run_id}"
echo "max_attempts=${max_attempts}"
echo "retry_delay_seconds=${retry_delay_seconds}"

for ((attempt = 1; attempt <= max_attempts; attempt++)); do
  echo "attempt=${attempt}/${max_attempts} started_at=$(date --iso-8601=seconds)"
  if bash "${train_script}"; then
    if [[ ! -f "${output_dir}/COMPLETED" ]]; then
      echo "Trainer returned success without COMPLETED marker" >&2
      exit 4
    fi
    echo "completed_at=$(date --iso-8601=seconds) attempts_used=${attempt}"
    exit 0
  else
    rc=$?
  fi

  echo "attempt=${attempt}/${max_attempts} failed_at=$(date --iso-8601=seconds) exit_code=${rc}" >&2
  if (( attempt == max_attempts )); then
    echo "Retry limit reached; preserving FAILED marker, logs, and checkpoints" >&2
    exit "${rc}"
  fi
  echo "Retrying the identical configuration after ${retry_delay_seconds} seconds; the trainer will select the newest complete checkpoint"
  sleep "${retry_delay_seconds}"
done
