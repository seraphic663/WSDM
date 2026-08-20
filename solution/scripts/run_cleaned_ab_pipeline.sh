#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
experiment_id="${EXPERIMENT_ID:-cleaned_ab_queryholdout_20260719}"
experiment_root="${EXPERIMENT_ROOT:-${project_root}/ccir/outputs/experiments/${experiment_id}}"
data_root="${DATA_ROOT:-${project_root}/ccir/data/experiments/${experiment_id}}"
raw_source="${RAW_SOURCE:-${project_root}/ccir/data/raw/LRAT-training-pairs.jsonl}"
cleaned_source="${CLEANED_SOURCE:-${project_root}/ccir/data/processed/LRAT-training-pairs-query-cleaned.jsonl}"
base_model="${BASE_MODEL:-${project_root}/ccir/models/Qwen3-Embedding-0.6B}"
split_threshold="${SPLIT_THRESHOLD:-405}"
short_steps="${SHORT_STEPS:-1000}"
save_steps="${SAVE_STEPS:-500}"
seed="${SEED:-20260719}"
bootstrap_samples="${BOOTSTRAP_SAMPLES:-10000}"

raw_run="${RAW_RUN_ID:-qwen3_m00_raw_queryholdout1000_20260719}"
cleaned_run="${CLEANED_RUN_ID:-qwen3_m00_cleaned_queryholdout1000_20260719}"
full_run="${FULL_RUN_ID:-qwen3_m00_cleaned_full_epoch1_20260719}"
freeze_target="${FREEZE_TARGET:-${project_root}/ccir/models/Qwen3-Embedding-0.6B-LRAT-cleaned-1epoch-20260719}"

cd "${project_root}"
mkdir -p "${experiment_root}" "${project_root}/ccir/outputs/logs"
pipeline_log="${PIPELINE_LOG:-${project_root}/ccir/outputs/logs/${experiment_id}.log}"
exec > >(tee -a "${pipeline_log}") 2>&1

if [[ -e "${experiment_root}/COMPLETED" || -e "${experiment_root}/STOPPED_BY_GATE" ]]; then
  echo "Pipeline already reached a terminal state: ${experiment_root}" >&2
  exit 2
fi
config_file="${experiment_root}/pipeline_config.env"
config_payload="$({
  printf 'experiment_id=%s\n' "${experiment_id}"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'raw_source=%s\ncleaned_source=%s\n' "${raw_source}" "${cleaned_source}"
  printf 'base_model=%s\nsplit_threshold=%s\nshort_steps=%s\nsave_steps=%s\nseed=%s\n' "${base_model}" "${split_threshold}" "${short_steps}" "${save_steps}" "${seed}"
  printf 'bootstrap_samples=%s\nraw_run=%s\ncleaned_run=%s\nfull_run=%s\nfreeze_target=%s\n' "${bootstrap_samples}" "${raw_run}" "${cleaned_run}" "${full_run}" "${freeze_target}"
})"
if [[ -e "${config_file}" ]]; then
  if [[ "$(<"${config_file}")" != "${config_payload}" ]]; then
    echo "Refusing changed pipeline configuration" >&2
    diff -u "${config_file}" <(printf '%s\n' "${config_payload}") || true
    exit 2
  fi
else
  printf '%s\n' "${config_payload}" > "${config_file}"
fi

if [[ -e "${experiment_root}/RUNNING" ]]; then
  previous_pid="$(awk -F= '$1 == "pid" {print $2}' "${experiment_root}/RUNNING")"
  if [[ -n "${previous_pid}" ]] && kill -0 "${previous_pid}" 2>/dev/null; then
    echo "Pipeline already running as PID ${previous_pid}" >&2
    exit 3
  fi
  mv "${experiment_root}/RUNNING" "${experiment_root}/RUNNING.stale.$(date +%Y%m%d-%H%M%S)"
fi
if [[ -e "${experiment_root}/FAILED" ]]; then
  mv "${experiment_root}/FAILED" "${experiment_root}/FAILED.previous.$(date +%Y%m%d-%H%M%S)"
fi
printf 'started_at=%s\npid=%s\n' "$(date --iso-8601=seconds)" "$$" > "${experiment_root}/RUNNING"

terminal_written=0
on_exit() {
  rc=$?
  trap - EXIT
  rm -f "${experiment_root}/RUNNING"
  if (( terminal_written == 0 )); then
    printf 'failed_at=%s\nexit_code=%s\n' "$(date --iso-8601=seconds)" "${rc}" > "${experiment_root}/FAILED"
  fi
  exit "${rc}"
}
trap on_exit EXIT

echo "=== cleaned A/B pipeline ==="
date --iso-8601=seconds
cat "${config_file}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

if [[ ! -f "${data_root}/manifest.json" ]]; then
  if [[ -e "${data_root}" ]]; then
    echo "Incomplete data root exists; refusing overwrite: ${data_root}" >&2
    exit 2
  fi
  .venv/bin/python solution/src/prepare_cleaned_ab.py \
    --raw "${raw_source}" \
    --cleaned "${cleaned_source}" \
    --output-root "${data_root}" \
    --split-threshold "${split_threshold}"
fi

train_short_arm() {
  local run_id="$1"
  local train_data="$2"
  local port="$3"
  if [[ -f "${project_root}/ccir/outputs/checkpoints/${run_id}/COMPLETED" ]]; then
    echo "short arm already complete: ${run_id}"
    return
  fi
  RUN_ID="${run_id}" \
  MODEL_PATH="${base_model}" \
  TRAIN_DATA="${train_data}" \
  NUM_EPOCHS=1 \
  MAX_STEPS="${short_steps}" \
  SAVE_STEPS="${save_steps}" \
  SAVE_TOTAL_LIMIT=3 \
  SEED="${seed}" \
  MASTER_PORT="${port}" \
  MAX_ATTEMPTS=3 \
  RETRY_DELAY_SECONDS=60 \
    bash solution/scripts/run_qwen3_dual_supervised.sh
}

train_short_arm "${raw_run}" "${data_root}/raw_train.jsonl" 29561
train_short_arm "${cleaned_run}" "${data_root}/cleaned_train.jsonl" 29562

eval_root="${experiment_root}/eval"
mkdir -p "${eval_root}"
run_eval() {
  local label="$1"
  local model_path="$2"
  local output_path="$3"
  local gpu="$4"
  local log_path="${output_path%.json}.log"
  if [[ -s "${output_path}" ]]; then
    .venv/bin/python -c 'import json,sys; value=json.load(open(sys.argv[1])); assert value["rows"] > 0 and value["details"]' "${output_path}"
    echo "evaluation already complete: ${label}"
    return
  fi
  if [[ -e "${output_path}" ]]; then
    mv "${output_path}" "${output_path}.invalid.$(date +%Y%m%d-%H%M%S)"
  fi
  if [[ -e "${log_path}" ]]; then
    mv "${log_path}" "${log_path}.previous.$(date +%Y%m%d-%H%M%S)"
  fi
  echo "evaluation ${label} started_at=$(date --iso-8601=seconds) gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python solution/src/evaluate_qwen3_pairs.py \
    --model "${model_path}" \
    --input "${data_root}/query_heldout_diag.jsonl" \
    --output "${output_path}" \
    --batch-size 16 \
    --max-length 512 \
    --device cuda:0 > "${log_path}" 2>&1
  echo "evaluation ${label} completed_at=$(date --iso-8601=seconds)"
}

eval_pair() {
  local label="$1"
  local raw_model="$2"
  local clean_model="$3"
  run_eval "${label}-raw" "${raw_model}" "${eval_root}/${label}_raw.json" 0 &
  local raw_pid=$!
  run_eval "${label}-cleaned" "${clean_model}" "${eval_root}/${label}_cleaned.json" 1 &
  local clean_pid=$!
  local rc=0
  wait "${raw_pid}" || rc=$?
  wait "${clean_pid}" || rc=$?
  return "${rc}"
}

run_eval "m00-query-heldout" "${base_model}" "${eval_root}/m00_query_heldout.json" 0
eval_pair "checkpoint-500" \
  "${project_root}/ccir/outputs/checkpoints/${raw_run}/checkpoint-500" \
  "${project_root}/ccir/outputs/checkpoints/${cleaned_run}/checkpoint-500"
eval_pair "final-1000" \
  "${project_root}/ccir/outputs/checkpoints/${raw_run}" \
  "${project_root}/ccir/outputs/checkpoints/${cleaned_run}"

gate_report="${experiment_root}/paired_gate.json"
if [[ ! -s "${gate_report}" ]]; then
  .venv/bin/python solution/src/compare_paired_evals.py \
    --pair checkpoint-500 "${eval_root}/checkpoint-500_raw.json" "${eval_root}/checkpoint-500_cleaned.json" \
    --pair final-1000 "${eval_root}/final-1000_raw.json" "${eval_root}/final-1000_cleaned.json" \
    --bootstrap-samples "${bootstrap_samples}" \
    --seed "${seed}" \
    --output "${gate_report}"
fi
gate_passed="$(.venv/bin/python -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["gate"]["passed"]).lower())' "${gate_report}")"
if [[ "${gate_passed}" != "true" ]]; then
  printf 'stopped_at=%s\nreason=paired_gate_failed\ngate_report=%s\n' "$(date --iso-8601=seconds)" "${gate_report}" > "${experiment_root}/STOPPED_BY_GATE"
  terminal_written=1
  rm -f "${experiment_root}/RUNNING"
  echo "Full cleaned epoch was not authorized by the predeclared gate."
  exit 0
fi

if [[ ! -f "${project_root}/ccir/outputs/checkpoints/${full_run}/COMPLETED" ]]; then
  RUN_ID="${full_run}" \
  MODEL_PATH="${base_model}" \
  TRAIN_DATA="${cleaned_source}" \
  NUM_EPOCHS=1 \
  MAX_STEPS=-1 \
  SAVE_STEPS=500 \
  SAVE_TOTAL_LIMIT=3 \
  SEED="${seed}" \
  MASTER_PORT=29563 \
  MAX_ATTEMPTS=3 \
  RETRY_DELAY_SECONDS=60 \
    bash solution/scripts/run_qwen3_dual_supervised.sh
fi

full_eval="${experiment_root}/cleaned_full_epoch1_training_overlap_dev500.json"
run_eval "cleaned-full-epoch1" "${project_root}/ccir/outputs/checkpoints/${full_run}" "${full_eval}" 0

if [[ ! -e "${freeze_target}" ]]; then
  CUDA_VISIBLE_DEVICES="" .venv/bin/python solution/src/freeze_inference_model.py \
    --source "${project_root}/ccir/outputs/checkpoints/${full_run}" \
    --target "${freeze_target}" > "${experiment_root}/freeze_manifest.stdout.json"
fi
.venv/bin/python solution/src/freeze_inference_model.py --validate-only "${freeze_target}" > "${experiment_root}/frozen_load_validation.json"

printf 'completed_at=%s\ngate_report=%s\nfull_run=%s\nfreeze_target=%s\n' \
  "$(date --iso-8601=seconds)" "${gate_report}" "${full_run}" "${freeze_target}" > "${experiment_root}/COMPLETED"
terminal_written=1
rm -f "${experiment_root}/RUNNING"
echo "Pipeline completed."
