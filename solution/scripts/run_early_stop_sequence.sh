#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
run_id="${RUN_ID:?Set RUN_ID to a unique experiment name}"
model_path="${MODEL_PATH:-${project_root}/ccir/models/Qwen3-Embedding-0.6B}"
train_data="${TRAIN_DATA:?Set TRAIN_DATA to the training JSONL}"
dev_data="${DEV_DATA:?Set DEV_DATA to the dev JSONL}"
baseline_eval="${BASELINE_EVAL:?Set BASELINE_EVAL to the M00 dev result}"
output_dir="${OUTPUT_DIR:-${project_root}/ccir/outputs/checkpoints/${run_id}}"
eval_dir="${EVAL_DIR:-${project_root}/ccir/outputs/eval/${run_id}}"
sequence_log="${SEQUENCE_LOG:-${project_root}/ccir/outputs/logs/${run_id}_sequence.log}"
max_steps="${MAX_STEPS:-11764}"
eval_interval="${EVAL_INTERVAL:-1000}"
expected_rows="${EXPECTED_ROWS:-1500}"
min_steps="${MIN_STEPS:-2000}"
min_delta="${MIN_DELTA:-0.001}"
patience="${PATIENCE:-2}"
guardrail_tolerance="${GUARDRAIL_TOLERANCE:-0.005}"
eval_batch_size="${EVAL_BATCH_SIZE:-16}"
eval_max_length="${EVAL_MAX_LENGTH:-512}"
eval_device="${EVAL_DEVICE:-cuda:0}"

if ! [[ "${max_steps}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${eval_interval}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_STEPS and EVAL_INTERVAL must be positive integers" >&2
  exit 2
fi
for required in "${model_path}" "${train_data}" "${dev_data}" "${baseline_eval}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required input does not exist: ${required}" >&2
    exit 2
  fi
done

cd "${project_root}"
source "${project_root}/solution/scripts/lib/early_stop_sequence_state.sh"
mkdir -p "${eval_dir}" "$(dirname "${sequence_log}")"
exec > >(tee -a "${sequence_log}") 2>&1

sequence_running="${eval_dir}/SEQUENCE_RUNNING"
sequence_failed="${eval_dir}/SEQUENCE_FAILED"
sequence_finished="${eval_dir}/SEQUENCE_FINISHED"
if [[ -e "${sequence_finished}" ]]; then
  echo "Sequence is already finished: ${sequence_finished}" >&2
  exit 2
fi
mkdir -p "${output_dir}"
if [[ -e "${sequence_running}" ]]; then
  old_pid="$(awk -F= '$1 == "pid" {print $2}' "${sequence_running}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Refusing to overlap live sequence process ${old_pid}" >&2
    exit 3
  fi
  mv "${sequence_running}" "${sequence_running}.stale.$(date +%Y%m%d-%H%M%S)"
fi
rm -f "${sequence_failed}"
printf 'started_at=%s\npid=%s\n' "$(date --iso-8601=seconds)" "$$" > "${sequence_running}"

on_exit() {
  rc=$?
  trap - EXIT
  rm -f "${sequence_running}"
  if [[ ${rc} -ne 0 ]]; then
    printf 'failed_at=%s\nexit_code=%s\n' "$(date --iso-8601=seconds)" "${rc}" > "${sequence_failed}"
  fi
  exit "${rc}"
}
trap on_exit EXIT

targets=()
step="${eval_interval}"
while (( step < max_steps )); do
  targets+=("${step}")
  step=$((step + eval_interval))
done
targets+=("${max_steps}")

eval_specs=()
for target in "${targets[@]}"; do
  eval_file="${eval_dir}/dev_step_${target}.json"
  if [[ ! -s "${eval_file}" ]]; then
    model_for_eval="$(
      early_stop_model_path "${output_dir}" "${target}" "${max_steps}"
    )"
    if ! early_stop_model_ready "${output_dir}" "${target}" "${max_steps}"; then
      echo "=== train through optimizer step ${target}/${max_steps} ==="
      RUN_ID="${run_id}" \
      MODEL_PATH="${model_path}" \
      TRAIN_DATA="${train_data}" \
      OUTPUT_DIR="${output_dir}" \
      LOG_FILE="${project_root}/ccir/outputs/logs/${run_id}.log" \
      MAX_STEPS="${max_steps}" \
      SAVE_STEPS="${eval_interval}" \
      SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}" \
      STOP_AFTER_STEP="${target}" \
      RESUME_MODE=auto \
      COLLISION_FREE_QUERIES="${COLLISION_FREE_QUERIES:-0}" \
      PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}" \
      GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}" \
      TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-6}" \
      QUERY_MAX_LEN="${QUERY_MAX_LEN:-128}" \
      PASSAGE_MAX_LEN="${PASSAGE_MAX_LEN:-512}" \
      LEARNING_RATE="${LEARNING_RATE:-1e-6}" \
      SEED="${SEED:-20260716}" \
      CACHE_PATH="${CACHE_PATH:-${project_root}/ccir/data/cache/early_stop_v1_shared}" \
      MASTER_PORT="${MASTER_PORT:-29517}" \
      bash solution/scripts/run_qwen3_dual_train.sh
    fi
    if ! early_stop_model_ready "${output_dir}" "${target}" "${max_steps}"; then
      echo "Expected model was not produced: ${model_for_eval}" >&2
      exit 4
    fi
    echo "=== evaluate optimizer step ${target} on dev only ==="
    .venv/bin/python solution/src/evaluate_qwen3_pairs.py \
      --model "${model_for_eval}" \
      --input "${dev_data}" \
      --output "${eval_file}" \
      --batch-size "${eval_batch_size}" \
      --max-length "${eval_max_length}" \
      --device "${eval_device}"
  fi
  eval_specs+=(--eval "${target}=${eval_file}")
  selection_file="${eval_dir}/selection_through_${target}.json"
  if [[ ! -s "${selection_file}" ]]; then
    .venv/bin/python solution/src/select_early_stop_checkpoint.py \
      "${eval_specs[@]}" \
      --baseline-eval "${baseline_eval}" \
      --expected-rows "${expected_rows}" \
      --min-steps "${min_steps}" \
      --min-delta "${min_delta}" \
      --patience "${patience}" \
      --guardrail-tolerance "${guardrail_tolerance}" \
      --output "${selection_file}"
  fi
  stop_step="$(.venv/bin/python - "${selection_file}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value.get("stop_triggered_at_step") or "")
PY
)"
  if [[ -n "${stop_step}" ]]; then
    cp -n "${selection_file}" "${eval_dir}/FINAL_SELECTION.json"
    printf 'stopped_at_step=%s\nselection=%s\n' "${stop_step}" "${selection_file}" \
      > "${sequence_finished}"
    echo "Early stopping triggered at step ${stop_step}"
    exit 0
  fi
done

cp -n "${eval_dir}/selection_through_${max_steps}.json" "${eval_dir}/FINAL_SELECTION.json"
printf 'reached_max_steps=%s\nselection=%s\n' "${max_steps}" \
  "${eval_dir}/selection_through_${max_steps}.json" > "${sequence_finished}"
echo "Reached MAX_STEPS=${max_steps} without an earlier stop"
