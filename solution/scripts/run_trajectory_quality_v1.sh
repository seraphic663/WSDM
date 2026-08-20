#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
experiment_id="${EXPERIMENT_ID:-trajectory_quality_v1_20260724}"
experiment_root="${EXPERIMENT_ROOT:-${project_root}/ccir/outputs/experiments/${experiment_id}}"
base_model="${BASE_MODEL:-${project_root}/ccir/models/Qwen3-Embedding-0.6B}"
arm_a_data="${ARM_A_DATA:-${project_root}/ccir/data/experiments/early_stop_v1/train.jsonl}"
arms_root="${ARMS_ROOT:-${project_root}/ccir/data/experiments/trajectory_quality_v1/arms}"
arm_b_data="${ARM_B_DATA:-${arms_root}/arm_b_hard_filter.jsonl}"
arm_c_data="${ARM_C_DATA:-${arms_root}/arm_c_soft_weight.jsonl}"
arms_manifest="${ARMS_MANIFEST:-${arms_root}/manifest.json}"
dev_data="${DEV_DATA:-${project_root}/ccir/data/experiments/early_stop_v1/dev.jsonl}"
eval_batch_size="${EVAL_BATCH_SIZE:-16}"
bootstrap_samples="${BOOTSTRAP_SAMPLES:-10000}"
seed="${SEED:-20260716}"
master_port="${MASTER_PORT:-29541}"

cd "${project_root}"
for required in \
  "${base_model}/model.safetensors" \
  "${arm_a_data}" \
  "${arm_b_data}" \
  "${arm_c_data}" \
  "${arms_manifest}" \
  "${dev_data}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required input missing: ${required}" >&2
    exit 2
  fi
done
if [[ "${dev_data}" == *test* || "${dev_data}" == *TEST* ]]; then
  echo "Refusing test-like evaluation input: ${dev_data}" >&2
  exit 2
fi

mkdir -p "${experiment_root}"
running="${experiment_root}/RUNNING"
failed="${experiment_root}/FAILED"
completed="${experiment_root}/COMPLETED"
if [[ -e "${completed}" ]]; then
  echo "Experiment is already complete: ${completed}" >&2
  exit 2
fi
if [[ -e "${running}" ]]; then
  old_pid="$(awk -F= '$1 == "pid" {print $2}' "${running}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Refusing overlap with live experiment process ${old_pid}" >&2
    exit 3
  fi
  mv "${running}" "${running}.stale.$(date +%Y%m%d-%H%M%S)"
fi
rm -f "${failed}"
printf 'started_at=%s\npid=%s\n' "$(date --iso-8601=seconds)" "$$" > "${running}"

on_exit() {
  rc=$?
  trap - EXIT
  rm -f "${running}"
  if [[ ${rc} -ne 0 ]]; then
    printf 'failed_at=%s\nexit_code=%s\n' "$(date --iso-8601=seconds)" "${rc}" > "${failed}"
  fi
  exit "${rc}"
}
trap on_exit EXIT

log_file="${experiment_root}/pipeline.log"
exec > >(tee -a "${log_file}") 2>&1

arm_a_sha="$(sha256sum "${arm_a_data}" | awk '{print $1}')"
arm_b_sha="$(sha256sum "${arm_b_data}" | awk '{print $1}')"
arm_c_sha="$(sha256sum "${arm_c_data}" | awk '{print $1}')"
dev_sha="$(sha256sum "${dev_data}" | awk '{print $1}')"
model_sha="$(sha256sum "${base_model}/model.safetensors" | awk '{print $1}')"
git_head="$(git rev-parse HEAD)"
config_file="${experiment_root}/config.env"
config_payload="$({
  printf 'experiment_id=%s\n' "${experiment_id}"
  printf 'git_head=%s\n' "${git_head}"
  printf 'base_model=%s\nbase_model_sha256=%s\n' "${base_model}" "${model_sha}"
  printf 'arm_a_data=%s\narm_a_sha256=%s\n' "${arm_a_data}" "${arm_a_sha}"
  printf 'arm_b_data=%s\narm_b_sha256=%s\n' "${arm_b_data}" "${arm_b_sha}"
  printf 'arm_c_data=%s\narm_c_sha256=%s\n' "${arm_c_data}" "${arm_c_sha}"
  printf 'arms_manifest=%s\narms_manifest_sha256=%s\n' "${arms_manifest}" "$(sha256sum "${arms_manifest}" | awk '{print $1}')"
  printf 'dev_data=%s\ndev_sha256=%s\n' "${dev_data}" "${dev_sha}"
  printf 'short_max_steps=1000\nsave_steps=500\n'
  printf 'per_device_batch=1\ngradient_accumulation=4\nworld_size=2\n'
  printf 'learning_rate=1e-6\nseed=%s\n' "${seed}"
  printf 'bootstrap_samples=%s\nlocked_test_used=false\n' "${bootstrap_samples}"
})"
if [[ -e "${config_file}" ]]; then
  if [[ "$(<"${config_file}")" != "${config_payload}" ]]; then
    echo "Refusing to resume with changed experiment configuration" >&2
    diff -u "${config_file}" <(printf '%s\n' "${config_payload}") || true
    exit 2
  fi
else
  printf '%s\n' "${config_payload}" > "${config_file}"
fi

echo "=== trajectory quality v1 ==="
date --iso-8601=seconds
cat "${config_file}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

train_short_arm() {
  local arm="$1"
  local data="$2"
  local run_id="trajectory_quality_v1_${arm}_1000_20260724"
  local output="${project_root}/ccir/outputs/checkpoints/${run_id}"
  if [[ ! -e "${output}/COMPLETED" ]]; then
    echo "=== train arm ${arm} through 1000 optimizer steps ==="
    RUN_ID="${run_id}" \
    MODEL_PATH="${base_model}" \
    TRAIN_DATA="${data}" \
    OUTPUT_DIR="${output}" \
    LOG_FILE="${project_root}/ccir/outputs/logs/${run_id}.log" \
    CACHE_PATH="${project_root}/ccir/data/cache/${run_id}" \
    NUM_EPOCHS=1 \
    MAX_STEPS=1000 \
    SAVE_STEPS=500 \
    SAVE_TOTAL_LIMIT=3 \
    PER_DEVICE_BATCH=1 \
    GRADIENT_ACCUMULATION=4 \
    TRAIN_GROUP_SIZE=6 \
    QUERY_MAX_LEN=128 \
    PASSAGE_MAX_LEN=512 \
    LEARNING_RATE=1e-6 \
    SEED="${seed}" \
    MASTER_PORT="${master_port}" \
    RESUME_MODE=auto \
    COLLISION_FREE_QUERIES=0 \
    bash solution/scripts/run_qwen3_dual_train.sh
  fi
  for step in 500 1000; do
    local model="${output}/checkpoint-${step}"
    local eval_dir="${experiment_root}/eval/arm_${arm}"
    local eval_file="${eval_dir}/step_${step}.json"
    if [[ ! -s "${eval_file}" ]]; then
      if [[ ! -f "${model}/model.safetensors" ]]; then
        echo "Missing arm ${arm} checkpoint ${step}: ${model}" >&2
        exit 4
      fi
      mkdir -p "${eval_dir}"
      echo "=== evaluate arm ${arm} checkpoint ${step} on dev1500 ==="
      .venv/bin/python solution/src/evaluate_qwen3_pairs.py \
        --model "${model}" \
        --input "${dev_data}" \
        --output "${eval_file}" \
        --batch-size "${eval_batch_size}" \
        --max-length 512 \
        --device cuda:0
    fi
  done
}

train_short_arm a "${arm_a_data}"
train_short_arm b "${arm_b_data}"
train_short_arm c "${arm_c_data}"

gate_dir="${experiment_root}/gate"
mkdir -p "${gate_dir}"
arm_b_gate="${gate_dir}/arm_b_vs_a.json"
arm_c_gate="${gate_dir}/arm_c_vs_a.json"
if [[ ! -s "${arm_b_gate}" ]]; then
  .venv/bin/python solution/src/compare_paired_evals.py \
    --pair checkpoint-500 "${experiment_root}/eval/arm_a/step_500.json" "${experiment_root}/eval/arm_b/step_500.json" \
    --pair final-1000 "${experiment_root}/eval/arm_a/step_1000.json" "${experiment_root}/eval/arm_b/step_1000.json" \
    --bootstrap-samples "${bootstrap_samples}" \
    --seed 20260724 \
    --output "${arm_b_gate}"
fi
if [[ ! -s "${arm_c_gate}" ]]; then
  .venv/bin/python solution/src/compare_paired_evals.py \
    --pair checkpoint-500 "${experiment_root}/eval/arm_a/step_500.json" "${experiment_root}/eval/arm_c/step_500.json" \
    --pair final-1000 "${experiment_root}/eval/arm_a/step_1000.json" "${experiment_root}/eval/arm_c/step_1000.json" \
    --bootstrap-samples "${bootstrap_samples}" \
    --seed 20260725 \
    --output "${arm_c_gate}"
fi

selection="${gate_dir}/selection.json"
if [[ ! -s "${selection}" ]]; then
  .venv/bin/python solution/src/select_trajectory_quality_gate.py \
    --arm-b-gate "${arm_b_gate}" \
    --arm-c-gate "${arm_c_gate}" \
    --output "${selection}"
fi
selected_arm="$(.venv/bin/python - "${selection}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value.get("selected_arm") or "")
PY
)"

if [[ -z "${selected_arm}" ]]; then
  printf 'stopped_at=%s\nselection=%s\n' "$(date --iso-8601=seconds)" "${selection}" \
    > "${experiment_root}/STOPPED_BY_GATE"
  printf 'completed_at=%s\nresult=stopped_by_gate\n' "$(date --iso-8601=seconds)" \
    > "${completed}"
  echo "Neither B nor C passed; full epoch not started."
  exit 0
fi

if [[ "${selected_arm}" == "B" ]]; then
  selected_data="${arm_b_data}"
  selected_slug="b"
elif [[ "${selected_arm}" == "C" ]]; then
  selected_data="${arm_c_data}"
  selected_slug="c"
else
  echo "Unexpected selected arm: ${selected_arm}" >&2
  exit 5
fi

full_run_id="trajectory_quality_v1_${selected_slug}_full_epoch_20260724"
full_output="${project_root}/ccir/outputs/checkpoints/${full_run_id}"
printf 'started_at=%s\nselected_arm=%s\nrun_id=%s\n' \
  "$(date --iso-8601=seconds)" "${selected_arm}" "${full_run_id}" \
  > "${experiment_root}/FULL_EPOCH_STARTED"
if [[ ! -e "${full_output}/COMPLETED" ]]; then
  RUN_ID="${full_run_id}" \
  MODEL_PATH="${base_model}" \
  TRAIN_DATA="${selected_data}" \
  OUTPUT_DIR="${full_output}" \
  LOG_FILE="${project_root}/ccir/outputs/logs/${full_run_id}.log" \
  CACHE_PATH="${project_root}/ccir/data/cache/${full_run_id}" \
  NUM_EPOCHS=1 \
  MAX_STEPS=-1 \
  SAVE_STEPS=500 \
  SAVE_TOTAL_LIMIT=3 \
  PER_DEVICE_BATCH=1 \
  GRADIENT_ACCUMULATION=4 \
  TRAIN_GROUP_SIZE=6 \
  QUERY_MAX_LEN=128 \
  PASSAGE_MAX_LEN=512 \
  LEARNING_RATE=1e-6 \
  SEED="${seed}" \
  MASTER_PORT="${master_port}" \
  RESUME_MODE=auto \
  COLLISION_FREE_QUERIES=0 \
  bash solution/scripts/run_qwen3_dual_train.sh
fi

full_eval="${experiment_root}/eval/full_${selected_slug}_dev1500.json"
if [[ ! -s "${full_eval}" ]]; then
  .venv/bin/python solution/src/evaluate_qwen3_pairs.py \
    --model "${full_output}" \
    --input "${dev_data}" \
    --output "${full_eval}" \
    --batch-size "${eval_batch_size}" \
    --max-length 512 \
    --device cuda:0
fi
printf 'completed_at=%s\nselected_arm=%s\nrun_id=%s\neval=%s\n' \
  "$(date --iso-8601=seconds)" "${selected_arm}" "${full_run_id}" "${full_eval}" \
  > "${experiment_root}/FULL_EPOCH_COMPLETED"
printf 'completed_at=%s\nresult=full_epoch_completed\nselected_arm=%s\n' \
  "$(date --iso-8601=seconds)" "${selected_arm}" > "${completed}"
echo "Trajectory quality experiment completed."
