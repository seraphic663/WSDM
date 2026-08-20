#!/usr/bin/env bash
# Run a gated raw-vs-search-index experiment after the current GPU job finishes.

set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
experiment_id="${EXPERIMENT_ID:-trajectory_search_idx_v2_20260726}"
experiment_root="${project_root}/ccir/outputs/experiments/${experiment_id}"
base_model="${project_root}/ccir/models/Qwen3-Embedding-0.6B"
raw_data="${project_root}/ccir/data/experiments/early_stop_v1/train.jsonl"
arm_root="${project_root}/ccir/data/experiments/trajectory_search_idx_v2"
candidate_data="${arm_root}/train_search_idx_soft.jsonl"
arm_manifest="${arm_root}/manifest.json"
arm_audit="${arm_root}/audit.json"
dev_data="${project_root}/ccir/data/experiments/early_stop_v1/dev.jsonl"
seeds=(20260716 20260717 20260718)
bootstrap_samples=10000
expected_model_sha="0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
expected_raw_sha="158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9"
expected_candidate_sha="8fd600b9d5c46bd21f91ad073a3bddeab375ef2fa5c9998c61c75a68ccb7fced"
expected_manifest_sha="b45bc4efee660f4e82164996162b30af0205e40cbceb3ec1863aef42ec1cdd4b"
expected_audit_sha="9360bce567050e69184dc90a5441621be91233e11aea6fe6c789774bd663e772"
expected_dev_sha="6b9cac0b1dc351062b4b23999bdd463b6df096be60c483734a1579256eef9f9e"

cd "${project_root}"
for required in \
  "${base_model}/model.safetensors" \
  "${raw_data}" \
  "${candidate_data}" \
  "${arm_manifest}" \
  "${arm_audit}" \
  "${dev_data}"; do
    [ -s "${required}" ] || { echo "Missing required input: ${required}" >&2; exit 2; }
    [ ! -L "${required}" ] || { echo "Symlinked input is forbidden: ${required}" >&2; exit 2; }
done
if [[ "${dev_data,,}" == *test* ]]; then
  echo "Refusing test-like evaluation input: ${dev_data}" >&2
  exit 2
fi
raw_sha="$(sha256sum "${raw_data}" | awk '{print $1}')"
candidate_sha="$(sha256sum "${candidate_data}" | awk '{print $1}')"
model_sha="$(sha256sum "${base_model}/model.safetensors" | awk '{print $1}')"
manifest_sha="$(sha256sum "${arm_manifest}" | awk '{print $1}')"
audit_sha="$(sha256sum "${arm_audit}" | awk '{print $1}')"
dev_sha="$(sha256sum "${dev_data}" | awk '{print $1}')"
for identity in \
  "model:${model_sha}:${expected_model_sha}" \
  "raw:${raw_sha}:${expected_raw_sha}" \
  "candidate:${candidate_sha}:${expected_candidate_sha}" \
  "manifest:${manifest_sha}:${expected_manifest_sha}" \
  "audit:${audit_sha}:${expected_audit_sha}" \
  "dev:${dev_sha}:${expected_dev_sha}"; do
  IFS=: read -r label actual expected <<< "${identity}"
  if [ "${actual}" != "${expected}" ]; then
    echo "${label} SHA-256 mismatch: ${actual}" >&2
    exit 2
  fi
done
if ! .venv/bin/python - "${arm_manifest}" "${arm_audit}" "${candidate_sha}" <<'PY'
import json, sys
manifest=json.load(open(sys.argv[1], encoding="utf-8"))
audit=json.load(open(sys.argv[2], encoding="utf-8"))
candidate_sha=sys.argv[3]
passed=(
    manifest.get("rows") == 94113
    and manifest.get("bucket_counts") == {"ambiguous": 136, "stable": 93977}
    and manifest.get("outputs", {}).get("train_sha256") == candidate_sha
    and manifest.get("contract", {}).get("locked_test_used") is False
    and audit.get("passed") is True
    and audit.get("rows") == 94113
    and audit.get("neutral_changed_rows") == 0
    and audit.get("output_sha256") == candidate_sha
    and audit.get("locked_test_used") is False
)
raise SystemExit(0 if passed else 1)
PY
then
  echo "Trajectory arm manifest/audit contract did not pass" >&2
  exit 2
fi

mkdir -p "${experiment_root}"
running="${experiment_root}/RUNNING"
failed="${experiment_root}/FAILED"
completed="${experiment_root}/COMPLETED"
if [ -e "${completed}" ]; then
  echo "Experiment already completed: ${experiment_root}" >&2
  exit 2
fi
if [ -e "${running}" ]; then
  old_pid="$(awk -F= '$1=="pid"{print $2}' "${running}" 2>/dev/null || true)"
  if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Experiment already running as PID ${old_pid}" >&2
    exit 3
  fi
  mv "${running}" "${running}.stale.$(date +%Y%m%d-%H%M%S)"
fi
rm -f "${failed}"
printf 'started_at=%s\npid=%s\n' "$(date -Is)" "$$" > "${running}"

on_exit() {
  rc=$?
  trap - EXIT
  rm -f "${running}"
  if [ "${rc}" -ne 0 ]; then
    printf 'failed_at=%s\nexit_code=%s\n' "$(date -Is)" "${rc}" > "${failed}"
  fi
  exit "${rc}"
}
trap on_exit EXIT
exec > >(tee -a "${experiment_root}/pipeline.log") 2>&1

git_head="$(git rev-parse HEAD)"
cat > "${experiment_root}/config.env" <<EOF
experiment_id=${experiment_id}
git_head=${git_head}
base_model=${base_model}
base_model_sha256=${model_sha}
raw_data=${raw_data}
raw_data_sha256=${raw_sha}
candidate_data=${candidate_data}
candidate_data_sha256=${candidate_sha}
arm_manifest_sha256=${manifest_sha}
arm_audit_sha256=${audit_sha}
dev_data=${dev_data}
dev_data_sha256=${dev_sha}
seeds=${seeds[*]}
short_steps=1000
save_steps=500
bootstrap_samples=${bootstrap_samples}
locked_test_used=false
EOF

train_and_eval() {
  local seed="$1"
  local arm="$2"
  local data="$3"
  local port="$4"
  local run_id="${experiment_id}_${arm}_seed${seed}_1000"
  local output="${project_root}/ccir/outputs/checkpoints/${run_id}"
  local eval_dir="${experiment_root}/eval/seed${seed}/${arm}"

  if [ ! -e "${output}/COMPLETED" ]; then
    RUN_ID="${run_id}" \
    MODEL_PATH="${base_model}" \
    TRAIN_DATA="${data}" \
    OUTPUT_DIR="${output}" \
    LOG_FILE="${project_root}/ccir/outputs/logs/${run_id}.log" \
    CACHE_PATH="${project_root}/ccir/data/cache/${run_id}" \
    NUM_EPOCHS=1 \
    MAX_STEPS=1000 \
    SAVE_STEPS=500 \
    SAVE_TOTAL_LIMIT=2 \
    PER_DEVICE_BATCH=1 \
    GRADIENT_ACCUMULATION=4 \
    TRAIN_GROUP_SIZE=6 \
    QUERY_MAX_LEN=128 \
    PASSAGE_MAX_LEN=512 \
    LEARNING_RATE=1e-6 \
    SEED="${seed}" \
    MASTER_PORT="${port}" \
    RESUME_MODE=auto \
    COLLISION_FREE_QUERIES=0 \
    bash solution/scripts/run_qwen3_dual_train.sh
  fi

  mkdir -p "${eval_dir}"
  for step in 500 1000; do
    local eval_file="${eval_dir}/step_${step}.json"
    if [ ! -s "${eval_file}" ]; then
      local eval_model="${output}/checkpoint-${step}"
      if [ "${step}" = "1000" ] && [ ! -d "${eval_model}" ] && [ -s "${output}/SHORT_RUN_PRUNE_MANIFEST.json" ]; then
        eval_model="${output}"
      fi
      .venv/bin/python solution/src/evaluate_qwen3_pairs.py \
        --model "${eval_model}" \
        --input "${dev_data}" \
        --output "${eval_file}" \
        --batch-size 16 \
        --max-length 512 \
      --device cuda:0
    fi
  done
  if [ ! -s "${output}/SHORT_RUN_PRUNE_MANIFEST.json" ]; then
    .venv/bin/python solution/src/prune_short_training_run.py \
      --output-dir "${output}" \
      --eval-500 "${eval_dir}/step_500.json" \
      --eval-1000 "${eval_dir}/step_1000.json" \
      --allowed-root "${project_root}/ccir/outputs/checkpoints"
  fi
}

gate_seed() {
  local seed="$1"
  local output="${experiment_root}/gate/seed${seed}.json"
  mkdir -p "${experiment_root}/gate"
  if [ ! -s "${output}" ]; then
    .venv/bin/python solution/src/compare_paired_evals.py \
      --pair checkpoint-500 \
        "${experiment_root}/eval/seed${seed}/raw/step_500.json" \
        "${experiment_root}/eval/seed${seed}/search/step_500.json" \
      --pair final-1000 \
        "${experiment_root}/eval/seed${seed}/raw/step_1000.json" \
        "${experiment_root}/eval/seed${seed}/search/step_1000.json" \
      --bootstrap-samples "${bootstrap_samples}" \
      --seed "${seed}" \
      --output "${output}"
  fi
}

gate_passed() {
  .venv/bin/python - "$1" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if value.get("gate", {}).get("passed") is True else 1)
PY
}

echo "=== trajectory search-index v2 ==="
date -Is
cat "${experiment_root}/config.env"

first_seed="${seeds[0]}"
train_and_eval "${first_seed}" raw "${raw_data}" 29561
train_and_eval "${first_seed}" search "${candidate_data}" 29562
gate_seed "${first_seed}"
if ! gate_passed "${experiment_root}/gate/seed${first_seed}.json"; then
  printf 'stopped_at=%s\nreason=first_seed_gate_failed\n' "$(date -Is)" \
    > "${experiment_root}/STOPPED_BY_GATE"
  printf 'completed_at=%s\nresult=stopped_by_gate\n' "$(date -Is)" > "${completed}"
  echo "First seed failed; no additional seeds or full epoch will run."
  exit 0
fi

train_and_eval "${seeds[1]}" raw "${raw_data}" 29563
train_and_eval "${seeds[1]}" search "${candidate_data}" 29564
gate_seed "${seeds[1]}"
train_and_eval "${seeds[2]}" raw "${raw_data}" 29565
train_and_eval "${seeds[2]}" search "${candidate_data}" 29566
gate_seed "${seeds[2]}"

aggregate_gate="${experiment_root}/gate/multiseed_selection.json"
if [ ! -s "${aggregate_gate}" ]; then
  .venv/bin/python solution/src/select_multiseed_trajectory_gate.py \
    --gate "${seeds[0]}" "${experiment_root}/gate/seed${seeds[0]}.json" \
    --gate "${seeds[1]}" "${experiment_root}/gate/seed${seeds[1]}.json" \
    --gate "${seeds[2]}" "${experiment_root}/gate/seed${seeds[2]}.json" \
    --output "${aggregate_gate}"
fi
if ! .venv/bin/python - "${aggregate_gate}" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if value.get("full_epoch_authorized") is True else 1)
PY
then
  printf 'stopped_at=%s\nreason=multiseed_gate_failed\n' "$(date -Is)" \
    > "${experiment_root}/STOPPED_BY_GATE"
  printf 'completed_at=%s\nresult=stopped_by_gate\n' "$(date -Is)" > "${completed}"
  echo "Multi-seed gate failed; full epoch will not run."
  exit 0
fi

full_run_id="${experiment_id}_search_full_seed${first_seed}"
full_output="${project_root}/ccir/outputs/checkpoints/${full_run_id}"
printf 'started_at=%s\nrun_id=%s\n' "$(date -Is)" "${full_run_id}" \
  > "${experiment_root}/FULL_EPOCH_STARTED"
if [ ! -e "${full_output}/COMPLETED" ]; then
  RUN_ID="${full_run_id}" \
  MODEL_PATH="${base_model}" \
  TRAIN_DATA="${candidate_data}" \
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
  SEED="${first_seed}" \
  MASTER_PORT=29567 \
  RESUME_MODE=auto \
  COLLISION_FREE_QUERIES=0 \
  bash solution/scripts/run_qwen3_dual_train.sh
fi

full_eval="${experiment_root}/eval/full_dev1500.json"
if [ ! -s "${full_eval}" ]; then
  .venv/bin/python solution/src/evaluate_qwen3_pairs.py \
    --model "${full_output}" \
    --input "${dev_data}" \
    --output "${full_eval}" \
    --batch-size 16 \
    --max-length 512 \
    --device cuda:0
fi
printf 'completed_at=%s\nresult=full_epoch_completed\nrun_id=%s\n' \
  "$(date -Is)" "${full_run_id}" > "${completed}"
echo "Trajectory search-index v2 pipeline completed."
