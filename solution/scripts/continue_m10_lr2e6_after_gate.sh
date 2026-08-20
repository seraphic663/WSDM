#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
short_run_id="${SHORT_RUN_ID:-m10_lr2e6_gate1000_20260729_seed20260716}"
full_run_id="${FULL_RUN_ID:-m10_lr2e6_full_earlystop_20260729_seed20260716}"
experiment_root="${EXPERIMENT_ROOT:-${project_root}/ccir/outputs/experiments/m10_lr2e6_final_candidate_20260729}"
short_output="${project_root}/ccir/outputs/checkpoints/${short_run_id}"
short_eval="${project_root}/ccir/outputs/eval/${short_run_id}"
full_output="${project_root}/ccir/outputs/checkpoints/${full_run_id}"
full_eval="${project_root}/ccir/outputs/eval/${full_run_id}"
raw_eval="${project_root}/ccir/outputs/experiments/trajectory_search_idx_v2_20260726/eval/seed20260716/raw"
base_model="${project_root}/ccir/models/Qwen3-Embedding-0.6B"
train_data="${project_root}/ccir/data/experiments/early_stop_v1/train.jsonl"
dev_data="${project_root}/ccir/data/experiments/early_stop_v1/dev.jsonl"
baseline_eval="${project_root}/ccir/outputs/eval/early_stop_v1/m00_dev1500.json"
gate_output="${experiment_root}/paired_gate_vs_raw.json"
running="${experiment_root}/CONTINUATION_RUNNING"
failed="${experiment_root}/CONTINUATION_FAILED"
completed="${experiment_root}/CONTINUATION_COMPLETED"

expected_model_sha="0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
expected_train_sha="158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9"

cd "${project_root}"
mkdir -p "${experiment_root}"
if [[ -e "${completed}" ]]; then
  echo "Continuation is already complete: ${completed}" >&2
  exit 2
fi
if [[ -e "${running}" ]]; then
  old_pid="$(awk -F= '$1 == "pid" {print $2}' "${running}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Refusing to overlap live continuation process ${old_pid}" >&2
    exit 3
  fi
  mv "${running}" "${running}.stale.$(date +%Y%m%d-%H%M%S)"
fi
rm -f "${failed}"
printf 'started_at=%s\npid=%s\nshort_run_id=%s\nfull_run_id=%s\n' \
  "$(date --iso-8601=seconds)" "$$" "${short_run_id}" "${full_run_id}" > "${running}"

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

test "$(sha256sum "${base_model}/model.safetensors" | awk '{print $1}')" = "${expected_model_sha}"
test "$(sha256sum "${train_data}" | awk '{print $1}')" = "${expected_train_sha}"
for required in \
  "${dev_data}" \
  "${baseline_eval}" \
  "${raw_eval}/step_500.json" \
  "${raw_eval}/step_1000.json"; do
  test -s "${required}" || { echo "Missing required input: ${required}" >&2; exit 4; }
done

while [[ ! -e "${short_eval}/SEQUENCE_FINISHED" ]]; do
  if [[ -e "${short_eval}/SEQUENCE_FAILED" || -e "${short_output}/FAILED" ]]; then
    echo "Short M10 sequence failed; refusing continuation" >&2
    exit 5
  fi
  sleep 30
done

for required in "${short_eval}/dev_step_500.json" "${short_eval}/dev_step_1000.json"; do
  test -s "${required}" || { echo "Missing short evaluation: ${required}" >&2; exit 6; }
done

if [[ ! -s "${gate_output}" ]]; then
  .venv/bin/python solution/src/compare_paired_evals.py \
    --pair checkpoint-500 \
      "${raw_eval}/step_500.json" \
      "${short_eval}/dev_step_500.json" \
    --pair final-1000 \
      "${raw_eval}/step_1000.json" \
      "${short_eval}/dev_step_1000.json" \
    --bootstrap-samples 10000 \
    --seed 20260716 \
    --output "${gate_output}"
fi

if [[ ! -s "${short_output}/SHORT_RUN_PRUNE_MANIFEST.json" ]]; then
  .venv/bin/python solution/src/prune_short_training_run.py \
    --output-dir "${short_output}" \
    --eval-500 "${short_eval}/dev_step_500.json" \
    --eval-1000 "${short_eval}/dev_step_1000.json" \
    --allowed-root "${project_root}/ccir/outputs/checkpoints"
fi

gate_passed="$(
  .venv/bin/python - "${gate_output}" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
print(str(value["gate"]["passed"]).lower())
PY
)"

if [[ "${gate_passed}" != "true" ]]; then
  printf 'completed_at=%s\nresult=stopped_by_short_gate\ngate=%s\nfull_epoch_started=false\n' \
    "$(date --iso-8601=seconds)" "${gate_output}" > "${completed}"
  rm -f "${running}"
  trap - EXIT
  echo "M10 LR=2e-6 failed the short gate; full sequence was not started."
  exit 0
fi

if [[ -e "${full_output}" || -e "${full_eval}" ]]; then
  echo "Full-run target already exists; refusing an ambiguous restart" >&2
  exit 7
fi

RUN_ID="${full_run_id}" \
MODEL_PATH="${base_model}" \
TRAIN_DATA="${train_data}" \
DEV_DATA="${dev_data}" \
BASELINE_EVAL="${baseline_eval}" \
OUTPUT_DIR="${full_output}" \
EVAL_DIR="${full_eval}" \
SEQUENCE_LOG="${project_root}/ccir/outputs/logs/${full_run_id}_sequence.log" \
MAX_STEPS=11764 \
EVAL_INTERVAL=1000 \
EXPECTED_ROWS=1500 \
MIN_STEPS=2000 \
MIN_DELTA=0.001 \
PATIENCE=2 \
GUARDRAIL_TOLERANCE=0.005 \
SAVE_TOTAL_LIMIT=3 \
PER_DEVICE_BATCH=1 \
GRADIENT_ACCUMULATION=4 \
TRAIN_GROUP_SIZE=6 \
QUERY_MAX_LEN=128 \
PASSAGE_MAX_LEN=512 \
LEARNING_RATE=2e-6 \
SEED=20260716 \
CACHE_PATH="${project_root}/ccir/data/cache/${full_run_id}" \
MASTER_PORT=29530 \
EVAL_DEVICE=cuda:0 \
bash solution/scripts/run_early_stop_sequence.sh

test -s "${full_eval}/FINAL_SELECTION.json"
printf 'completed_at=%s\nresult=full_early_stop_sequence_complete\ngate=%s\nfull_run=%s\nselection=%s\n' \
  "$(date --iso-8601=seconds)" \
  "${gate_output}" \
  "${full_run_id}" \
  "${full_eval}/FINAL_SELECTION.json" > "${completed}"
rm -f "${running}"
trap - EXIT
echo "M10 LR=2e-6 full early-stop sequence completed."
