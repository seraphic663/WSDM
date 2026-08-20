#!/usr/bin/env bash
# Test the effect of removing LRAT reweight_rate with a 500/1000-step paired gate.

set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
experiment_id="${EXPERIMENT_ID:-unit_weight_m01like_20260728}"
experiment_root="${project_root}/ccir/outputs/experiments/${experiment_id}"
base_model="${project_root}/ccir/models/Qwen3-Embedding-0.6B"
raw_data="${project_root}/ccir/data/experiments/early_stop_v1/train.jsonl"
dev_data="${project_root}/ccir/data/experiments/early_stop_v1/dev.jsonl"
unit_root="${project_root}/ccir/data/experiments/unit_weight_m01like"
unit_data="${unit_root}/train_unit_weight.jsonl"
unit_manifest="${unit_root}/manifest.json"
baseline_root="${project_root}/ccir/outputs/experiments/trajectory_search_idx_v2_20260726/eval/seed20260716/raw"
baseline_config="${project_root}/ccir/outputs/checkpoints/trajectory_search_idx_v2_20260726_raw_seed20260716_1000/run_config.env"
run_id="${experiment_id}_seed20260716_1000"
output="${project_root}/ccir/outputs/checkpoints/${run_id}"
eval_root="${experiment_root}/eval"
gate_output="${experiment_root}/gate.json"

expected_model_sha="0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
expected_raw_sha="158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9"
expected_dev_sha="6b9cac0b1dc351062b4b23999bdd463b6df096be60c483734a1579256eef9f9e"
expected_baseline_500_sha="95117bf4bdaa63ec293927d8d4df4f9e79ef346a4202d70f2ce08ce27a7f3b91"
expected_baseline_1000_sha="497cd63f1b6220a59f588c3723e961361c3163b9bebabd7cd9fd58d809a7305c"

cd "${project_root}"
for required in \
  "${base_model}/model.safetensors" \
  "${raw_data}" \
  "${dev_data}" \
  "${baseline_config}" \
  "${baseline_root}/step_500.json" \
  "${baseline_root}/step_1000.json"; do
  [ -s "${required}" ] || { echo "Missing required input: ${required}" >&2; exit 2; }
  [ ! -L "${required}" ] || { echo "Symlinked input is forbidden: ${required}" >&2; exit 2; }
done
for identity in \
  "model:${base_model}/model.safetensors:${expected_model_sha}" \
  "raw:${raw_data}:${expected_raw_sha}" \
  "dev:${dev_data}:${expected_dev_sha}" \
  "baseline500:${baseline_root}/step_500.json:${expected_baseline_500_sha}" \
  "baseline1000:${baseline_root}/step_1000.json:${expected_baseline_1000_sha}"; do
  IFS=: read -r label path expected <<< "${identity}"
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [ "${actual}" = "${expected}" ] || {
    echo "${label} SHA-256 mismatch: ${actual} != ${expected}" >&2
    exit 2
  }
done

.venv/bin/python - "${baseline_config}" <<'PY'
import sys
from pathlib import Path

values = {}
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    key, value = line.split("=", 1)
    values[key] = value
expected = {
    "model_sha256": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
    "data_sha256": "158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9",
    "world_size": "2",
    "train_group_size": "6",
    "query_max_len": "128",
    "passage_max_len": "512",
    "per_device_batch": "1",
    "gradient_accumulation": "4",
    "learning_rate": "1e-6",
    "num_epochs": "1",
    "max_steps": "1000",
    "save_steps": "500",
    "save_total_limit": "2",
    "seed": "20260716",
}
bad = {key: (values.get(key), value) for key, value in expected.items() if values.get(key) != value}
if bad:
    raise SystemExit(f"baseline config mismatch: {bad}")
PY

if [ ! -s "${unit_data}" ] || [ ! -s "${unit_manifest}" ]; then
  if [ -e "${unit_data}" ] || [ -e "${unit_manifest}" ]; then
    echo "Incomplete unit-weight artifact exists; refusing overwrite" >&2
    exit 2
  fi
  mkdir -p "${unit_root}"
  .venv/bin/python solution/src/build_unit_weight_pairs.py \
    --input "${raw_data}" \
    --output "${unit_data}" \
    --manifest "${unit_manifest}" \
    --expected-input-sha256 "${expected_raw_sha}" \
    --expected-rows 94113
fi
unit_sha="$(sha256sum "${unit_data}" | awk '{print $1}')"
.venv/bin/python - "${unit_manifest}" "${unit_sha}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
passed = (
    value.get("rows") == 94113
    and value.get("changed_rows", 0) > 0
    and value.get("output", {}).get("sha256") == sys.argv[2]
    and value.get("unit_reweight_rate") == {
        "min": 1.0, "max": 1.0, "sum": 94113.0, "mean": 1.0
    }
    and value.get("contract", {}).get("source_fields_preserved_except") == ["reweight_rate"]
    and value.get("contract", {}).get("locked_test_used") is False
)
raise SystemExit(0 if passed else 1)
PY

mkdir -p "${experiment_root}" "${eval_root}"
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

cat > "${experiment_root}/config.env" <<EOF
experiment_id=${experiment_id}
git_head=$(git rev-parse HEAD)
base_model_sha256=${expected_model_sha}
raw_data_sha256=${expected_raw_sha}
unit_data_sha256=${unit_sha}
dev_data_sha256=${expected_dev_sha}
seed=20260716
short_steps=1000
save_steps=500
bootstrap_samples=10000
only_changed_field=reweight_rate
candidate_weight=1.0
locked_test_used=false
full_epoch_auto_authorized=false
EOF

if [ ! -e "${output}/COMPLETED" ]; then
  RUN_ID="${run_id}" \
  MODEL_PATH="${base_model}" \
  TRAIN_DATA="${unit_data}" \
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
  SEED=20260716 \
  MASTER_PORT=29571 \
  RESUME_MODE=auto \
  COLLISION_FREE_QUERIES=0 \
  bash solution/scripts/run_qwen3_dual_train.sh
fi

for step in 500 1000; do
  eval_file="${eval_root}/step_${step}.json"
  if [ ! -s "${eval_file}" ]; then
    eval_model="${output}/checkpoint-${step}"
    .venv/bin/python solution/src/evaluate_qwen3_pairs.py \
      --model "${eval_model}" \
      --input "${dev_data}" \
      --output "${eval_file}" \
      --batch-size 16 \
      --max-length 512 \
      --device cuda:0
  fi
done

if [ ! -s "${gate_output}" ]; then
  .venv/bin/python solution/src/compare_paired_evals.py \
    --pair checkpoint-500 \
      "${baseline_root}/step_500.json" \
      "${eval_root}/step_500.json" \
    --pair final-1000 \
      "${baseline_root}/step_1000.json" \
      "${eval_root}/step_1000.json" \
    --bootstrap-samples 10000 \
    --seed 20260716 \
    --output "${gate_output}"
fi

if [ ! -s "${output}/SHORT_RUN_PRUNE_MANIFEST.json" ]; then
  .venv/bin/python solution/src/prune_short_training_run.py \
    --output-dir "${output}" \
    --eval-500 "${eval_root}/step_500.json" \
    --eval-1000 "${eval_root}/step_1000.json" \
    --allowed-root "${project_root}/ccir/outputs/checkpoints"
fi

gate_passed="$(
  .venv/bin/python - "${gate_output}" <<'PY'
import json, sys
print(str(json.load(open(sys.argv[1], encoding="utf-8"))["gate"]["passed"]).lower())
PY
)"
printf 'completed_at=%s\nresult=short_gate_complete\ngate_passed=%s\nfull_epoch_started=false\n' \
  "$(date -Is)" "${gate_passed}" > "${completed}"
rm -f "${running}"
trap - EXIT
echo "Unit-weight short experiment completed; full epoch was not started."
