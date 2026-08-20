#!/usr/bin/env bash
# Run the interpretable aggressive search-stage 500/1000-step experiment.

set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
experiment_id="${EXPERIMENT_ID:-aggressive_search_weight_v1_20260728}"
experiment_root="${project_root}/ccir/outputs/experiments/${experiment_id}"
base_model="${project_root}/ccir/models/Qwen3-Embedding-0.6B"
raw_data="${project_root}/ccir/data/experiments/early_stop_v1/train.jsonl"
train_provenance="${project_root}/ccir/data/experiments/trajectory_provenance_v1/provenance.jsonl"
full_provenance="${project_root}/ccir/data/experiments/trajectory_provenance_m01_full_v3/provenance.jsonl"
dev_data="${project_root}/ccir/data/experiments/early_stop_v1/dev.jsonl"
config="${project_root}/solution/configs/aggressive_search_weight_v1.json"
arm_root="${project_root}/ccir/data/experiments/aggressive_search_weight_v1"
candidate_data="${arm_root}/train.jsonl"
manifest="${arm_root}/manifest.json"
audit="${arm_root}/audit.json"
official_eval_root="${project_root}/ccir/outputs/experiments/trajectory_search_idx_v2_20260726/eval/seed20260716/raw"
unit_eval_root="${project_root}/ccir/outputs/experiments/unit_weight_m01like_20260728/eval"
run_id="${experiment_id}_seed20260716_1000"
output="${project_root}/ccir/outputs/checkpoints/${run_id}"
eval_root="${experiment_root}/eval"
gate_official="${experiment_root}/gate_vs_official.json"
gate_unit="${experiment_root}/gate_vs_unit.json"
selection="${experiment_root}/selection.json"
explanation="${experiment_root}/search_stage_effects.json"

expected_model_sha="0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
expected_raw_sha="158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9"
expected_train_provenance_sha="924d373cffbc8f229c326c1de02e7e223c87109e57c32c1f2be29ea2e65f0b15"
expected_full_provenance_sha="0dd510fcc41444bfcff99381ce077e5048375855add8498f9ab10afabea4b690"
expected_dev_sha="6b9cac0b1dc351062b4b23999bdd463b6df096be60c483734a1579256eef9f9e"
expected_config_sha="4ec33cae576ed5aac5d62c0684cd58d0ee3c0b6e3866fd0182dc0db375c62eef"
expected_official_500_sha="95117bf4bdaa63ec293927d8d4df4f9e79ef346a4202d70f2ce08ce27a7f3b91"
expected_official_1000_sha="497cd63f1b6220a59f588c3723e961361c3163b9bebabd7cd9fd58d809a7305c"
expected_unit_500_sha="3f0fd9c87d10aa5e9a91179838a91580209d93eff12bfcd8f9772255e6ef26ff"
expected_unit_1000_sha="7ac2b77888d5e03960025d34b337264f5fd238c4f7027c9a762c629a15c65f9a"

cd "${project_root}"
for identity in \
  "model:${base_model}/model.safetensors:${expected_model_sha}" \
  "raw:${raw_data}:${expected_raw_sha}" \
  "train_provenance:${train_provenance}:${expected_train_provenance_sha}" \
  "full_provenance:${full_provenance}:${expected_full_provenance_sha}" \
  "dev:${dev_data}:${expected_dev_sha}" \
  "config:${config}:${expected_config_sha}" \
  "official500:${official_eval_root}/step_500.json:${expected_official_500_sha}" \
  "official1000:${official_eval_root}/step_1000.json:${expected_official_1000_sha}" \
  "unit500:${unit_eval_root}/step_500.json:${expected_unit_500_sha}" \
  "unit1000:${unit_eval_root}/step_1000.json:${expected_unit_1000_sha}"; do
  IFS=: read -r label path expected <<< "${identity}"
  [ -s "${path}" ] || { echo "Missing required input: ${path}" >&2; exit 2; }
  [ ! -L "${path}" ] || { echo "Symlinked input is forbidden: ${path}" >&2; exit 2; }
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [ "${actual}" = "${expected}" ] || {
    echo "${label} SHA-256 mismatch: ${actual} != ${expected}" >&2
    exit 2
  }
done

available_bytes="$(df -B1 --output=avail "${project_root}" | tail -n 1 | tr -d ' ')"
minimum_bytes=$((60 * 1024 * 1024 * 1024))
if (( available_bytes < minimum_bytes )); then
  echo "Insufficient disk headroom: ${available_bytes} < ${minimum_bytes}" >&2
  exit 3
fi

if [ ! -s "${candidate_data}" ] || [ ! -s "${manifest}" ]; then
  if [ -e "${candidate_data}" ] || [ -e "${manifest}" ]; then
    echo "Incomplete candidate artifact exists; refusing overwrite" >&2
    exit 2
  fi
  mkdir -p "${arm_root}"
  .venv/bin/python solution/src/build_aggressive_search_weight_arm.py \
    --pairs "${raw_data}" \
    --provenance "${train_provenance}" \
    --config "${config}" \
    --output "${candidate_data}" \
    --manifest "${manifest}" \
    --expected-pairs-sha256 "${expected_raw_sha}" \
    --expected-provenance-sha256 "${expected_train_provenance_sha}" \
    --expected-rows 94113
fi
if [ ! -s "${audit}" ]; then
  .venv/bin/python solution/src/audit_aggressive_search_weight_arm.py \
    --pairs "${raw_data}" \
    --candidate "${candidate_data}" \
    --provenance "${train_provenance}" \
    --config "${config}" \
    --manifest "${manifest}" \
    --output "${audit}"
fi
candidate_sha="$(sha256sum "${candidate_data}" | awk '{print $1}')"
.venv/bin/python - "${manifest}" "${audit}" "${candidate_sha}" <<'PY'
import json, math, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
audit = json.load(open(sys.argv[2], encoding="utf-8"))
candidate_sha = sys.argv[3]
passed = (
    manifest.get("rows") == 94113
    and manifest.get("output", {}).get("sha256") == candidate_sha
    and manifest.get("bucket_counts") == {
        "ambiguous": 136, "idx0": 14481, "idx1_2": 21357, "idx3plus": 58139
    }
    and math.isclose(manifest.get("total_weight"), 94113.0, abs_tol=1e-6)
    and manifest.get("contract", {}).get("stable_weight_ratio_idx3plus_to_idx0") == 2.0
    and manifest.get("contract", {}).get("locked_test_used") is False
    and audit.get("passed") is True
    and audit.get("candidate_sha256") == candidate_sha
    and audit.get("locked_test_used") is False
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
candidate_data_sha256=${candidate_sha}
train_provenance_sha256=${expected_train_provenance_sha}
full_provenance_sha256=${expected_full_provenance_sha}
dev_data_sha256=${expected_dev_sha}
weight_config_sha256=${expected_config_sha}
seed=20260716
short_steps=1000
save_steps=500
bootstrap_samples=10000
baselines=official_weight,unit_weight
locked_test_used=false
full_epoch_auto_authorized=false
EOF

if [ ! -e "${output}/COMPLETED" ]; then
  RUN_ID="${run_id}" \
  MODEL_PATH="${base_model}" \
  TRAIN_DATA="${candidate_data}" \
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
  MASTER_PORT=29572 \
  RESUME_MODE=auto \
  COLLISION_FREE_QUERIES=0 \
  bash solution/scripts/run_qwen3_dual_train.sh
fi

for step in 500 1000; do
  eval_file="${eval_root}/step_${step}.json"
  if [ ! -s "${eval_file}" ]; then
    .venv/bin/python solution/src/evaluate_qwen3_pairs.py \
      --model "${output}/checkpoint-${step}" \
      --input "${dev_data}" \
      --output "${eval_file}" \
      --batch-size 16 \
      --max-length 512 \
      --device cuda:0
  fi
done

if [ ! -s "${gate_official}" ]; then
  .venv/bin/python solution/src/compare_paired_evals.py \
    --pair checkpoint-500 "${official_eval_root}/step_500.json" "${eval_root}/step_500.json" \
    --pair final-1000 "${official_eval_root}/step_1000.json" "${eval_root}/step_1000.json" \
    --bootstrap-samples 10000 \
    --seed 20260716 \
    --output "${gate_official}"
fi
if [ ! -s "${gate_unit}" ]; then
  .venv/bin/python solution/src/compare_paired_evals.py \
    --pair checkpoint-500 "${unit_eval_root}/step_500.json" "${eval_root}/step_500.json" \
    --pair final-1000 "${unit_eval_root}/step_1000.json" "${eval_root}/step_1000.json" \
    --bootstrap-samples 10000 \
    --seed 20260716 \
    --output "${gate_unit}"
fi
if [ ! -s "${selection}" ]; then
  .venv/bin/python - "${gate_official}" "${gate_unit}" "${selection}" <<'PY'
import json, os, sys, tempfile
from datetime import datetime
official = json.load(open(sys.argv[1], encoding="utf-8"))
unit = json.load(open(sys.argv[2], encoding="utf-8"))
report = {
    "created_at": datetime.now().astimezone().isoformat(),
    "gate_vs_official_passed": official["gate"]["passed"],
    "gate_vs_unit_passed": unit["gate"]["passed"],
    "both_baselines_passed": official["gate"]["passed"] and unit["gate"]["passed"],
    "full_epoch_authorized": False,
    "interpretation": (
        "A full epoch remains manually gated even if both short gates pass; "
        "this file never authorizes training by itself."
    ),
    "locked_test_used": False,
}
directory = os.path.dirname(sys.argv[3])
fd, temporary = tempfile.mkstemp(prefix=".selection.", dir=directory)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
os.replace(temporary, sys.argv[3])
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
fi
if [ ! -s "${explanation}" ]; then
  .venv/bin/python solution/src/analyze_search_weight_effects.py \
    --dev "${dev_data}" \
    --provenance "${full_provenance}" \
    --candidate-eval "${eval_root}/step_1000.json" \
    --baseline official "${official_eval_root}/step_1000.json" \
    --baseline unit "${unit_eval_root}/step_1000.json" \
    --output "${explanation}"
fi

if [ ! -s "${output}/SHORT_RUN_PRUNE_MANIFEST.json" ]; then
  .venv/bin/python solution/src/prune_short_training_run.py \
    --output-dir "${output}" \
    --eval-500 "${eval_root}/step_500.json" \
    --eval-1000 "${eval_root}/step_1000.json" \
    --allowed-root "${project_root}/ccir/outputs/checkpoints"
fi

both_passed="$(
  .venv/bin/python - "${selection}" <<'PY'
import json, sys
print(str(json.load(open(sys.argv[1], encoding="utf-8"))["both_baselines_passed"]).lower())
PY
)"
printf 'completed_at=%s\nresult=short_experiment_complete\nboth_baselines_passed=%s\nfull_epoch_started=false\n' \
  "$(date -Is)" "${both_passed}" > "${completed}"
rm -f "${running}"
trap - EXIT
echo "Aggressive search-weight short experiment completed; full epoch was not started."
