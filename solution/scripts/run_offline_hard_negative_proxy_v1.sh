#!/usr/bin/env bash
# Historical competition-compliant hard-negative proxy. This is not the
# paper's Agent trajectory data flywheel; use run_paper_flywheel_v1.sh for the
# research reproduction.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
experiment_id="${EXPERIMENT_ID:-offline_flywheel_v1_20260728}"
experiment_root="${project_root}/ccir/outputs/experiments/${experiment_id}"
data_root="${project_root}/ccir/data/experiments/${experiment_id}"
checkpoint_root="${project_root}/ccir/outputs/checkpoints"
log_root="${project_root}/ccir/outputs/logs"

source_pairs="${project_root}/ccir/data/experiments/early_stop_v1/train.jsonl"
dev_pairs="${project_root}/ccir/data/experiments/early_stop_v1/dev.jsonl"
initial_parent="${checkpoint_root}/trajectory_search_idx_v2_20260726_raw_seed20260716_1000/checkpoint-500"
initial_parent_eval="${project_root}/ccir/outputs/experiments/trajectory_search_idx_v2_20260726/eval/seed20260716/raw/step_500.json"

expected_source_sha="158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9"
expected_dev_sha="6b9cac0b1dc351062b4b23999bdd463b6df096be60c483734a1579256eef9f9e"
expected_parent_sha="acd2eb20228caf9aff83d20afb6bc35d514cd378739cc117489b7165bc213373"
salt="offline-flywheel-v1"
modulus=16
pool_size=32
hard_k=8
loop_steps=500
save_steps=1000
seed=20260728
bootstrap_samples=10000
minimum_free_gib=150
train_attempt_suffix="${TRAIN_ATTEMPT_SUFFIX:-_retry1}"

cd "${project_root}"
mkdir -p "${experiment_root}" "${data_root}" "${log_root}"
launcher_log="${experiment_root}/launcher.log"
exec > >(tee -a "${launcher_log}") 2>&1

if [[ -e "${experiment_root}/COMPLETED" ]]; then
  echo "Experiment is already complete: ${experiment_root}" >&2
  exit 2
fi
if [[ -e "${experiment_root}/RUNNING" ]]; then
  previous_pid="$(awk -F= '$1 == "pid" {print $2}' "${experiment_root}/RUNNING")"
  if [[ -n "${previous_pid}" ]] && kill -0 "${previous_pid}" 2>/dev/null; then
    echo "Experiment already has live process ${previous_pid}" >&2
    exit 3
  fi
  mv "${experiment_root}/RUNNING" \
    "${experiment_root}/RUNNING.stale.$(date +%Y%m%d-%H%M%S)"
fi
printf 'started_at=%s\npid=%s\n' "$(date --iso-8601=seconds)" "$$" \
  > "${experiment_root}/RUNNING"

on_exit() {
  rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    printf 'failed_at=%s\nexit_code=%s\n' "$(date --iso-8601=seconds)" "${rc}" \
      > "${experiment_root}/FAILED"
  fi
  rm -f "${experiment_root}/RUNNING"
  exit "${rc}"
}
trap on_exit EXIT

actual_source_sha="$(sha256sum "${source_pairs}" | awk '{print $1}')"
actual_dev_sha="$(sha256sum "${dev_pairs}" | awk '{print $1}')"
actual_parent_sha="$(sha256sum "${initial_parent}/model.safetensors" | awk '{print $1}')"
[[ "${actual_source_sha}" == "${expected_source_sha}" ]] || {
  echo "source SHA mismatch" >&2
  exit 4
}
[[ "${actual_dev_sha}" == "${expected_dev_sha}" ]] || {
  echo "dev SHA mismatch" >&2
  exit 4
}
[[ "${actual_parent_sha}" == "${expected_parent_sha}" ]] || {
  echo "parent SHA mismatch" >&2
  exit 4
}

if ps -eo pid=,args= \
  | grep -E '[t]orchrun|[r]un_qwen3_dual_train|[e]valuate_qwen3_pairs[.]py' \
  | grep -v 'grep -E' >/dev/null; then
  echo "Unknown overlapping training/evaluation process detected" >&2
  ps -eo pid=,args= \
    | grep -E '[t]orchrun|[r]un_qwen3_dual_train|[e]valuate_qwen3_pairs[.]py' >&2 || true
  exit 3
fi
while IFS=',' read -r gpu_index gpu_memory gpu_util; do
  gpu_memory="${gpu_memory//[^0-9]/}"
  gpu_util="${gpu_util//[^0-9]/}"
  if (( gpu_memory > 2048 || gpu_util > 20 )); then
    echo "GPU ${gpu_index} is not idle: ${gpu_memory} MiB / ${gpu_util}%" >&2
    exit 3
  fi
done < <(
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
)
free_bytes="$(df -B1 --output=avail /root/data | tail -n 1 | tr -d ' ')"
minimum_bytes="$(( minimum_free_gib * 1024 * 1024 * 1024 ))"
if (( free_bytes < minimum_bytes )); then
  echo "Insufficient free space: ${free_bytes} < ${minimum_bytes}" >&2
  exit 3
fi

cat > "${experiment_root}/config.env" <<EOF
experiment_id=${experiment_id}
git_head=$(git rev-parse HEAD)
source_pairs=${source_pairs}
source_sha256=${actual_source_sha}
dev_pairs=${dev_pairs}
dev_sha256=${actual_dev_sha}
initial_parent=${initial_parent}
initial_parent_sha256=${actual_parent_sha}
initial_parent_eval=${initial_parent_eval}
salt=${salt}
modulus=${modulus}
pool_size=${pool_size}
hard_k=${hard_k}
loop_steps=${loop_steps}
save_steps=${save_steps}
seed=${seed}
bootstrap_samples=${bootstrap_samples}
minimum_free_gib=${minimum_free_gib}
train_attempt_suffix=${train_attempt_suffix}
locked_test_used=false
agent_inference_used=false
external_data_used=false
EOF

run_loop() {
  local loop="$1"
  local bucket="$2"
  local parent_model="$3"
  local parent_eval="$4"
  local loop_data="${data_root}/loop${loop}"
  local loop_exp="${experiment_root}/loop${loop}"
  local control_run="${experiment_id}_loop${loop}_control${train_attempt_suffix}"
  local candidate_run="${experiment_id}_loop${loop}_candidate${train_attempt_suffix}"
  local control_output="${checkpoint_root}/${control_run}"
  local candidate_output="${checkpoint_root}/${candidate_run}"
  local parent_alias="${experiment_root}/model_aliases/Qwen3-loop${loop}-parent"
  mkdir -p "${loop_exp}/eval"

  mkdir -p "$(dirname "${parent_alias}")"
  if [[ -L "${parent_alias}" ]]; then
    [[ "$(readlink -f "${parent_alias}")" == "$(readlink -f "${parent_model}")" ]] || {
      echo "Parent alias points to a different model: ${parent_alias}" >&2
      exit 4
    }
  elif [[ -e "${parent_alias}" ]]; then
    echo "Parent alias path exists but is not a symlink: ${parent_alias}" >&2
    exit 4
  else
    ln -s "$(readlink -f "${parent_model}")" "${parent_alias}"
  fi

  echo "=== flywheel loop ${loop}: mine official negatives ==="
  date --iso-8601=seconds
  if [[ ! -e "${loop_data}/manifest.json" ]]; then
    .venv/bin/python solution/src/build_offline_flywheel_shard.py \
      --source "${source_pairs}" \
      --model "${parent_model}" \
      --output-dir "${loop_data}" \
      --salt "${salt}" \
      --modulus "${modulus}" \
      --bucket "${bucket}" \
      --pool-size "${pool_size}" \
      --hard-k "${hard_k}" \
      --batch-size 64 \
      --query-max-length 128 \
      --passage-max-length 512 \
      --device cuda:0
  fi
  .venv/bin/python solution/src/audit_offline_flywheel_shard.py \
    --source "${source_pairs}" \
    --shard-dir "${loop_data}" \
    --output "${loop_data}/audit.json"

  echo "=== flywheel loop ${loop}: uniform control ==="
  if [[ ! -e "${control_output}/COMPLETED" ]]; then
    PROJECT_ROOT="${project_root}" \
    RUN_ID="${control_run}" \
    MODEL_PATH="${parent_alias}" \
    TRAIN_DATA="${loop_data}/control.jsonl" \
    OUTPUT_DIR="${control_output}" \
    LOG_FILE="${log_root}/${control_run}.log" \
    CACHE_PATH="${project_root}/ccir/data/cache/${control_run}" \
    MAX_STEPS="${loop_steps}" \
    SAVE_STEPS="${save_steps}" \
    SAVE_TOTAL_LIMIT=1 \
    NUM_EPOCHS=1 \
    SEED="${seed}" \
    MASTER_PORT="$((29620 + loop * 2))" \
    RESUME_MODE=never \
      bash solution/scripts/run_qwen3_dual_train.sh
  fi

  echo "=== flywheel loop ${loop}: model-refreshed hard negatives ==="
  if [[ ! -e "${candidate_output}/COMPLETED" ]]; then
    PROJECT_ROOT="${project_root}" \
    RUN_ID="${candidate_run}" \
    MODEL_PATH="${parent_alias}" \
    TRAIN_DATA="${loop_data}/candidate.jsonl" \
    OUTPUT_DIR="${candidate_output}" \
    LOG_FILE="${log_root}/${candidate_run}.log" \
    CACHE_PATH="${project_root}/ccir/data/cache/${candidate_run}" \
    MAX_STEPS="${loop_steps}" \
    SAVE_STEPS="${save_steps}" \
    SAVE_TOTAL_LIMIT=1 \
    NUM_EPOCHS=1 \
    SEED="${seed}" \
    MASTER_PORT="$((29621 + loop * 2))" \
    RESUME_MODE=never \
      bash solution/scripts/run_qwen3_dual_train.sh
  fi

  if [[ ! -e "${loop_exp}/eval/control.json" ]]; then
    .venv/bin/python solution/src/evaluate_qwen3_pairs.py \
      --model "${control_output}" \
      --input "${dev_pairs}" \
      --output "${loop_exp}/eval/control.json" \
      --batch-size 16 \
      --max-length 512 \
      --device cuda:0
  fi
  if [[ ! -e "${loop_exp}/eval/candidate.json" ]]; then
    .venv/bin/python solution/src/evaluate_qwen3_pairs.py \
      --model "${candidate_output}" \
      --input "${dev_pairs}" \
      --output "${loop_exp}/eval/candidate.json" \
      --batch-size 16 \
      --max-length 512 \
      --device cuda:0
  fi
  if [[ ! -e "${loop_exp}/gate.json" ]]; then
    .venv/bin/python solution/src/evaluate_offline_flywheel_loop.py \
      --parent-eval "${parent_eval}" \
      --control-eval "${loop_exp}/eval/control.json" \
      --candidate-eval "${loop_exp}/eval/candidate.json" \
      --output "${loop_exp}/gate.json" \
      --bootstrap-samples "${bootstrap_samples}" \
      --seed "$((seed + loop * 10))"
  fi
  .venv/bin/python solution/src/analyze_offline_flywheel_effects.py \
    --parent-eval "${parent_eval}" \
    --control-eval "${loop_exp}/eval/control.json" \
    --candidate-eval "${loop_exp}/eval/candidate.json" \
    --output "${loop_exp}/effects.json"
  .venv/bin/python - "${loop_exp}/selection.json" "${loop_exp}/gate.json" \
    "${candidate_output}" "${loop_exp}/eval/candidate.json" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
selection, gate_path, candidate_model, candidate_eval = map(Path, sys.argv[1:])
gate = json.load(open(gate_path))
value = {
    "created_at": datetime.now().astimezone().isoformat(),
    "gate_passed": gate["gate"]["passed"],
    "next_parent_model": str(candidate_model) if gate["gate"]["passed"] else None,
    "next_parent_eval": str(candidate_eval) if gate["gate"]["passed"] else None,
    "locked_test_used": False,
    "full_epoch_started": False,
}
selection.write_text(json.dumps(value, indent=2) + "\n")
print(json.dumps(value))
PY
}

run_loop 1 0 "${initial_parent}" "${initial_parent_eval}"
loop1_passed="$(
  .venv/bin/python -c \
    "import json; print(str(json.load(open('${experiment_root}/loop1/gate.json'))['gate']['passed']).lower())"
)"
loops_completed=1
if [[ "${loop1_passed}" == "true" ]]; then
  run_loop 2 1 \
    "${checkpoint_root}/${experiment_id}_loop1_candidate${train_attempt_suffix}" \
    "${experiment_root}/loop1/eval/candidate.json"
  loops_completed=2
else
  printf 'stopped_at=%s\nreason=loop1_gate_failed\n' "$(date --iso-8601=seconds)" \
    > "${experiment_root}/STOPPED_BY_GATE"
fi

printf 'completed_at=%s\nloops_completed=%s\nfull_epoch_started=false\nlocked_test_used=false\n' \
  "$(date --iso-8601=seconds)" "${loops_completed}" \
  > "${experiment_root}/COMPLETED"
rm -f "${experiment_root}/FAILED" "${experiment_root}/RUNNING"
trap - EXIT
echo "Offline flywheel experiment complete"
