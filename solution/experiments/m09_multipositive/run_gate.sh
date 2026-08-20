#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
experiment_id="${EXPERIMENT_ID:-m09_joint_v2_20260728}"
experiment_root="${project_root}/ccir/outputs/experiments/${experiment_id}"
candidate_data="${project_root}/ccir/data/experiments/m09_multipositive_v2/train.jsonl"
manifest="${project_root}/ccir/data/experiments/m09_multipositive_v2/manifest.json"
dev_data="${project_root}/ccir/data/experiments/early_stop_v1/dev.jsonl"
raw_eval_root="${project_root}/ccir/outputs/experiments/trajectory_search_idx_v2_20260726/eval/seed20260716/raw"
short_run_id="${experiment_id}_seed20260716_1000"
short_output="${project_root}/ccir/outputs/checkpoints/${short_run_id}"
short_eval="${experiment_root}/eval/seed20260716/multipositive"
gate_output="${experiment_root}/gate/seed20260716.json"
base_model="${project_root}/ccir/models/Qwen3-Embedding-0.6B"
expected_model_sha="0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
completed="${experiment_root}/COMPLETED"

cd "${project_root}"
export PYTHONPATH="${project_root}/FlagEmbedding:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${experiment_root}"

if [[ ! -s "${candidate_data}" || ! -s "${manifest}" ]]; then
  .venv/bin/python solution/experiments/m09_multipositive/prepare.py
fi

candidate_sha="$(
  .venv/bin/python - "${manifest}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["output_sha256"])
PY
)"
actual_candidate_sha="$(sha256sum "${candidate_data}" | awk '{print $1}')"
[[ "${candidate_sha}" == "${actual_candidate_sha}" ]]
[[ "$(sha256sum "${base_model}/model.safetensors" | awk '{print $1}')" == "${expected_model_sha}" ]]

cat > "${experiment_root}/config.env" <<EOF
experiment_id=${experiment_id}
git_commit=$(git rev-parse HEAD)
candidate_data=${candidate_data}
candidate_sha256=${candidate_sha}
base_model=${base_model}
base_model_sha256=${expected_model_sha}
seed=20260716
max_positives_per_query=3
short_steps=1000
bootstrap_samples=10000
locked_test_used=false
EOF

if [[ ! -e "${short_output}/COMPLETED" ]]; then
  LRAT_MAX_POSITIVES_PER_QUERY=3 \
  TRAIN_MODULE=solution.experiments.m09_multipositive.train_entry \
  RUN_ID="${short_run_id}" \
  MODEL_PATH="${base_model}" \
  MODEL_SHA256="${expected_model_sha}" \
  TRAIN_DATA="${candidate_data}" \
  DATA_SHA256="${candidate_sha}" \
  OUTPUT_DIR="${short_output}" \
  LOG_FILE="${project_root}/ccir/outputs/logs/${short_run_id}.log" \
  CACHE_PATH="${project_root}/ccir/data/cache/${short_run_id}" \
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
  bash solution/scripts/run_qwen3_dual_train.sh
fi

mkdir -p "${short_eval}" "$(dirname "${gate_output}")"
if [[ ! -s "${short_eval}/step_500.json" ]]; then
  .venv/bin/python solution/src/evaluate_qwen3_pairs.py \
    --model "${short_output}/checkpoint-500" \
    --input "${dev_data}" \
    --output "${short_eval}/step_500.json" \
    --batch-size 16 \
    --max-length 512 \
    --device cuda:0
fi
if [[ ! -s "${short_eval}/step_1000.json" ]]; then
  .venv/bin/python solution/src/evaluate_qwen3_pairs.py \
    --model "${short_output}" \
    --input "${dev_data}" \
    --output "${short_eval}/step_1000.json" \
    --batch-size 16 \
    --max-length 512 \
    --device cuda:0
fi

if [[ ! -s "${gate_output}" ]]; then
  .venv/bin/python solution/src/compare_paired_evals.py \
    --pair checkpoint-500 \
      "${raw_eval_root}/step_500.json" \
      "${short_eval}/step_500.json" \
    --pair final-1000 \
      "${raw_eval_root}/step_1000.json" \
      "${short_eval}/step_1000.json" \
    --bootstrap-samples 10000 \
    --seed 20260728 \
    --output "${gate_output}"
fi

if ! .venv/bin/python - "${gate_output}" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8"))["gate"]["passed"] is True else 1)
PY
then
  printf 'stopped_at=%s\nreason=first_seed_gate_failed\n' "$(date -Is)" > "${experiment_root}/STOPPED_BY_GATE"
  printf 'completed_at=%s\nresult=stopped_by_gate\n' "$(date -Is)" > "${completed}"
  exit 0
fi

printf 'authorized_at=%s\nreason=first_seed_gate_passed\n' "$(date -Is)" > "${experiment_root}/FULL_EPOCH_AUTHORIZED"
printf 'completed_at=%s\nresult=short_gate_passed\n' "$(date -Is)" > "${completed}"
