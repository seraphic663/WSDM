#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
run_id="${RUN_ID:?Set RUN_ID to the completed model-b training run}"
model_a="${MODEL_A:?Set MODEL_A to the first model directory}"
model_b="${MODEL_B:-${project_root}/ccir/outputs/checkpoints/${run_id}}"
grid_prefix="${GRID_PREFIX:-qwen3_epoch1_epoch2_interp}"
alphas="${ALPHAS:-0.25 0.50 0.75}"

cd "${project_root}"
if [[ ! -f "ccir/outputs/checkpoints/${run_id}/COMPLETED" ]]; then
  echo "Model-b run is not complete: ${run_id}" >&2
  exit 2
fi

results=()
for alpha in ${alphas}; do
  suffix="$(printf '%s' "${alpha}" | tr -d '.' | sed 's/^0*//')"
  [[ -n "${suffix}" ]] || suffix=0
  model_output="${project_root}/ccir/models/${grid_prefix}_a${suffix}"
  eval_id="${grid_prefix}_a${suffix}_dev500"
  eval_output="${project_root}/ccir/outputs/eval/${eval_id}.json"
  eval_log="${project_root}/ccir/outputs/logs/${eval_id}.log"

  .venv/bin/python solution/src/interpolate_safetensors.py \
    --model-a "${model_a}" \
    --model-b "${model_b}" \
    --output "${model_output}" \
    --alpha "${alpha}"

  env \
    RUN_ID="${run_id}" \
    EVAL_ID="${eval_id}" \
    MODEL_PATH="${model_output}" \
    EVAL_OUTPUT="${eval_output}" \
    EVAL_LOG="${eval_log}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    bash solution/scripts/eval_completed_qwen3_run.sh >/dev/null
  results+=("${eval_output}")
done

.venv/bin/python - "${results[@]}" <<'PY'
import json
import sys

rows = []
for path in sys.argv[1:]:
    result = json.load(open(path, encoding="utf-8"))
    rows.append((result["metrics"]["mrr"], path, result["metrics"]))
for _, path, metrics in sorted(rows, reverse=True):
    print(json.dumps({"path": path, "metrics": metrics}, ensure_ascii=False))
PY
