#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
run_id="${RUN_ID:?Set RUN_ID to a unique experiment name}"
model_path="${MODEL_PATH:-${project_root}/ccir/models/Qwen3-Embedding-0.6B}"
train_data="${TRAIN_DATA:?Set TRAIN_DATA to a checked JSONL file}"
output_dir="${OUTPUT_DIR:-${project_root}/ccir/outputs/checkpoints/${run_id}}"
log_file="${LOG_FILE:-${project_root}/ccir/outputs/logs/${run_id}.log}"
cache_path="${CACHE_PATH:-${project_root}/ccir/data/cache/${run_id}}"

num_epochs="${NUM_EPOCHS:-1}"
max_steps="${MAX_STEPS:--1}"
save_steps="${SAVE_STEPS:-500}"
save_total_limit="${SAVE_TOTAL_LIMIT:-3}"
save_strategy="${SAVE_STRATEGY:-steps}"
train_group_size="${TRAIN_GROUP_SIZE:-6}"
query_max_len="${QUERY_MAX_LEN:-128}"
passage_max_len="${PASSAGE_MAX_LEN:-512}"
per_device_batch="${PER_DEVICE_BATCH:-1}"
gradient_accumulation="${GRADIENT_ACCUMULATION:-4}"
learning_rate="${LEARNING_RATE:-1e-6}"
seed="${SEED:-20260716}"
master_port="${MASTER_PORT:-29517}"
resume_mode="${RESUME_MODE:-auto}"
collision_free_queries="${COLLISION_FREE_QUERIES:-0}"
stop_after_step="${STOP_AFTER_STEP:-}"

instruction='Given a web search query, retrieve relevant passages that answer the query'
query_format=$'Instruct: {}\nQuery:{}'

cd "${project_root}"
export PYTHONPATH="${project_root}/FlagEmbedding${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

if [[ "${CUDA_VISIBLE_DEVICES}" != *,* ]]; then
  echo "CUDA_VISIBLE_DEVICES must expose exactly two GPUs" >&2
  exit 2
fi
if [[ "${resume_mode}" != "auto" && "${resume_mode}" != "never" ]]; then
  echo "RESUME_MODE must be auto or never" >&2
  exit 2
fi
if [[ "${save_strategy}" != "steps" && "${save_strategy}" != "no" ]]; then
  echo "SAVE_STRATEGY must be steps or no" >&2
  exit 2
fi
if [[ "${collision_free_queries}" != "0" && "${collision_free_queries}" != "1" ]]; then
  echo "COLLISION_FREE_QUERIES must be 0 or 1" >&2
  exit 2
fi
if [[ -n "${stop_after_step}" ]] && ! [[ "${stop_after_step}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STOP_AFTER_STEP must be a positive integer when set" >&2
  exit 2
fi
train_module="${TRAIN_MODULE:-FlagEmbedding.finetune.embedder.decoder_only.base}"
if [[ -n "${stop_after_step}" ]]; then
  if [[ -n "${TRAIN_MODULE:-}" ]]; then
    echo "TRAIN_MODULE cannot be combined with STOP_AFTER_STEP" >&2
    exit 2
  fi
  train_module="solution.src.run_segmented_early_stop_train"
elif [[ "${collision_free_queries}" == "1" ]]; then
  if [[ -n "${TRAIN_MODULE:-}" ]]; then
    echo "TRAIN_MODULE cannot be combined with COLLISION_FREE_QUERIES" >&2
    exit 2
  fi
  train_module="solution.src.run_query_collision_train"
fi
if [[ -e "${output_dir}/COMPLETED" ]]; then
  echo "Run is already complete: ${output_dir}" >&2
  exit 2
fi

mkdir -p "${output_dir}" "$(dirname "${log_file}")" "${cache_path}"

data_sha256="${DATA_SHA256:-$(sha256sum "${train_data}" | awk '{print $1}')}"
model_sha256="${MODEL_SHA256:-$(sha256sum "${model_path}/model.safetensors" | awk '{print $1}')}"
config_file="${output_dir}/run_config.env"
config_payload="$({
  printf 'model_path=%s\n' "${model_path}"
  printf 'model_sha256=%s\n' "${model_sha256}"
  printf 'train_data=%s\n' "${train_data}"
  printf 'data_sha256=%s\n' "${data_sha256}"
  printf 'world_size=2\n'
  printf 'train_group_size=%s\n' "${train_group_size}"
  printf 'query_max_len=%s\n' "${query_max_len}"
  printf 'passage_max_len=%s\n' "${passage_max_len}"
  printf 'per_device_batch=%s\n' "${per_device_batch}"
  printf 'gradient_accumulation=%s\n' "${gradient_accumulation}"
  printf 'gradient_checkpointing_use_reentrant=false\n'
  printf 'learning_rate=%s\n' "${learning_rate}"
  printf 'num_epochs=%s\n' "${num_epochs}"
  printf 'max_steps=%s\n' "${max_steps}"
  printf 'save_strategy=%s\n' "${save_strategy}"
  printf 'save_steps=%s\n' "${save_steps}"
  printf 'save_total_limit=%s\n' "${save_total_limit}"
  printf 'seed=%s\n' "${seed}"
  if [[ -n "${TRAIN_MODULE:-}" ]]; then
    printf 'train_module=%s\n' "${train_module}"
  fi
  if [[ -n "${LRAT_MAX_POSITIVES_PER_QUERY:-}" ]]; then
    printf 'max_positives_per_query=%s\n' "${LRAT_MAX_POSITIVES_PER_QUERY}"
  fi
  if [[ "${collision_free_queries}" == "1" ]]; then
    printf 'collision_free_queries=1\n'
  fi
  if [[ -n "${stop_after_step}" ]]; then
    printf 'segmented_early_stop=1\n'
  fi
  printf 'instruction=%s\n' "${instruction}"
  printf 'query_format=Instruct: {}\\nQuery:{}\n'
})"

if [[ -e "${config_file}" ]]; then
  if [[ "$(<"${config_file}")" != "${config_payload}" ]]; then
    echo "Refusing to resume with a different run configuration" >&2
    diff -u "${config_file}" <(printf '%s\n' "${config_payload}") || true
    exit 2
  fi
else
  printf '%s\n' "${config_payload}" > "${config_file}"
fi

resume_checkpoint=""
if [[ "${resume_mode}" == "auto" ]]; then
  resume_checkpoint="$(
    .venv/bin/python solution/src/find_valid_checkpoint.py \
      --show-rejected "${output_dir}"
  )"
fi

if [[ -z "${resume_checkpoint}" ]]; then
  unexpected="$(
    find "${output_dir}" -mindepth 1 -maxdepth 1 \
      ! -name run_config.env \
      ! -name RUNNING \
      ! -name 'FAILED*' \
      -print -quit
  )"
  if [[ -n "${unexpected}" ]]; then
    echo "No complete checkpoint found; refusing to overwrite ${unexpected}" >&2
    exit 2
  fi
fi

if [[ -e "${output_dir}/RUNNING" ]]; then
  previous_pid="$(awk -F= '$1 == "pid" {print $2}' "${output_dir}/RUNNING")"
  if [[ -n "${previous_pid}" ]] && kill -0 "${previous_pid}" 2>/dev/null; then
    echo "Refusing to overlap live run process ${previous_pid}" >&2
    exit 3
  fi
  mv "${output_dir}/RUNNING" \
    "${output_dir}/RUNNING.stale.$(date +%Y%m%d-%H%M%S)"
fi
if [[ -e "${output_dir}/FAILED" ]]; then
  mv "${output_dir}/FAILED" \
    "${output_dir}/FAILED.previous.$(date +%Y%m%d-%H%M%S)"
fi
printf 'started_at=%s\npid=%s\n' "$(date --iso-8601=seconds)" "$$" \
  > "${output_dir}/RUNNING"

on_exit() {
  rc=$?
  trap - EXIT
  if [[ ${rc} -eq 0 ]]; then
    if [[ -e "${output_dir}/PAUSED_AT_STEP" ]]; then
      echo "Training intentionally paused for dev evaluation"
    else
      printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" \
        > "${output_dir}/COMPLETED"
    fi
    rm -f "${output_dir}/RUNNING"
  else
    printf 'failed_at=%s\nexit_code=%s\n' \
      "$(date --iso-8601=seconds)" "${rc}" > "${output_dir}/FAILED"
    rm -f "${output_dir}/RUNNING"
  fi
  exit "${rc}"
}
trap on_exit EXIT

exec > >(tee -a "${log_file}") 2>&1

echo "=== dual-GPU Qwen3 training attempt ==="
date --iso-8601=seconds
echo "run_id=${run_id}"
echo "output_dir=${output_dir}"
echo "resume_checkpoint=${resume_checkpoint:-none}"
echo "git_commit=$(git rev-parse HEAD)"
cat "${config_file}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

while IFS=',' read -r gpu_index gpu_memory gpu_util; do
  gpu_memory="${gpu_memory//[^0-9]/}"
  gpu_util="${gpu_util//[^0-9]/}"
  if (( gpu_memory > 2048 || gpu_util > 20 )); then
    echo "GPU ${gpu_index} is not idle enough: ${gpu_memory} MiB, ${gpu_util}%" >&2
    exit 3
  fi
done < <(
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
)

train_args=(
  --model_name_or_path "${model_path}"
  --train_data "${train_data}"
  --cache_path "${cache_path}"
  --train_group_size "${train_group_size}"
  --query_max_len "${query_max_len}"
  --passage_max_len "${passage_max_len}"
  --pad_to_multiple_of 8
  --query_instruction_for_retrieval "${instruction}"
  --query_instruction_format "${query_format}"
  --knowledge_distillation False
  --output_dir "${output_dir}"
  --learning_rate "${learning_rate}"
  --bf16
  --num_train_epochs "${num_epochs}"
  --per_device_train_batch_size "${per_device_batch}"
  --gradient_accumulation_steps "${gradient_accumulation}"
  --dataloader_drop_last True
  --warmup_ratio 0.1
  --gradient_checkpointing
  --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  --logging_steps 1
  --save_strategy "${save_strategy}"
  --save_steps "${save_steps}"
  --save_total_limit "${save_total_limit}"
  --negatives_cross_device
  --temperature 0.02
  --sentence_pooling_method last_token
  --normalize_embeddings True
  --seed "${seed}"
  --data_seed "${seed}"
  --ddp_find_unused_parameters False
  --report_to none
)
if (( max_steps > 0 )); then
  train_args+=(--max_steps "${max_steps}")
fi
if [[ -n "${resume_checkpoint}" ]]; then
  train_args+=(--resume_from_checkpoint "${resume_checkpoint}")
fi

printf 'command:'
printf ' %q' \
  .venv/bin/torchrun --standalone --nproc_per_node=2 \
  --master_port "${master_port}" \
  -m "${train_module}" \
  "${train_args[@]}"
printf '\n'

.venv/bin/torchrun --standalone --nproc_per_node=2 \
  --master_port "${master_port}" \
  -m "${train_module}" \
  "${train_args[@]}"
