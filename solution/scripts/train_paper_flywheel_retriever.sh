#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
current_model="${1:?current model directory required}"
train_data="${2:?training pairs JSONL required}"
output_dir="${3:?output directory required}"
log_path="${4:?log path required}"
config="${PAPER_FLYWHEEL_CONFIG:-${project_root}/solution/configs/paper_flywheel_v1.json}"
profile="${PAPER_TRAINING_PROFILE:-paper_reported_batch32}"
smoke_max_steps="${PAPER_SMOKE_MAX_STEPS:-}"

cd "${project_root}"
for path in "${current_model}/model.safetensors" "${train_data}" "${config}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing required input: ${path}" >&2
    exit 2
  fi
done
case "$(realpath -m "${output_dir}")" in
  "$(realpath ccir/research/paper_flywheel_v1)"/*) ;;
  *) echo "output must stay under ccir/research/paper_flywheel_v1" >&2; exit 2 ;;
esac
case "$(realpath -m "${train_data}")" in
  "$(realpath ccir/research/paper_flywheel_v1)"/*) ;;
  *) echo "training data must stay under the paper research root" >&2; exit 2 ;;
esac
resolved_model="$(realpath "${current_model}")"
resolved_base="$(realpath ccir/models/Qwen3-Embedding-0.6B)"
resolved_research="$(realpath -m ccir/research/paper_flywheel_v1)"
case "${resolved_model}" in
  "${resolved_base}"|"${resolved_research}"/*) ;;
  *) echo "current retriever is outside the paper lineage boundary" >&2; exit 2 ;;
esac
case "$(realpath -m "${log_path}")" in
  "${resolved_research}"/*) ;;
  *) echo "training log must stay under the paper research root" >&2; exit 2 ;;
esac
smoke_only=false
if [[ -n "${smoke_max_steps}" ]]; then
  if ! [[ "${smoke_max_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PAPER_SMOKE_MAX_STEPS must be a positive integer" >&2
    exit 2
  fi
  case "$(realpath -m "${output_dir}")" in
    "${resolved_research}/smoke/"*) ;;
    *)
      echo "smoke output must stay under the dedicated research smoke root" >&2
      exit 2
      ;;
  esac
  smoke_only=true
fi
if [[ -e "${output_dir}/RUNNING" ]]; then
  echo "training already marked RUNNING: ${output_dir}" >&2
  exit 2
fi
if [[ -e "${output_dir}/COMPLETED" ]]; then
  echo "training already complete: ${output_dir}" >&2
  exit 0
fi

mapfile -t training_values < <(
  "${project_root}/.venv/bin/python" - "${config}" "${profile}" <<'PY'
import json,sys
config=json.load(open(sys.argv[1]))
contract=config["paper_contract"]
profile_name=sys.argv[2]
profiles=config["runtime"]["training_profiles"]
if profile_name not in profiles:
    raise SystemExit(f"unknown training profile: {profile_name}")
profile=profiles[profile_name]
print(contract["train_group_size"])
print(contract["query_max_tokens"])
print(contract["passage_max_tokens"])
print(contract["train_epochs_per_loop"])
print(contract["learning_rate"])
print(contract["temperature"])
print(profile["world_size"])
print(profile["per_device_train_batch_size"])
print(profile["gradient_accumulation_steps"])
print(str(profile["exact_in_batch_negative_pool"]).lower())
PY
)
if [[ "${#training_values[@]}" -ne 10 ]]; then
  echo "failed to parse paper training contract" >&2
  exit 2
fi
train_group_size="${training_values[0]}"
query_max_len="${training_values[1]}"
passage_max_len="${training_values[2]}"
num_epochs="${training_values[3]}"
learning_rate="${training_values[4]}"
temperature="${training_values[5]}"
world_size="${training_values[6]}"
per_device_batch="${training_values[7]}"
gradient_accumulation="${training_values[8]}"
exact_negative_pool="${training_values[9]}"
if [[ "${world_size}" -ne 2 ]]; then
  echo "this server entrypoint requires a two-GPU profile" >&2
  exit 2
fi
if [[ "${profile}" != "paper_reported_batch32" && "${ALLOW_SCALED_PAPER_PROFILE:-0}" != "1" ]]; then
  echo "scaled profile refused: set ALLOW_SCALED_PAPER_PROFILE=1" >&2
  exit 2
fi
if ps -eo args= | grep -E '[v]llm[[:space:]]+serve|[t]orchrun' >/dev/null; then
  echo "overlapping vLLM or torchrun process detected" >&2
  ps -eo pid=,args= | grep -E '[v]llm[[:space:]]+serve|[t]orchrun' >&2 || true
  exit 3
fi
while IFS=',' read -r gpu memory utilization; do
  memory="${memory//[^0-9]/}"
  utilization="${utilization//[^0-9]/}"
  if (( memory > 2048 || utilization > 20 )); then
    echo "GPU ${gpu} is not idle: ${memory} MiB / ${utilization}%" >&2
    exit 3
  fi
done < <(
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
)
free_bytes="$(df -B1 --output=avail /root/data | tail -n 1 | tr -d ' ')"
minimum_bytes="$((150 * 1024 * 1024 * 1024))"
if (( free_bytes < minimum_bytes )); then
  echo "paper training requires at least 150 GiB free" >&2
  exit 3
fi

mkdir -p "${output_dir}" "$(dirname "${log_path}")"
"${project_root}/.venv/bin/python" - \
  "${config}" "${profile}" "${current_model}" "${train_data}" "${output_dir}" \
  "${smoke_only}" "${smoke_max_steps}" <<'PY'
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
config,profile,model,data,output=map(Path,sys.argv[1:6])
smoke_only=sys.argv[6].lower()=="true"
smoke_max_steps=int(sys.argv[7]) if sys.argv[7] else None
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""): h.update(chunk)
    return h.hexdigest()
cfg=json.load(config.open())
contract=cfg["paper_contract"]
training=cfg["runtime"]["training_profiles"][str(profile)]
value={
    "created_at":datetime.now().astimezone().isoformat(),
    "research_only":True,
    "competition_submission_eligible":False,
    "smoke_only":smoke_only,
    "smoke_max_steps":smoke_max_steps,
    "runtime_weight_reduction":"paper_mean",
    "config":str(config.resolve()),
    "config_sha256":sha(config),
    "profile":str(profile),
    "profile_contract":training,
    "paper_contract":{
        key:contract[key] for key in (
            "train_epochs_per_loop","learning_rate","temperature",
            "weighted_loss_reduction",
            "reported_train_batch_size","train_group_size",
            "query_max_tokens","passage_max_tokens",
        )
    },
    "current_retriever":str(model.resolve()),
    "current_retriever_sha256":sha(model/"model.safetensors"),
    "training_pairs":str(data.resolve()),
    "training_pairs_bytes":data.stat().st_size,
    "training_pairs_sha256":sha(data),
    "output":str(output.resolve()),
}
(output/"TRAINING_CONTRACT.json").write_text(json.dumps(value,indent=2)+"\n")
PY
printf 'started_at=%s\n' "$(date -Is)" > "${output_dir}/RUNNING"
rm -f "${output_dir}/FAILED"
save_strategy=epoch
logging_steps=10
smoke_args=()
if [[ "${smoke_only}" == true ]]; then
  save_strategy=no
  logging_steps=1
  smoke_args+=(--max_steps "${smoke_max_steps}")
fi
set +e
PATH="${project_root}/.venv/bin:${PATH}" \
PYTHONPATH="${project_root}/FlagEmbedding${PYTHONPATH:+:${PYTHONPATH}}" \
LRAT_WEIGHT_REDUCTION=paper_mean \
OMP_NUM_THREADS=4 "${project_root}/.venv/bin/torchrun" \
  --standalone --nproc_per_node="${world_size}" \
  -m FlagEmbedding.finetune.embedder.decoder_only.base \
  --model_name_or_path "${current_model}" \
  --train_data "${train_data}" \
  --cache_path "${output_dir}/dataset_cache" \
  --train_group_size "${train_group_size}" \
  --query_max_len "${query_max_len}" \
  --passage_max_len "${passage_max_len}" \
  --pad_to_multiple_of 8 \
  --query_instruction_for_retrieval "Given a web search query, retrieve relevant passages that answer the query" \
  --query_instruction_format $'Instruct: {}\nQuery:{}' \
  --knowledge_distillation False \
  --same_dataset_within_batch True \
  --small_threshold 0 \
  --drop_threshold 0 \
  --output_dir "${output_dir}" \
  --overwrite_output_dir \
  --learning_rate "${learning_rate}" \
  --bf16 \
  --num_train_epochs "${num_epochs}" \
  --per_device_train_batch_size "${per_device_batch}" \
  --gradient_accumulation_steps "${gradient_accumulation}" \
  --dataloader_drop_last True \
  --warmup_ratio 0.1 \
  --gradient_checkpointing \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --deepspeed "${project_root}/FlagEmbedding/examples/finetune/ds_stage0.json" \
  --logging_steps "${logging_steps}" \
  --save_strategy "${save_strategy}" \
  --save_total_limit 2 \
  --negatives_cross_device \
  --temperature "${temperature}" \
  --sentence_pooling_method last_token \
  --normalize_embeddings True \
  --kd_loss_type m3_kd_loss \
  --seed 2025 \
  --data_seed 2025 \
  --ddp_find_unused_parameters False \
  --report_to none \
  "${smoke_args[@]}" \
  2>&1 | tee "${log_path}"
status=${PIPESTATUS[0]}
set -e
rm -f "${output_dir}/RUNNING"
if [[ "${status}" -eq 0 ]]; then
  if [[ ! -s "${output_dir}/model.safetensors" ]]; then
    printf 'failed_at=%s\nexit_code=4\nreason=missing_final_model\n' \
      "$(date -Is)" > "${output_dir}/FAILED"
    exit 4
  fi
  printf 'completed_at=%s\nprofile=%s\nexact_in_batch_negative_pool=%s\nweight_reduction=paper_mean\nsmoke_only=%s\nsmoke_max_steps=%s\n' \
    "$(date -Is)" "${profile}" "${exact_negative_pool}" \
    "${smoke_only}" "${smoke_max_steps}" \
    > "${output_dir}/COMPLETED"
else
  printf 'failed_at=%s\nexit_code=%s\n' "$(date -Is)" "${status}" > "${output_dir}/FAILED"
fi
exit "${status}"
