#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
model_path="${MODEL_PATH:-${project_root}/ccir/models/Qwen3-Embedding-0.6B}"
train_data="${TRAIN_DATA:-${project_root}/ccir/data/smoke/train64.jsonl}"
output_dir="${OUTPUT_DIR:-${project_root}/ccir/outputs/checkpoints/qwen3_smoke_10step}"
cache_path="${CACHE_PATH:-${project_root}/ccir/data/cache/train64}"
max_steps="${MAX_STEPS:-10}"

cd "${project_root}"
export PYTHONPATH="${project_root}/FlagEmbedding${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TOKENIZERS_PARALLELISM=false

query_format=$'Instruct: {}\nQuery:{}'

.venv/bin/python -m FlagEmbedding.finetune.embedder.decoder_only.base \
  --model_name_or_path "${model_path}" \
  --train_data "${train_data}" \
  --cache_path "${cache_path}" \
  --train_group_size 6 \
  --query_max_len 128 \
  --passage_max_len 512 \
  --pad_to_multiple_of 8 \
  --query_instruction_for_retrieval 'Given a web search query, retrieve relevant passages that answer the query' \
  --query_instruction_format "${query_format}" \
  --knowledge_distillation False \
  --output_dir "${output_dir}" \
  --overwrite_output_dir \
  --learning_rate 1e-6 \
  --bf16 \
  --max_steps "${max_steps}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --dataloader_drop_last True \
  --warmup_ratio 0.1 \
  --gradient_checkpointing \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "${max_steps}" \
  --save_total_limit 1 \
  --temperature 0.02 \
  --sentence_pooling_method last_token \
  --normalize_embeddings True \
  --seed 20260716 \
  --data_seed 20260716 \
  --report_to none
