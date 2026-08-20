#!/usr/bin/env bash

early_stop_model_path() {
  local output_dir="$1"
  local target="$2"
  local max_steps="$3"
  if (( target == max_steps )); then
    printf '%s\n' "${output_dir}"
  else
    printf '%s\n' "${output_dir}/checkpoint-${target}"
  fi
}

early_stop_model_ready() {
  local output_dir="$1"
  local target="$2"
  local max_steps="$3"
  local model_path
  model_path="$(early_stop_model_path "${output_dir}" "${target}" "${max_steps}")"
  if (( target == max_steps )); then
    [[ -f "${model_path}/model.safetensors" ]] \
      && [[ -e "${output_dir}/COMPLETED" ]] \
      && [[ ! -e "${output_dir}/PAUSED_AT_STEP" ]]
  else
    [[ -f "${model_path}/model.safetensors" ]]
  fi
}
