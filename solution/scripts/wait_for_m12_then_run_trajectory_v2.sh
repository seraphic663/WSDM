#!/usr/bin/env bash
# Persistent trigger: wait for the current M12 diagnostic run, then launch v2.

set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
m12_root="${project_root}/ccir/outputs/experiments/m12_full/checkpoints"
target_root="${project_root}/ccir/outputs/experiments/trajectory_search_idx_v2_20260726"
log_file="${project_root}/ccir/outputs/logs/trajectory_v2_trigger.log"
running="${target_root}/TRIGGER_RUNNING"
failed="${target_root}/TRIGGER_FAILED"
completed="${target_root}/TRIGGER_COMPLETED"

cd "${project_root}"
mkdir -p "${target_root}" "$(dirname "${log_file}")"
if [ -e "${completed}" ]; then
  echo "Trigger already completed" >&2
  exit 2
fi
if [ -e "${running}" ]; then
  old_pid="$(awk -F= '$1=="pid"{print $2}' "${running}" 2>/dev/null || true)"
  if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Trigger already running as PID ${old_pid}" >&2
    exit 3
  fi
  mv "${running}" "${running}.stale.$(date +%Y%m%d-%H%M%S)"
fi
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
exec > >(tee -a "${log_file}") 2>&1

echo "$(date -Is): waiting for M12 terminal marker"
while true; do
  if [ -f "${m12_root}/FAILED" ]; then
    echo "M12 failed; refusing automatic launch"
    exit 4
  fi
  if [ -f "${m12_root}/COMPLETED" ]; then
    break
  fi
  sleep 60
done

echo "$(date -Is): M12 completed; waiting for its GPU processes to exit"
for _ in $(seq 1 30); do
  if ! ps -eo cmd | grep -E 'torchrun.*m12_full|FlagEmbedding.*train_m12.jsonl' | grep -v grep >/dev/null; then
    break
  fi
  sleep 10
done
if ps -eo cmd | grep -E 'torchrun.*m12_full|FlagEmbedding.*train_m12.jsonl' | grep -v grep >/dev/null; then
  echo "M12 processes did not exit within five minutes"
  exit 5
fi

sleep 30
bash solution/scripts/run_trajectory_search_idx_v2.sh
printf 'completed_at=%s\n' "$(date -Is)" > "${completed}"
echo "$(date -Is): trajectory v2 trigger completed"
