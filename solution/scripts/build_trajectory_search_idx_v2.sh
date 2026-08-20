#!/usr/bin/env bash
# Build and independently audit the real 94,113-row search-index arm.

set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
source_data="${project_root}/ccir/data/experiments/early_stop_v1/train.jsonl"
provenance="${project_root}/ccir/data/experiments/trajectory_provenance_v1/provenance.jsonl"
config="${project_root}/solution/configs/trajectory_search_idx_v2.json"
output_root="${project_root}/ccir/data/experiments/trajectory_search_idx_v2"
status_root="${project_root}/ccir/outputs/experiments/trajectory_search_idx_v2_data_build"
expected_source_sha="158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9"
expected_provenance_sha="924d373cffbc8f229c326c1de02e7e223c87109e57c32c1f2be29ea2e65f0b15"

cd "${project_root}"
mkdir -p "${status_root}"
running="${status_root}/RUNNING"
failed="${status_root}/FAILED"
completed="${status_root}/COMPLETED"

if [ -s "${completed}" ]; then
  echo "Data build already completed: ${completed}"
  exit 0
fi
if [ -e "${output_root}" ]; then
  echo "Output root exists without completed marker: ${output_root}" >&2
  exit 2
fi

actual_source_sha="$(sha256sum "${source_data}" | awk '{print $1}')"
actual_provenance_sha="$(sha256sum "${provenance}" | awk '{print $1}')"
if [ "${actual_source_sha}" != "${expected_source_sha}" ]; then
  echo "Source SHA mismatch: ${actual_source_sha}" >&2
  exit 2
fi
if [ "${actual_provenance_sha}" != "${expected_provenance_sha}" ]; then
  echo "Provenance SHA mismatch: ${actual_provenance_sha}" >&2
  exit 2
fi

printf 'started_at=%s\npid=%s\n' "$(date -Is)" "$$" > "${running}"
rm -f "${failed}"
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
exec > >(tee -a "${status_root}/build.log") 2>&1

echo "=== trajectory search-index v2 data build ==="
date -Is
echo "git_head=$(git rev-parse HEAD)"
echo "source_sha=${actual_source_sha}"
echo "provenance_sha=${actual_provenance_sha}"

PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  .venv/bin/python -m solution.src.build_trajectory_search_idx_arm \
  --pairs "${source_data}" \
  --provenance "${provenance}" \
  --config "${config}" \
  --output-root "${output_root}" \
  --expected-rows 94113

PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  .venv/bin/python -m solution.src.audit_trajectory_search_idx_arm \
  --source "${source_data}" \
  --output "${output_root}/train_search_idx_soft.jsonl" \
  --provenance "${provenance}" \
  --manifest "${output_root}/manifest.json" \
  --report "${output_root}/audit.json"

printf 'completed_at=%s\noutput_sha256=%s\n' \
  "$(date -Is)" \
  "$(sha256sum "${output_root}/train_search_idx_soft.jsonl" | awk '{print $1}')" \
  > "${completed}"
echo "Data build completed."
