#!/usr/bin/env bash
# Persist only predeclared diagnostic checkpoints from a long CCIR training run.
#
# Usage:
#   bash solution/experiments/backup_sparse_checkpoints.sh <experiment_id>
#
# Optional environment variables:
#   KEEP_STEPS     Space-separated checkpoint steps.
#   POLL_SECONDS   Poll interval, default 60.
#   MIN_FREE_GB    Stop copying when /root/data has less free space, default 60.

set -euo pipefail

EXPERIMENT_ID="${1:?Usage: backup_sparse_checkpoints.sh <experiment_id>}"
SERVER_ROOT="/root/data/LRAT"
SOURCE_DIR="${SERVER_ROOT}/ccir/outputs/experiments/${EXPERIMENT_ID}/checkpoints"
BACKUP_DIR="${SERVER_ROOT}/ccir/outputs/experiments/${EXPERIMENT_ID}/checkpoint_backups"
LOG_FILE="${BACKUP_DIR}/backup.log"
KEEP_STEPS="${KEEP_STEPS:-500 1000 2000 4000 6000 8000 10000 11500}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MIN_FREE_GB="${MIN_FREE_GB:-60}"

cd "${SERVER_ROOT}"
mkdir -p "${BACKUP_DIR}"

log() {
    printf '%s: %s\n' "$(date -Is)" "$*" >> "${LOG_FILE}"
}

free_gb() {
    df --output=avail -BG /root/data | tail -n 1 | tr -dc '0-9'
}

copy_checkpoint() {
    local step="$1"
    local name="checkpoint-${step}"
    local source="${SOURCE_DIR}/${name}"
    local target="${BACKUP_DIR}/${name}"
    local staging="${BACKUP_DIR}/.${name}.copying.$$"

    [ -d "${source}" ] || return 0
    [ -f "${source}/model.safetensors" ] || return 0
    [ -f "${source}/trainer_state.json" ] || return 0
    [ ! -e "${target}" ] || return 0

    local available
    available="$(free_gb)"
    if [ "${available}" -lt "${MIN_FREE_GB}" ]; then
        log "STOPPED_LOW_DISK available_gb=${available} threshold_gb=${MIN_FREE_GB}"
        exit 3
    fi

    rm -rf -- "${staging}"
    cp -a -- "${source}" "${staging}"
    mv -- "${staging}" "${target}"
    log "Backed up ${name}; available_gb=$(free_gb)"
}

log "Sparse backup started; keep_steps=${KEEP_STEPS}; min_free_gb=${MIN_FREE_GB}"

while true; do
    for step in ${KEEP_STEPS}; do
        copy_checkpoint "${step}"
    done

    if [ -f "${SOURCE_DIR}/COMPLETED" ] || [ -f "${SOURCE_DIR}/FAILED" ]; then
        log "Training terminal marker detected; sparse backup complete"
        break
    fi

    sleep "${POLL_SECONDS}"
done
