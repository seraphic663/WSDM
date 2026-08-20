#!/usr/bin/env bash
# Compatibility entrypoint: the flywheel name now means the paper-style
# Agent -> trajectory -> LRAT pairs -> retriever -> new trajectory loop.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/run_paper_flywheel_v1.sh" "$@"
