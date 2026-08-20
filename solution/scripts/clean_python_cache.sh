#!/usr/bin/env bash
# Remove Python bytecode and common Python-generated caches from the XIR tree.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DRY_RUN=0
INCLUDE_GENERIC_CACHE=0

usage() {
    cat <<'EOF'
Usage: clean_python_cache.sh [OPTIONS]

Remove Python bytecode and common Python-generated caches.

Options:
  --dry-run             List what would be removed without deleting anything.
  --include-cache       Also remove directories named .cache.
  --root DIR            Clean DIR instead of the XIR repository root.
  -h, --help            Show this help.

The default cleanup removes *.pyc, *.pyo, __pycache__/, .pytest_cache/,
.mypy_cache/, .ruff_cache/, and .hypothesis/. All .git directories are
excluded. The default includes .download-env.
EOF
}

while (($# > 0)); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --include-cache)
            INCLUDE_GENERIC_CACHE=1
            shift
            ;;
        --root)
            if (($# < 2)); then
                echo "ERROR: --root requires a directory" >&2
                exit 2
            fi
            PROJECT_ROOT="$(cd -- "$2" && pwd)"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "ERROR: project root does not exist: $PROJECT_ROOT" >&2
    exit 1
fi

declare -a CACHE_FILES=()
declare -a CACHE_DIRS=()

while IFS= read -r -d '' path; do
    CACHE_FILES+=("$path")
done < <(
    find "$PROJECT_ROOT" \
        ! -path '*/.git' ! -path '*/.git/*' \
        -type f \( -name '*.pyc' -o -name '*.pyo' \) \
        -print0
)

while IFS= read -r -d '' path; do
    CACHE_DIRS+=("$path")
done < <(
    find "$PROJECT_ROOT" -depth \
        ! -path '*/.git' ! -path '*/.git/*' \
        -type d \( \
            -name '__pycache__' \
            -o -name '.pytest_cache' \
            -o -name '.mypy_cache' \
            -o -name '.ruff_cache' \
            -o -name '.hypothesis' \
        \) -print0
)

if (( INCLUDE_GENERIC_CACHE )); then
    while IFS= read -r -d '' path; do
        CACHE_DIRS+=("$path")
    done < <(
        find "$PROJECT_ROOT" -depth \
            ! -path '*/.git' ! -path '*/.git/*' \
            -type d -name '.cache' -print0
    )
fi

printf 'root: %s\n' "$PROJECT_ROOT"
printf 'bytecode files: %d\n' "${#CACHE_FILES[@]}"
printf 'cache directories: %d\n' "${#CACHE_DIRS[@]}"

if (( DRY_RUN )); then
    for path in "${CACHE_FILES[@]}"; do
        printf 'would remove file: %s\n' "$path"
    done
    for path in "${CACHE_DIRS[@]}"; do
        printf 'would remove directory: %s\n' "$path"
    done
    exit 0
fi

for path in "${CACHE_FILES[@]}"; do
    rm -f -- "$path"
done
for path in "${CACHE_DIRS[@]}"; do
    rm -rf -- "$path"
done

printf 'removed bytecode files: %d\n' "${#CACHE_FILES[@]}"
printf 'removed cache directories: %d\n' "${#CACHE_DIRS[@]}"
