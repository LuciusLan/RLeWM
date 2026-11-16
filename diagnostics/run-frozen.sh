#!/usr/bin/env bash
# Separate processes keep each task's verified archived numerical core isolated.
set -euo pipefail
TASK_DIAGNOSTIC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for TASK_ARGUMENT in "$@"; do
    case "$TASK_ARGUMENT" in
        --output|--output=*|--dataset|--dataset=*|--target-replay|--target-replay=*)
            printf 'Use run.sh frozen for a custom output, dataset, or target replay.\n' >&2
            exit 2
            ;;
    esac
done
for TASK_DATASET in pusht cube; do
    bash "$TASK_DIAGNOSTIC_DIR/run.sh" frozen --dataset "$TASK_DATASET" "$@"
done
