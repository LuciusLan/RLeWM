#!/usr/bin/env bash
set -euo pipefail
TASK_DIAGNOSTIC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PROJECT_ROOT="$(cd -- "$TASK_DIAGNOSTIC_DIR/../.." && pwd)"
cd "$TASK_PROJECT_ROOT"
source codes/cuda_env.sh
exec "${PYTHON:-/home/ubuntu/.venv/bin/python}" -X faulthandler -u codes/diagnostics/cli.py "$@"
