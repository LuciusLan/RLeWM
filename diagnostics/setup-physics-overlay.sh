#!/usr/bin/env bash
# Install two paired physics wheels into a separate target, never the active env.
set -euo pipefail
TASK_DIAGNOSTIC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PROJECT_ROOT="$(cd -- "$TASK_DIAGNOSTIC_DIR/../.." && pwd)"
TASK_PYTHON="${PYTHON:-/home/ubuntu/.venv/bin/python}"
TASK_OVERLAY="${PHYSICS_OVERLAY:-$TASK_PROJECT_ROOT/.env-overlays/mj36}"
if [[ "${1:-}" == "--dry-run" ]]; then
    printf 'Interpreter: %s\nSeparate package target: %s\n' "$TASK_PYTHON" "$TASK_OVERLAY"
    printf 'Install only mujoco==3.6.0 and dm-control==1.0.38 with --no-deps. No active packages will be removed.\n'
    exit 0
fi
if [[ $# -ne 0 ]]; then
    printf 'Usage: bash codes/diagnostics/setup-physics-overlay.sh [--dry-run]\n' >&2
    exit 2
fi
if [[ -e "$TASK_OVERLAY" || -e "$TASK_OVERLAY.installing" ]]; then
    printf 'Preserved existing/partial target; choose another PHYSICS_OVERLAY: %s\n' "$TASK_OVERLAY" >&2
    exit 1
fi
mkdir -p "$TASK_OVERLAY.installing"
"$TASK_PYTHON" -m pip install --no-deps --only-binary=:all: \
    --target "$TASK_OVERLAY.installing" --report "$TASK_OVERLAY.installing/pip-report.json" \
    'mujoco==3.6.0' 'dm-control==1.0.38'
mv "$TASK_OVERLAY.installing" "$TASK_OVERLAY"
printf 'Physics overlay installed: %s\nRun the documented environment smoke check before collection.\n' "$TASK_OVERLAY"
