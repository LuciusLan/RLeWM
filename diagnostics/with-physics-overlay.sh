#!/usr/bin/env bash
# A fresh Python process must resolve the overlay before importing physics modules.
set -euo pipefail
TASK_DIAGNOSTIC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PROJECT_ROOT="$(cd -- "$TASK_DIAGNOSTIC_DIR/../.." && pwd)"
TASK_OVERLAY="${PHYSICS_OVERLAY:-$TASK_PROJECT_ROOT/.env-overlays/mj36}"
TASK_PYTHON="${PYTHON:-/home/ubuntu/.venv/bin/python}"
if [[ $# -eq 0 || ! -d "$TASK_OVERLAY/mujoco" || ! -d "$TASK_OVERLAY/dm_control" ]]; then
    printf 'Usage: bash codes/diagnostics/with-physics-overlay.sh COMMAND [ARGS...]\nPrepare the paired physics overlay first.\n' >&2
    exit 2
fi
export PYTHONPATH="$TASK_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
# Metadata only; no bindings/models/simulators are imported by this guard.
"$TASK_PYTHON" - "$TASK_OVERLAY" <<'PY'
from importlib.metadata import distribution
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
for name, version in [('mujoco', '3.6.0'), ('dm-control', '1.0.38')]:
    item = distribution(name)
    if item.version != version or not Path(item.locate_file('')).resolve().is_relative_to(root):
        raise SystemExit(f'Overlay resolution failed for {name}: {item.version} at {item.locate_file("")}')
PY
exec "$@"
