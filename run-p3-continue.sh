#!/usr/bin/env bash
# Continue recovered core validation with the same CUDA library setup as run.sh.
set -euo pipefail
P3_CONTINUE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$P3_CONTINUE_ROOT/codes/cuda_env.sh"
exec "$LEWM_VENV/bin/python" -X faulthandler -u "$P3_CONTINUE_ROOT/codes/continue-p3-core.py" "$@"
