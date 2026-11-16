#!/usr/bin/env bash
set -euo pipefail
LEWM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$LEWM_ROOT/codes/cuda_env.sh"
cube_python="$PWD/.venvs/physics-mj36/bin/python"
exec "$cube_python" -X faulthandler -u "$LEWM_ROOT/codes/experiment.py" "$@"
