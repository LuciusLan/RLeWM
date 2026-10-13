#!/usr/bin/env bash
set -euo pipefail
LEWM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$LEWM_ROOT/codes/cuda_env.sh"
exec "$LEWM_VENV/bin/python" "$LEWM_ROOT/codes/p1.py" "$@"
