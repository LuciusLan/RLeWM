#!/usr/bin/env bash
# Run the small P0 prediction baseline with virtual-environment CUDA libraries.
set -euo pipefail

P0_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
P0_VENV="${P0_VENV:-/home/ubuntu/.venv}"
LEWM_VENV="$P0_VENV"
source "$P0_ROOT/codes/cuda_env.sh"
exec "$P0_VENV/bin/python" "$P0_ROOT/codes/p0_smoke.py" --samples 30 --device cuda "$@"
