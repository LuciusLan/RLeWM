#!/usr/bin/env bash
set -euo pipefail
LEWM_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$LEWM_PROJECT_ROOT/codes/run.sh" shifted prepare "$@"
