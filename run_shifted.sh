#!/usr/bin/env bash
set -euo pipefail
LEWM_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LEWM_SHIFT_ACTION="validation"
if [[ $# -gt 0 ]]; then
  case "$1" in
    status|check|smoke|validation|test|all)
      LEWM_SHIFT_ACTION="$1"
      shift
      ;;
  esac
fi
exec bash "$LEWM_PROJECT_ROOT/codes/run.sh" shifted "$LEWM_SHIFT_ACTION" "$@"
