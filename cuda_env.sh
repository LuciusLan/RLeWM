#!/usr/bin/env bash
# Source before launching Python; Torch loads these libraries at import time.
LEWM_VENV="${LEWM_VENV:-/home/ubuntu/.venv}"
LEWM_LIBRARY_PATH=""
for LEWM_LIBRARY_DIR in "$LEWM_VENV"/lib/python*/site-packages/nvidia/*/lib; do
    if [[ -d "$LEWM_LIBRARY_DIR" ]]; then
        LEWM_LIBRARY_PATH="${LEWM_LIBRARY_PATH:+$LEWM_LIBRARY_PATH:}$LEWM_LIBRARY_DIR"
    fi
done
if [[ -n "$LEWM_LIBRARY_PATH" ]]; then
    export LD_LIBRARY_PATH="$LEWM_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
