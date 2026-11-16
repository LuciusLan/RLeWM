#!/usr/bin/env bash
# User-run extraction; retains the original downloaded archive.
set -euo pipefail

P0_ARCHIVE_PATH="${P0_ARCHIVE_PATH:-/home/ubuntu/project/datasets/lewm_pusht/pusht_expert_train.h5.zst}"
P0_DATA_DIR="${P0_DATA_DIR:-/opt/dlami/nvme/ubuntu/lewm_pusht}"
P0_HDF5_PATH="$P0_DATA_DIR/pusht_expert_train.h5"
P0_PARTIAL="$P0_HDF5_PATH.partial"

if [[ ! -f "$P0_ARCHIVE_PATH" ]]; then
    printf 'Archive not found: %s\n' "$P0_ARCHIVE_PATH" >&2
    exit 1
fi
mkdir -p "$P0_DATA_DIR"
if [[ -f "$P0_HDF5_PATH" ]]; then
    printf 'Using existing HDF5: %s\n' "$P0_HDF5_PATH"
    exit 0
fi
if [[ -e "$P0_PARTIAL" ]]; then
    printf 'Partial extraction already exists: %s\n' "$P0_PARTIAL" >&2
    printf 'Inspect or remove that incomplete file before rerunning.\n' >&2
    exit 1
fi
trap 'rm -f -- "$P0_PARTIAL"' EXIT
printf 'Extracting the original dataset to NVMe: %s\n' "$P0_HDF5_PATH"
zstd --decompress --keep -o "$P0_PARTIAL" "$P0_ARCHIVE_PATH"
mv -- "$P0_PARTIAL" "$P0_HDF5_PATH"
trap - EXIT
printf 'Extraction complete: %s\n' "$P0_HDF5_PATH"
