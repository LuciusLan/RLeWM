#!/usr/bin/env bash
# Sequential core experiments; each process releases GPU resources before the next.
set -euo pipefail
core_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
core_datasets=(pusht tworooms reacher cube)
core_root="artifacts/evaluation/p3-core"
core_phase="validation"
core_options=()
while (($#)); do
    case "$1" in
        --datasets)
            shift
            core_datasets=()
            while (($#)) && [[ "$1" != --* ]]; do
                core_datasets+=("$1")
                shift
            done
            ((${#core_datasets[@]})) || { echo "--datasets needs at least one task" >&2; exit 2; }
            ;;
        --output-root)
            (($# >= 2)) || { echo "--output-root needs a directory" >&2; exit 2; }
            core_root="$2"; shift 2 ;;
        --phase)
            (($# >= 2)) || { echo "--phase needs validation or test" >&2; exit 2; }
            core_phase="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash codes/run-p3-core.sh [--datasets pusht tworooms reacher cube] [--output-root DIR] [--phase validation|test] [p3 core options]"
            echo "Default: 4 cells (Gaussian/retrieval init x Frozen/checkpoint), one seed, J2/4/8/16, N300/E30, 25 controls, 8h cap per task."
            echo "Add --mid-search to restore the historical six-cell comparison."
            echo "--checkpoints J ... sets adaptation times; --adaptation-schedule pooled-depth|sequential-depth."
            echo "--updates U is per checkpoint by default, or per selected depth with sequential-depth."
            echo "For 50-control real MPC, add --control-budget 50 with a separate output root."
            echo "Test reads DIR/TASK/locks/core-validation.json; create each lock from complete validation first."
            echo "Add --resume to continue verified clean runtime-cap stops without repeating completed queries."
            exit 0 ;;
        *) core_options+=("$1"); shift ;;
    esac
done
[[ "$core_phase" == validation || "$core_phase" == test ]] || { echo "Invalid --phase" >&2; exit 2; }
for core_task in "${core_datasets[@]}"; do
    case "$core_task" in pusht|tworooms|reacher|cube) ;; *) echo "Unknown task: $core_task" >&2; exit 2 ;; esac
done
cd -- "$core_repo"
for core_task in "${core_datasets[@]}"; do
    if [[ "$core_phase" == validation ]]; then
        bash codes/run.sh p3 core --dataset "$core_task" --output-root "$core_root/$core_task" "${core_options[@]}"
    else
        bash codes/run.sh p3 core --phase test --locked-config "$core_root/$core_task/locks/core-validation.json" "${core_options[@]}"
    fi
done
