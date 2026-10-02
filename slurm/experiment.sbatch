#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/experiment-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source paper_pipeline/slurm/common.sh
prepare_run_root
"$PYTHON" scripts/run_paper.py run-case --run-root "$RUN_ROOT" --historical-root "$HISTORICAL_ROOT" --group "${1:?Supply an inventory group}" --index "${SLURM_ARRAY_TASK_ID:-0}"
