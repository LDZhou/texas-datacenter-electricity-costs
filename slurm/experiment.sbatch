#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/experiment-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONHASHSEED=0
"${PYTHON:-python}" scripts/run_paper.py "$@" --index "${SLURM_ARRAY_TASK_ID:-0}" --threads "${SLURM_CPUS_PER_TASK:-16}"
