#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export PYTHONHASHSEED=0 MPLBACKEND=Agg
"${PYTHON:-python}" scripts/smoke_test.py
