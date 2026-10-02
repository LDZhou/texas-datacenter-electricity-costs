#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/results/paper}"
HISTORICAL_ROOT="${HISTORICAL_ROOT:-$REPO_ROOT}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
export REPO_ROOT RUN_ROOT HISTORICAL_ROOT PYTHON
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0 MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1
prepare_run_root() {
  mkdir -p "$RUN_ROOT"/{configs,logs,starting_networks,profiles,results,manifests}
}
# This workflow requires CPU resources only. Select an allowed partition at submission.
case "${SLURM_JOB_PARTITION:-ampere}" in
  ampere|eecs|share|preempt) ;;
  *) printf 'Unsupported partition: %s\n' "$SLURM_JOB_PARTITION" >&2; return 1 2>/dev/null || exit 1 ;;
esac
