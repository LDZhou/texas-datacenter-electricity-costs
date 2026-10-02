#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/base-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source paper_pipeline/slurm/common.sh
year=$((2019 + ${SLURM_ARRAY_TASK_ID:?Use array 0-4%2}))
mkdir -p configs/.runtime
"$PYTHON" -m paper_pipeline.scripts.generate_runtime_config --template configs/config_vopt.yaml --year "$year" --output "configs/.runtime/config_vopt_${year}.yaml"
"$(dirname "$PYTHON")/snakemake" --snakefile workflow/Snakefile --cores "${SLURM_CPUS_PER_TASK:-16}" --configfile "configs/.runtime/config_vopt_${year}.yaml" -- "results/vopt_${year}/texas/networks/elec_s500_c185_ec_lvopt_3h_E_operations.nc"
