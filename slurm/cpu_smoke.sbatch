#!/bin/bash
#SBATCH --job-name=cpu_smoke
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/cpu_smoke_%j.out
#SBATCH --error=slurm/logs/cpu_smoke_%j.err

# Step 7: CPU smoke test. Verifies `module load pytorch/2.8.0` inside a real job and
# runs prepare -> tokenizer -> encode -> 50 training steps (scripts/pipeline_tiny.sh).
#
# Submit from the repo root (slurm/logs/ must already exist):
#     sbatch slurm/cpu_smoke.sbatch
# Also test the 2-process gloo DDP path:
#     sbatch --export=ALL,RUN_DDP=1 slurm/cpu_smoke.sbatch
# Needs the offline raw Gutenberg files in data/raw/ (compute nodes may have no internet).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs

echo "=================================================="
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Start:       $(date)"
echo "Working dir: $(pwd)"
echo "=================================================="

module load pytorch/2.8.0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Record the environment first, so it is saved even if the pipeline fails.
ENV_FILE="slurm/logs/env_snapshot_${SLURM_JOB_ID:-local}.txt"
{
  echo "# job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"
  echo "## module list";  module list 2>&1
  echo "## python";       torch-python --version 2>&1
  echo "## pip freeze";   torch-python -m pip freeze 2>&1
} > "$ENV_FILE"
echo "environment snapshot written to $ENV_FILE"

export RUN_DDP="${RUN_DDP:-0}"
bash scripts/pipeline_tiny.sh

echo "=================================================="
echo "Finished: $(date)"
echo "=================================================="
