#!/bin/bash
#SBATCH --job-name=train_check
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/train_check_%j.out
#SBATCH --error=slurm/logs/train_check_%j.err

# Step 6a check: single-device train.py on the real tiny.bin, CPU only.
# Submit from the repo root:
#     sbatch slurm/train_check.sbatch
# Requires data/tokenized/ from slurm/tokenize.sbatch.
# Short run (200 steps, global batch 16) just to see the loss go down on real text.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "=================================================="
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Start:       $(date)"
echo "Working dir: $(pwd)"
echo "=================================================="

module load pytorch/2.8.0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
torch-python -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__, '| threads', torch.get_num_threads())"

test -f data/tokenized/tiny.bin || { echo "[ERROR] data/tokenized/tiny.bin missing - run slurm/tokenize.sbatch first"; exit 1; }

torch-python src/train.py \
    --dataset tiny --device cpu --seed 1337 \
    --max_steps 200 --global_batch 16 --warmup_steps 20 \
    --eval_interval 50 --eval_iters 10 --log_interval 10 \
    --job smoke --runs_csv results/smoke_runs.csv \
    --summary_json slurm/logs/train_check_${SLURM_JOB_ID:-local}.json

echo "=================================================="
echo "Finished: $(date)"
echo "=================================================="
