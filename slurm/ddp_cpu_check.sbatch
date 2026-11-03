#!/bin/bash
#SBATCH --job-name=ddp_cpu_check
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/ddp_cpu_check_%j.out
#SBATCH --error=slurm/logs/ddp_cpu_check_%j.err

# Step 6b check: train.py with DDP, 2 processes on CPU (gloo), on the real tiny.bin.
# Submit from the repo root:
#     sbatch slurm/ddp_cpu_check.sbatch
# Requires data/tokenized/ from slurm/tokenize.sbatch.
# Runs the same short config with 1 and then 2 processes (same seed, same global
# batch) so the log shows both rows side by side. Rows go to results/smoke_runs.csv
# (git-ignored), never to results/runs.csv; they are printed at the end of the log.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "=================================================="
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Start:       $(date)"
echo "Working dir: $(pwd)"
echo "=================================================="

module load pytorch/2.8.0
torch-python -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__, '| gloo', torch.distributed.is_gloo_available())"

test -f data/tokenized/tiny.bin || { echo "[ERROR] data/tokenized/tiny.bin missing - run slurm/tokenize.sbatch first"; exit 1; }

OUT=results/smoke_runs.csv
ARGS=(--dataset tiny --device cpu --seed 1337
      --max_steps 100 --global_batch 16 --warmup_steps 20
      --eval_interval 50 --eval_iters 10 --log_interval 10
      --timing_warmup 20 --comm_measure_steps 10
      --job smoke --runs_csv "$OUT")

echo "---------------- 1 process ----------------"
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} torch-python src/train.py "${ARGS[@]}" \
    --experiment ddp_check_ws1

echo "---------------- 2 processes (gloo) ----------------"
# split the cores between the two ranks so they don't oversubscribe
OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / 2 )) torch-python -m torch.distributed.run \
    --standalone --nproc_per_node=2 src/train.py "${ARGS[@]}" \
    --experiment ddp_check_ws2

echo "---------------- rows written to $OUT ----------------"
head -1 "$OUT"
grep "^${SLURM_JOB_ID:-local}" "$OUT" || tail -2 "$OUT"

echo "=================================================="
echo "Finished: $(date)"
echo "=================================================="
