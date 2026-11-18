#!/bin/bash
#SBATCH --job-name=job_B
#SBATCH --partition=gpu
#SBATCH --nodelist=julia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6000bb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Adjust from Job A: sacct -j <job A id> --format=JobID,Elapsed  (then add ~50% margin)
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/job_B_%j.out
#SBATCH --error=slurm/logs/job_B_%j.err

# Job B: 1 GPU on julia (RTX PRO 6000 Blackwell), ONE bundled job.
# Same workload as the Job A baseline, so the two GPU generations can be compared
# (Job A = Ada on haskell, Job B = Blackwell on julia):
#   fp32 x3 (500 steps, gives val_loss_500)
#   bf16 x3 (500 steps; judged by the 2% rule against THIS job's fp32 runs)
#
# Submit from the repo root:   sbatch slurm/job_B_julia_1gpu.sbatch
# If pending:  squeue --me --start > slurm/logs/queue_$(date +%Y%m%d_%H%M).txt

set -uo pipefail          # no -e: one failed run must not kill the rest of the bundle
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs

echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID:-local} | Node: $(hostname) | Start: $(date)"
echo "=================================================="

nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

module load pytorch/2.8.0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

JOB=B
SEED=1337                 # same seed everywhere: required by the correctness rule
REPEATS=3
FAILED=0

torch-python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))" \
  || { echo "[ERROR] no CUDA device"; exit 1; }
test -s data/tokenized/train.bin -a -s data/tokenized/val.bin \
  || { echo "[ERROR] data/tokenized/{train,val}.bin missing"; exit 1; }

# run <label> <train.py args...>   (continues on failure, counts failures)
run() {
  local label="$1"; shift
  echo; echo "=== [RUN] $label | $(date '+%F %T') ==="
  if torch-python src/train.py --device cuda --seed "$SEED" --job "$JOB" \
        --experiment "$label" --data_dir data/tokenized "$@"; then
    echo "=== [OK] $label ==="
  else
    echo "=== [FAILED] $label (exit $?) ==="; FAILED=$((FAILED + 1))
  fi
}

# 1) fp32 baseline x3: 500 steps so val_loss_500 exists
for r in $(seq 1 $REPEATS); do
  run "baseline_fp32_r$r" --precision fp32 --seq_len 256 --max_steps 500 --eval_interval 100
done

# 2) bf16 x3: 500 steps, compare val_loss_500 to the fp32 runs above (2% rule)
for r in $(seq 1 $REPEATS); do
  run "baseline_bf16_r$r" --precision bf16 --seq_len 256 --max_steps 500 --eval_interval 100
done

echo "=================================================="
echo "Job B finished: $(date) | failed runs: $FAILED"
echo "=================================================="
exit $(( FAILED > 0 ? 1 : 0 ))
