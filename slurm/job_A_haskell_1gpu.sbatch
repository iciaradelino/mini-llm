#!/bin/bash
#SBATCH --job-name=job_A
# TODO(cluster.md): fill in the GPU partition and account before submitting.
#SBATCH --partition=TODO_GPU_PARTITION
##SBATCH --account=TODO_ACCOUNT
#SBATCH --nodelist=haskell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# TODO: set to the partition max wall time; estimate below is a guess (see EXPERIMENTS).
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/job_A_%j.out
#SBATCH --error=slurm/logs/job_A_%j.err

# Job A: 1 GPU on haskell, ONE bundled job:
#   fp32 baseline x3 (500 steps, gives val_loss_500 for the 2% rule)
#   seq-len sweep 128/512/1024 x3 (fp32, 100 steps; 256 = baseline above)
#   bf16 x3 (500 steps)
#   profiler run
#
# NOT READY TO SUBMIT until: (1) the smoke test passes, (2) the TODOs above are filled,
# (3) full data/tokenized/ exists (slurm/tokenize.sbatch).
# Submit from the repo root:   sbatch slurm/job_A_haskell_1gpu.sbatch
# If pending:  squeue --me --start > slurm/logs/queue_$(date +%Y%m%d_%H%M).txt

set -uo pipefail          # no -e: one failed run must not kill the rest of the bundle
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs

echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID:-local} | Node: $(hostname) | Start: $(date)"
echo "=================================================="

module load pytorch/2.8.0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

JOB=A
SEED=1337                 # same seed everywhere: required by the correctness rule
REPEATS=3
FAILED=0

nvidia-smi
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

# 2) sequence-length sweep, fp32, 100 steps (timing only, no step-500 loss)
for L in 128 512 1024; do
  for r in $(seq 1 $REPEATS); do
    run "seqlen${L}_fp32_r$r" --precision fp32 --seq_len "$L" --max_steps 100 --eval_interval 50
  done
done

# 3) bf16 x3: 500 steps, compare val_loss_500 to the fp32 baseline (2% rule)
for r in $(seq 1 $REPEATS); do
  run "baseline_bf16_r$r" --precision bf16 --seq_len 256 --max_steps 500 --eval_interval 100
done

# 4) profiler run. src/train.py has NO profiler flag yet. Ask Persons 3/4 for one (or a
#    wrapper script) and set PROFILE_CMD; until then this is recorded as SKIPPED.
if [ -n "${PROFILE_CMD:-}" ]; then
  echo; echo "=== [RUN] profiler ==="
  eval "$PROFILE_CMD" || { echo "=== [FAILED] profiler ==="; FAILED=$((FAILED + 1)); }
else
  echo; echo "=== [SKIPPED] profiler: train.py has no profiler option and PROFILE_CMD is not set ==="
fi

echo "=================================================="
echo "Job A finished: $(date) | failed runs: $FAILED"
echo "=================================================="
exit $(( FAILED > 0 ? 1 : 0 ))
