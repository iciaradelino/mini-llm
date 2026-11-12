#!/bin/bash
#SBATCH --job-name=job_opt
#SBATCH --partition=gpu
#SBATCH --nodelist=haskell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/job_opt_%j.out
#SBATCH --error=slurm/logs/job_opt_%j.err

# Step 12: before/after of torch.compile on 1 GPU (haskell), ONE bundled job.
# Prediction: docs/optimization.md (committed before any run of this job).
#   baseline fp32 x3 and fp32 --compile x3, interleaved (baseline, compiled, baseline, ...)
#   so slow drift in the node affects both variants equally.
# Same settings for both: seed 1337, global batch 64, seq 256, 500 steps, default lr
# schedule. Only --compile differs. 500 steps gives val_loss_500 for the 2% rule.
#
# Before submitting: git pull (needs --compile from master) and data/tokenized/ built.
# Submit from the repo root:   sbatch slurm/job_opt_compile.sbatch
# If pending:  squeue --me --start > slurm/logs/queue_$(date +%Y%m%d_%H%M).txt

set -uo pipefail          # no -e: one failed run must not kill the rest of the bundle
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs

echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID:-local} | Node: $(hostname) | Start: $(date)"
echo "=================================================="

module load pytorch/2.8.0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# torch.compile (Triton) needs a C compiler and the container has none (jobs 28658, 28778:
# "Failed to find C compiler"). The host has gcc 10.4.1 on CVMFS; bind /cvmfs into the
# container and point CC at it. Applies to every run of this job, baseline included.
export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}/cvmfs"
export CC=/cvmfs/software.eessi.io/versions/2023.06/compat/linux/x86_64/usr/bin/gcc

JOB=opt
SEED=1337                 # same seed everywhere: required by the correctness rule
REPEATS=3
FAILED=0

nvidia-smi
torch-python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))" \
  || { echo "[ERROR] no CUDA device"; exit 1; }
test -s data/tokenized/train.bin -a -s data/tokenized/val.bin \
  || { echo "[ERROR] data/tokenized/{train,val}.bin missing"; exit 1; }
grep -q -- "--compile" src/train.py \
  || { echo "[ERROR] src/train.py has no --compile: git pull first"; exit 1; }

# Fail fast: does torch.compile work inside the container on this GPU? (needs triton and
# a C compiler). Smoke row only; if it fails the bundle would only produce failed rows.
echo; echo "=== [CHECK] torch.compile, 30 steps | $(date '+%F %T') ==="
if ! torch-python src/train.py --device cuda --seed "$SEED" --job smoke \
      --experiment compile_check --runs_csv results/smoke_runs.csv \
      --data_dir data/tokenized --max_steps 30 --eval_interval 30 --compile; then
  echo "[ERROR] torch.compile does not work here: report as 'could not run' with this log"
  exit 1
fi

# run <label> <train.py args...>   (continues on failure, counts failures)
# Every run gets its own empty Inductor/Triton cache directory (and the Inductor caches
# are disabled), so each compiled run pays the full cold compile cost and its wall_s
# shows it. Without this, the compile_check run above would warm the cache for them.
run() {
  local label="$1"; shift
  local cache; cache=$(mktemp -d)
  echo; echo "=== [RUN] $label | $(date '+%F %T') ==="
  if env TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCHINDUCTOR_CACHE_DIR="$cache/inductor" \
        TRITON_CACHE_DIR="$cache/triton" \
        torch-python src/train.py --device cuda --seed "$SEED" --job "$JOB" \
        --experiment "$label" --data_dir data/tokenized \
        --precision fp32 --seq_len 256 --max_steps 500 --eval_interval 100 \
        --summary_json "slurm/logs/job_opt_${SLURM_JOB_ID:-local}_${label}.json" "$@"; then
    echo "=== [OK] $label ==="
  else
    echo "=== [FAILED] $label (exit $?) ==="; FAILED=$((FAILED + 1))
  fi
  rm -rf "$cache"
}

for r in $(seq 1 $REPEATS); do
  run "opt_fp32_r$r"
  run "opt_fp32_compile_r$r" --compile
done

echo "=================================================="
echo "Job opt finished: $(date) | failed runs: $FAILED"
echo "=================================================="
exit $(( FAILED > 0 ? 1 : 0 ))
