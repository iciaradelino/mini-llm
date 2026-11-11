#!/bin/bash
#SBATCH --job-name=job_E
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread

#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/job_E_%j.out
#SBATCH --error=slurm/logs/job_E_%j.err





# Job E: CPU-only strong scaling of the baseline model (no GPU).
#   threads 1, 2, 4, 8, 16, 32, 64, x3 repeats, all in ONE job on ONE node,
#   so every run uses the same CPU model. Since the October maintenance Slurm
#   gives the job exactly --cpus-per-task cores (ConstrainCores), so 64 is real.
#
# Timing only: 50 steps per run, the first 10 not timed (40 timed steps for the
# median). A step with batch 64 takes ~25-30 s on 1 thread, so 500 steps would not
# fit; the 2% rule does not apply to job E (val_loss_500 stays empty -> n/a).
# Repeats are interleaved (all thread counts, then again), so a node that gets
# busier during the job affects every thread count equally.
#
# Expected duration ~2-3 h (dominated by the 1- and 2-thread runs).
# Submit from the repo root:   sbatch slurm/job_E_cpu_threads.sbatch
# Quick local test (no Slurm):  PY=python3 THREADS="1 2" REPEATS=1 STEPS=6 WARMUP=2 BATCH=8 \
#                               RUNS_CSV=/tmp/test_E.csv bash slurm/job_E_cpu_threads.sbatch


set -uo pipefail          # no -e: one failed run must not kill the rest of the job
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs

PY="${PY:-torch-python}"
THREADS="${THREADS:-1 2 4 8 16 32 64}"
REPEATS="${REPEATS:-3}"
STEPS="${STEPS:-50}"
WARMUP="${WARMUP:-10}"
BATCH="${BATCH:-64}"          # keep 64 for real runs (baseline global batch)
RUNS_CSV="${RUNS_CSV:-results/runs.csv}"
JOB=E
SEED=1337
FAILED=0




echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID:-local} | Node: $(hostname) | Start: $(date)"
echo "Cores granted: ${SLURM_CPUS_PER_TASK:-unknown} | threads: $THREADS | repeats: $REPEATS"
echo "Steps per run: $STEPS ($WARMUP not timed) | global batch: $BATCH | runs.csv: $RUNS_CSV"
echo "=================================================="

if [ "$PY" = "torch-python" ]; then
  module load pytorch/2.8.0
fi

# record the hardware and PyTorch's threading setup for the report
echo; echo "---- lscpu ----"; lscpu | grep -E "Model name|Socket|Core\(s\) per|Thread\(s\) per|NUMA node\(s\)|^CPU\(s\)|L3" || true
echo "---- cores visible to this job (nproc): $(nproc)"
echo "---- CPUs allowed: $(grep Cpus_allowed_list /proc/self/status | cut -f2)"
"$PY" -c "import torch; print('torch', torch.__version__); print(torch.__config__.parallel_info())" \
  || { echo "[ERROR] python/torch not available"; exit 1; }

max_t=$(echo $THREADS | tr ' ' '\n' | sort -n | tail -1)
if [ "$(nproc)" -lt "$max_t" ]; then
  echo "[WARN] only $(nproc) cores visible but up to $max_t threads requested: those runs will oversubscribe"
fi
test -s data/tokenized/train.bin -a -s data/tokenized/val.bin \
  || { echo "[ERROR] data/tokenized/{train,val}.bin missing (run slurm/tokenize.sbatch)"; exit 1; }



# run <threads> <repeat>   (continues on failure, counts failures)
run() {
  local t="$1" r="$2" label="cpu_threads${1}_r${2}"
  echo; echo "=== [RUN] $label | $(date '+%F %T') ==="
  # every thread pool PyTorch may use gets the same count
  if OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS="$t" MKL_NUM_THREADS="$t" OPENBLAS_NUM_THREADS="$t" \
     "$PY" src/train.py --device cpu --num_threads "$t" --seed "$SEED" \
       --job "$JOB" --experiment "$label" --data_dir data/tokenized \
       --precision fp32 --seq_len 256 --global_batch "$BATCH" \
       --max_steps "$STEPS" --timing_warmup "$WARMUP" \
       --eval_interval "$STEPS" --eval_iters 2 --log_interval 10 \
       --runs_csv "$RUNS_CSV"; then
    echo "=== [OK] $label ==="
  else
    echo "=== [FAILED] $label (exit $?) ==="; FAILED=$((FAILED + 1))
  fi
}

for r in $(seq 1 "$REPEATS"); do
  for t in $THREADS; do
    run "$t" "$r"
  done
done

echo "=================================================="
echo "Job E finished: $(date) | failed runs: $FAILED"
echo "=================================================="
exit $(( FAILED > 0 ? 1 : 0 ))