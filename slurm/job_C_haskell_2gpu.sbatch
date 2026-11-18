#!/bin/bash
#SBATCH --job-name=job_C
#SBATCH --partition=gpu
#SBATCH --nodelist=haskell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6000ada:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# Adjust from Job A: sacct -j <job A id> --format=JobID,Elapsed  (then add ~50% margin)
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/job_C_%j.out
#SBATCH --error=slurm/logs/job_C_%j.err

# Job C: 2 GPUs on haskell (RTX 6000 Ada, PCIe, no NVLink), DDP over NCCL, ONE bundled job.
# The pytorch/2.8.0 module help says multi-GPU on haskell REQUIRES NCCL_P2P_DISABLE=1 (the
# P2P path is slower, pending an IOMMU fix). So the known-good setting runs FIRST and the
# P2P-enabled runs have a timeout: if they hang or crash, that is a result to report
# (with the log), not a bug in the script.
#   probes   30 steps with NCCL_DEBUG=INFO, P2P off then on. Evidence of which transport
#            NCCL really used (P2P vs SHM). Not written to runs.csv.
#   DDP, NCCL_P2P_DISABLE=1   x3  (500 steps, global batch 64 = 32 per GPU)
#   DDP, P2P enabled          x3  (same workload; may hang/fail, see above)
#   profiler run (P2P disabled, 50 steps; row marked PROFILED, not comparable)
# Compare with the Job A fp32 baseline (1 GPU, same seed and global batch).
#
# DRAFT for review by Icíar (owner of DDP/instrumentation): never run on GPU with NCCL yet.
# Submit from the repo root:   sbatch slurm/job_C_haskell_2gpu.sbatch
# If pending:  squeue --me --start > slurm/logs/queue_$(date +%Y%m%d_%H%M).txt

set -uo pipefail          # no -e: one failed run must not kill the rest of the bundle
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs results/profiles

echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID:-local} | Node: $(hostname) | Start: $(date)"
echo "=================================================="

nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi topo -m          # PCIe vs NVLink: goes into env/cluster.md

module load pytorch/2.8.0
export OMP_NUM_THREADS=8    # per rank (16 CPUs / 2 ranks)
export NCCL_DEBUG=VERSION   # prints the NCCL version once per run, nothing else

JOB=C
SEED=1337                   # same seed as the single-GPU baseline (2% rule)
REPEATS=3
FAILED=0

torch-python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count(), '| nccl', torch.cuda.nccl.version())" \
  || { echo "[ERROR] no CUDA device"; exit 1; }
test "$(torch-python -c 'import torch; print(torch.cuda.device_count())')" -ge 2 \
  || { echo "[ERROR] fewer than 2 GPUs visible"; exit 1; }
test -s data/tokenized/train.bin -a -s data/tokenized/val.bin \
  || { echo "[ERROR] data/tokenized/{train,val}.bin missing"; exit 1; }

# ddp <train.py args...>: 2 ranks on this node. The timeout (exit 124) stops a hung NCCL
# init / all-reduce from using the rest of the allocation.
DDP_TIMEOUT=${DDP_TIMEOUT:-600}
ddp() {
  timeout "$DDP_TIMEOUT" torch-python -m torch.distributed.run --standalone --nproc_per_node=2 src/train.py \
    --device cuda --seed "$SEED" --job "$JOB" --data_dir data/tokenized "$@"
}

# run <label> <train.py args...>   (continues on failure, counts failures)
run() {
  local label="$1"; shift
  echo; echo "=== [RUN] $label | $(date '+%F %T') ==="
  if ddp --experiment "$label" "$@"; then
    echo "=== [OK] $label ==="
  else
    echo "=== [FAILED] $label (exit $?) ==="; FAILED=$((FAILED + 1))
  fi
}

# probe <tag>: short run with NCCL_DEBUG=INFO; keeps the full log, prints the key lines
probe() {
  local tag="$1" f="slurm/logs/job_C_${SLURM_JOB_ID:-local}_nccl_$1.log"
  echo; echo "=== [PROBE] nccl_$tag | $(date '+%F %T') ==="
  NCCL_DEBUG=INFO ddp --max_steps 30 --eval_interval 30 --comm_measure_steps 0 \
      --runs_csv none > "$f" 2>&1 \
    || { echo "=== [FAILED] probe $tag ==="; FAILED=$((FAILED + 1)); }
  grep -E "NCCL version|Channel .* via|Using network" "$f" | sort -u | head -n 8 || true
  echo "full NCCL log: $f"
}

# 1) probes: which transport does NCCL pick? Known-good (P2P off) first.
export NCCL_P2P_DISABLE=1
probe p2p_off
unset NCCL_P2P_DISABLE
probe p2p_on                # may hang: stopped by DDP_TIMEOUT

# 2) DDP, P2P disabled x3 (traffic goes through host memory). Required setting on haskell.
export NCCL_P2P_DISABLE=1
for r in $(seq 1 $REPEATS); do
  run "ddp2_p2p_off_r$r" --precision fp32 --seq_len 256 --max_steps 500 --eval_interval 100
done

# 3) profiler run, P2P disabled. Needs >= 27 steps (20 timing warm-up + 2 + 5).
#    Untested under DDP: if --profile breaks with 2 ranks, this is the only run that fails.
run "ddp2_profile_fp32" --precision fp32 --seq_len 256 --max_steps 50 --eval_interval 50 \
  --profile --profile_steps 5 --profile_dir results/profiles
unset NCCL_P2P_DISABLE

# 4) DDP, P2P enabled (default) x3. If the module note is right these can hang or fail;
#    each gets DDP_TIMEOUT seconds. Keep the log either way: it is the evidence.
for r in $(seq 1 $REPEATS); do
  run "ddp2_p2p_on_r$r" --precision fp32 --seq_len 256 --max_steps 500 --eval_interval 100
done

echo "=================================================="
echo "Job C finished: $(date) | failed runs: $FAILED"
echo "=================================================="
exit $(( FAILED > 0 ? 1 : 0 ))
