#!/bin/bash
#SBATCH --job-name=job_D
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --nodelist=fortran,julia
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx6000bb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Job B (same GPU, 6 runs of 500 steps) took well under 15 min; 8 runs + 1 probe across
# nodes, each capped by DDP_TIMEOUT, fit in 1 h even if the network is slow.
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/job_D_%j.out
#SBATCH --error=slurm/logs/job_D_%j.err

# Job D: cross-node DDP, 1 GPU on each of fortran and julia (same GPU model, RTX PRO 6000
# Blackwell, so the comparison is clean), ONE bundled job.
#   probe       30 steps with NCCL_DEBUG=INFO: which network NCCL used between the nodes
#   fp32 x3     500 steps, global batch 64 = 32 per GPU   -> compare with Job B baseline_fp32
#   bf16 x3     same workload in bf16                     -> compare with Job B baseline_bf16
# Runs are interleaved (fp32 r1, bf16 r1, fp32 r2, ...) so a change in network load during
# the job affects both precisions equally.
#
# Why bf16: autocast makes compute ~2.8x faster on Blackwell (Job B) but the gradients that
# DDP all-reduces stay fp32 (13.9 M params = 55 MB per step), so communication should be a
# much larger share of the step. Prediction (written before running): docs/job_D.md.
#
# Reviewed by Icíar (DDP/instrumentation). If it never gets scheduled, record it as SKIPPED
# with the reason and keep the queue evidence:
#   squeue --me --start > slurm/logs/queue_$(date +%Y%m%d_%H%M).txt
# Submit from the repo root:   sbatch slurm/job_D_crossnode.sbatch

set -uo pipefail          # no -e: one failed run must not kill the rest of the bundle
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs

echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID:-local} | Nodes: ${SLURM_JOB_NODELIST:-?} | Start: $(date)"
echo "=================================================="

module load pytorch/2.8.0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NCCL_DEBUG=VERSION   # prints the NCCL version once per run, nothing else
# If NCCL hangs at init, pick the right network interface from the 'ip' output below and
# resubmit with it, e.g.:  NCCL_SOCKET_IFNAME=<interface> sbatch slurm/job_D_crossnode.sbatch
# (sbatch exports the submitting shell's environment by default).

JOB=D
SEED=1337                   # same seed as the single-GPU baseline (2% rule)
REPEATS=3
FAILED=0
RUN_N=0
PORT_BASE=29500
DDP_TIMEOUT=${DDP_TIMEOUT:-900}
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "rendezvous head node (rank 0): $HEAD_NODE"

# Every step gets one task per node and the job's CPUs. Since Slurm 22.05 srun no longer
# inherits --cpus-per-task from sbatch: without it each task would be pinned to 1 CPU.
# --kill-on-bad-exit: if one node's torchrun dies, kill the other instead of letting it
# wait in the rendezvous until DDP_TIMEOUT.
SRUN=(srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}"
      --kill-on-bad-exit=1)

# One block per node: GPU, interfaces, InfiniBand, CPU load. Goes into env/cluster.md.
"${SRUN[@]}" bash -c \
  'echo "--- $(hostname) | load $(cut -d" " -f1-3 /proc/loadavg) | cpus $(nproc)";
   nvidia-smi -L;
   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader;
   ip -brief addr 2>/dev/null;
   (ibstat 2>/dev/null | head -n 12) || echo "ibstat: not available";
   ls /sys/class/infiniband 2>/dev/null || echo "/sys/class/infiniband: none"'
srun --nodes=1 --ntasks=1 nvidia-smi

"${SRUN[@]}" torch-python -c "import torch, socket; print(socket.gethostname(), 'torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0), '| nccl', torch.cuda.nccl.version())" \
  || { echo "[ERROR] CUDA/torch check failed on one of the nodes"; exit 1; }
test -s data/tokenized/train.bin -a -s data/tokenized/val.bin \
  || { echo "[ERROR] data/tokenized/{train,val}.bin missing"; exit 1; }

# launch <train.py args...>: 2 nodes x 1 GPU. A new rendezvous port and id per launch
# avoids clashes with the previous one; timeout stops a hung rendezvous / NCCL init from
# eating the allocation.
launch() {
  RUN_N=$((RUN_N + 1))
  timeout "$DDP_TIMEOUT" "${SRUN[@]}" \
    torch-python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 \
      --rdzv_backend=c10d --rdzv_endpoint="${HEAD_NODE}:$((PORT_BASE + RUN_N))" \
      --rdzv_id="${SLURM_JOB_ID:-local}_${RUN_N}" \
      src/train.py --device cuda --seed "$SEED" --job "$JOB" --data_dir data/tokenized "$@"
}

# run <label> <train.py args...>   (continues on failure, counts failures)
run() {
  local label="$1"; shift
  echo; echo "=== [RUN] $label | $(date '+%F %T') ==="
  if launch --experiment "$label" "$@"; then
    echo "=== [OK] $label ==="
  else
    echo "=== [FAILED] $label (exit $?) ==="; FAILED=$((FAILED + 1))
  fi
}

# 1) probe: which network does NCCL use between the nodes (InfiniBand / RoCE / sockets)?
#    Not written to runs.csv. The NET/... and "via" lines go into env/cluster.md.
f="slurm/logs/job_D_${SLURM_JOB_ID:-local}_nccl_probe.log"
echo; echo "=== [PROBE] nccl_crossnode | $(date '+%F %T') ==="
NCCL_DEBUG=INFO launch --max_steps 30 --eval_interval 30 \
    --comm_measure_steps 0 --runs_csv none > "$f" 2>&1 \
  || { echo "=== [FAILED] probe (see $f) ==="; FAILED=$((FAILED + 1)); }
grep -E "NCCL version|NET/|Using network|via NET|Channel .* via" "$f" | sort -u | head -n 12 || true
echo "full NCCL log: $f"

# 2) cross-node DDP, fp32 and bf16 interleaved, x3 each
for r in $(seq 1 $REPEATS); do
  run "ddp_crossnode_fp32_r$r" --precision fp32 --seq_len 256 --max_steps 500 --eval_interval 100
  run "ddp_crossnode_bf16_r$r" --precision bf16 --seq_len 256 --max_steps 500 --eval_interval 100
done

echo "=================================================="
echo "Job D finished: $(date) | failed runs: $FAILED"
echo "=================================================="
exit $(( FAILED > 0 ? 1 : 0 ))
