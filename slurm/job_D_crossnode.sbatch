#!/bin/bash
#SBATCH --job-name=job_D
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --nodelist=fortran,julia
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx6000bb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Adjust from Job A: sacct -j <job A id> --format=JobID,Elapsed  (then add ~50% margin)
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/job_D_%j.out
#SBATCH --error=slurm/logs/job_D_%j.err

# Job D: cross-node DDP, 1 GPU on each of fortran and julia (same GPU model, RTX PRO 6000
# Blackwell, so the comparison is clean), ONE bundled job.
#   probe   30 steps with NCCL_DEBUG=INFO: which network NCCL used between the nodes
#   DDP x3  (500 steps, fp32, global batch 64 = 32 per GPU)
# Compare with the Job B fp32 runs (1 GPU Blackwell, same seed and global batch).
#
# DRAFT for review by Icíar (owner of DDP/instrumentation): never run multi-node yet.
# If it never gets scheduled, record it as SKIPPED with the reason and keep the queue
# evidence:  squeue --me --start > slurm/logs/queue_$(date +%Y%m%d_%H%M).txt
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
# If NCCL hangs at init, pick the right network interface from the 'ip' output below:
# export NCCL_SOCKET_IFNAME=<interface>

JOB=D
SEED=1337                   # same seed as the single-GPU baseline (2% rule)
REPEATS=3
FAILED=0
RUN_N=0
PORT_BASE=29500
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "rendezvous head node: $HEAD_NODE"

# One line per node: GPU, interfaces, topology. Goes into env/cluster.md.
srun --ntasks=2 --ntasks-per-node=1 bash -c \
  'echo "--- $(hostname)"; nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; ip -brief addr 2>/dev/null; (ibstat 2>/dev/null | head -n 5) || true'
srun --ntasks=1 --nodes=1 nvidia-smi

srun --ntasks=2 --ntasks-per-node=1 torch-python -c "import torch, socket; print(socket.gethostname(), 'torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0), '| nccl', torch.cuda.nccl.version())" \
  || { echo "[ERROR] CUDA/torch check failed on one of the nodes"; exit 1; }
test -s data/tokenized/train.bin -a -s data/tokenized/val.bin \
  || { echo "[ERROR] data/tokenized/{train,val}.bin missing"; exit 1; }

# launch <train.py args...>: 2 nodes x 1 GPU. A new rendezvous port per launch avoids
# clashes with the previous one; timeout stops a hung rendezvous from eating the allocation.
launch() {
  RUN_N=$((RUN_N + 1))
  timeout 900 srun --ntasks=2 --ntasks-per-node=1 \
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

# 1) probe: which network does NCCL use between the nodes (InfiniBand / Ethernet sockets)?
f="slurm/logs/job_D_${SLURM_JOB_ID:-local}_nccl_probe.log"
echo; echo "=== [PROBE] nccl_crossnode | $(date '+%F %T') ==="
NCCL_DEBUG=INFO launch --max_steps 30 --eval_interval 30 --comm_measure_steps 0 \
    --runs_csv none > "$f" 2>&1 \
  || { echo "=== [FAILED] probe ==="; FAILED=$((FAILED + 1)); }
grep -E "NCCL version|Channel .* via|Using network|NET/" "$f" | sort -u | head -n 8 || true
echo "full NCCL log: $f"

# 2) cross-node DDP x3
for r in $(seq 1 $REPEATS); do
  run "ddp_crossnode_r$r" --precision fp32 --seq_len 256 --max_steps 500 --eval_interval 100
done

echo "=================================================="
echo "Job D finished: $(date) | failed runs: $FAILED"
echo "=================================================="
exit $(( FAILED > 0 ? 1 : 0 ))
