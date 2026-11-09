# Slurm scripts

All scripts write their logs to `slurm/logs/` (`#SBATCH --output=slurm/logs/<name>_%j.out`).
Keep every log — they are the evidence for the report, including queue blockers.
Submit every script from the repo root, e.g. `sbatch slurm/cpu_smoke.sbatch`.

| Script | Partition / resources | Purpose | Status |
|--------|-----------------------|---------|--------|
| `prep_data.sbatch` | cpu, 4 cores, 8 GB, 30 min | Step 4: download corpus, clean, train/val split | done (jobs 25843, 26989) |
| `tokenize.sbatch` | cpu, 4 cores, 8 GB, 30 min | Step 5: BPE tokenizer, encode `.bin` files, round-trip test | done (job 26101) |
| `train_check.sbatch` | cpu, 8 cores, 8 GB, 30 min | Step 6a: single-device `train.py` on `tiny.bin`, 200 steps | done (job 26102) |
| `ddp_cpu_check.sbatch` | cpu, 8 cores, 8 GB, 30 min | Step 6b: `train.py` with 1 and 2 gloo processes | script ready, no log yet |
| `cpu_smoke.sbatch` | cpu, 8 cores, 16 GB, 45 min | Step 7: full pipeline on the tiny subset (`scripts/pipeline_tiny.sh`), 50 steps | done (job 26995) |
| `job_A_haskell_1gpu.sbatch` | 1 GPU on haskell | Baseline, seq-len sweep, bf16, profiler | template, GPU partition TODO |
| `job_B_julia_1gpu.sbatch` | 1 GPU on julia | fp32 baseline + bf16 (GPU generation comparison) | planned |
| `job_C_haskell_2gpu.sbatch` | 2 GPUs on haskell | DDP, DDP with `NCCL_P2P_DISABLE=1`, profiler | planned |
| `job_D_crossnode.sbatch` | 1 GPU on each node | Cross-node DDP ×3 | planned |
| `job_E_cpu_threads.sbatch` | cpu, full node | Thread sweep 1–64 × 3 | planned |

No script sets `--account` or `--qos`; the cluster assigns them (assignment §7).

## Queue evidence

When a GPU job is pending, save the queue state so it can be reported as a
measured blocker:

```bash
squeue --me --start > slurm/logs/queue_$(date +%Y%m%d_%H%M).txt
```