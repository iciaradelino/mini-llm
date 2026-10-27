# Slurm scripts

All scripts write their logs to `slurm/logs/` (`#SBATCH --output=slurm/logs/%x_%j.out`).
Keep every log — they are the evidence for the report, including queue blockers.

| Script (planned)        | Partition / resources   | Purpose |
|-------------------------|-------------------------|---------|
| `prepare_data.sbatch`   | cpu                     | Steps 4–5: corpus, tokenizer, encoding |
| `cpu_smoke.sbatch`      | cpu                     | Step 7: tiny model, few steps, check `module load pytorch/2.8.0` |
| `job_E_cpu_threads.sbatch` | cpu, full node       | Thread sweep 1–64 × 3 |
| `job_A_haskell_1gpu.sbatch`| 1 GPU on haskell     | Baseline, seq-len sweep, bf16, profiler |
| `job_B_julia_1gpu.sbatch`  | 1 GPU on julia       | Baseline + bf16 (GPU generation comparison) |
| `job_C_haskell_2gpu.sbatch`| 2 GPUs on haskell    | DDP, DDP with `NCCL_P2P_DISABLE=1`, profiler |
| `job_D_crossnode.sbatch`   | 1 GPU on each node   | Cross-node DDP ×3 |

## Queue evidence

When a GPU job is pending, save the queue state so it can be reported as a
measured blocker:

```bash
squeue --me --start > slurm/logs/queue_$(date +%Y%m%d_%H%M).txt
```
