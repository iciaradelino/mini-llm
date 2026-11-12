# Slurm scripts

All scripts write their logs to `slurm/logs/` (`#SBATCH --output=slurm/logs/<name>_%j.out`).
Keep every log — they are the evidence for the report, including queue blockers.
Submit every script from the repo root, e.g. `sbatch slurm/cpu_smoke.sbatch`.
Partitions (`cpu`, `gpu`) and GPU models are in `env/cluster.md`.

| Script | Partition / resources | Purpose | Status |
|--------|-----------------------|---------|--------|
| `prep_data.sbatch` | cpu, 4 cores, 8 GB, 30 min | Step 4: download corpus, clean, train/val split | done (jobs 25843, 26989, 27918; 81-work corpus 28436) |
| `tokenize.sbatch` | cpu, 4 cores, 8 GB, 30 min | Step 5: BPE tokenizer, encode `.bin` files, round-trip test | done (jobs 26101, 27919; 81-work corpus 28437) |
| `train_check.sbatch` | cpu, 8 cores, 8 GB, 30 min | Step 6a: single-device `train.py` on `tiny.bin`, 200 steps | done (jobs 26102, 28297: identical losses) |
| `ddp_cpu_check.sbatch` | cpu, 8 cores, 8 GB, 30 min | Step 6b: `train.py` with 1 and 2 gloo processes | done (job 27920) |
| `cpu_smoke.sbatch` | cpu, 8 cores, 16 GB, 45 min | Step 7: full pipeline on the tiny subset (`scripts/pipeline_tiny.sh`), 50 steps | done (jobs 26995, 27921) |
| `job_A_haskell_1gpu.sbatch` | gpu, 1 GPU on haskell, 8 cores, 32 GB, 2 h | Baseline fp32 ×3, seq-len sweep, bf16 ×3, profiler run | ready except `--partition` (still `TODO_GPU_PARTITION`, should be `gpu`); not run |
| `job_B_julia_1gpu.sbatch` | 1 GPU on julia | fp32 baseline + bf16 (GPU generation comparison) | planned |
| `job_C_haskell_2gpu.sbatch` | 2 GPUs on haskell | DDP, DDP with `NCCL_P2P_DISABLE=1`, profiler | planned |
| `job_D_crossnode.sbatch` | 1 GPU on each node | Cross-node DDP ×3 | planned |
| `job_E_cpu_threads.sbatch` | cpu, 64 cores, one thread per core (`--hint=nomultithread`), 32 GB, 6 h | Thread sweep 1–64 × 3, 50 steps, interleaved repeats (timing only) | ready; not run |
| `job_opt_compile.sbatch` | gpu, 1 GPU on haskell, 8 cores, 32 GB, 45 min | Step 12: fp32 vs fp32 `--compile`, ×3 each, interleaved; prediction in `docs/optimization.md` | ready; not run |

No script sets `--account` or `--qos`; the cluster assigns them (assignment §7).

## Jobs that depend on each other

`prep_data` → `tokenize` → everything else: each step reads what the previous one
wrote, so submitting them all at once makes the later jobs fail (jobs 27914–27916).
Chain them so Slurm starts each job only after the previous one succeeded:

```bash
J1=$(sbatch --parsable slurm/prep_data.sbatch)
J2=$(sbatch --parsable --dependency=afterok:$J1 slurm/tokenize.sbatch)
sbatch --dependency=afterok:$J2 slurm/cpu_smoke.sbatch
```

See `REPRODUCE.md`, "Run Phase 1 in one go".

## Queue evidence

When a GPU job is pending, save the queue state so it can be reported as a
measured blocker:

```bash
squeue --me --start > slurm/logs/queue_$(date +%Y%m%d_%H%M).txt
```