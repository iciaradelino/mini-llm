# Project plan

Tick items off as they are done. Milestones: first GPU results by **Session 25**,
deliverables before **Session 29**.

## Phase 0 — Organize
- [x] 1. Split roles (`docs/roles.md`)
- [ ] 2. Ask the instructor the three questions (`docs/roles.md`; Q2 answered by the assignment brief)
- [x] 3. Create the repository skeleton; record cluster facts in `env/cluster.md` (partitions and GPU models recorded; per-node details still TODO)

## Phase 1 — Data and code (CPU only)
- [x] 4. Corpus script: download, strip Gutenberg headers/licence, hold out 1–2 works (cpu partition) — jobs 25843, 26989, 27918; expanded to 81 works (69 train, 12 val) in job 28436
- [x] 5. Train BPE tokenizer (~8k, train only), encode to `.bin`, make tiny subset — jobs 26101, 27919; 81-work corpus in job 28437
- [x] 6. Training script from nanoGPT: CLI args, DDP, per-phase timing (skip ~20 warm-up), append to `results/runs.csv` — single device job 26102 (reproduced exactly by 28297); 1 vs 2 gloo processes job 27920 (val loss differs by 0.2 %)
- [x] 7. CPU smoke test on cpu partition; verify `module load pytorch/2.8.0` in a job — jobs 26995, 27921
- [x] 8. Correctness rule in README — operational definition in `docs/interfaces.md` §5; `src/check_correctness.py` with test `tests/test_check_correctness.py`

## Phase 2 — First GPU results (Session 25)
- [ ] 9. Submit first bundled GPU job as soon as CPU test passes; save `squeue --start` output if blocked — `slurm/job_A_haskell_1gpu.sbatch` ready except `--partition` (should be `gpu`)
- [ ] 10. CPU/OpenMP thread sweep: 1,2,4,8,16,32,64 threads × 3 repeats — `slurm/job_E_cpu_threads.sbatch` ready, not run yet

## Phase 3 — GPU experiments
- [ ] Job A — 1 GPU haskell: fp32 ×3, seq-len 128–1024, bf16 ×3, profiler
- [ ] Job B — 1 GPU julia: fp32 ×3, bf16 ×3
- [ ] Job C — 2 GPU haskell: DDP ×3, P2P off ×3, profiler
- [ ] Job D — cross-node DDP ×3 (SKIPPED with reason if never scheduled)
- [ ] Job E — CPU thread sweep (= step 10)

## Phase 4 — Bottleneck and optimization
- [ ] 11. Analyse profiler output (A, C); save table/screenshot to `results/profiles/` — `train.py --profile` ready
- [ ] 12. One optimization with predicted effect; before/after on same workload; apply correctness rule — `torch.compile`: prediction in `docs/optimization.md`, job `slurm/job_opt_compile.sbatch` ready, not run yet

## Phase 5 — Final model and analysis
- [ ] 13. Long run (1–3 h) with best config; samples from "En un lugar de la Mancha" — checkpoints (`--save_checkpoint`) and `src/sample.py` ready
- [ ] 14. Analysis script: medians, speedup, efficiency, all plots — no owner yet

## Phase 6 — Deliverables (before Session 29)
- [ ] 15. Technical report (5–7 pages, nine sections, technology matrix)
- [ ] 16. EuroHPC scale-up plan (1–2 pages, Leonardo/LUMI, node-hours)
- [ ] 17. REPRODUCE.md complete; all logs and raw CSV committed — Phase 1 (sections 0–3) and job index done; sections 4–6 filled in as the jobs run
- [ ] 18. Presentation (5 slides, 5 minutes)

## Risks and fallbacks
- GPU queue blocks us → CPU experiments still give baseline, scaling and sensitivity results; document the queue as a measured blocker.
- Job D never scheduled → record as SKIPPED with reason.
- Never run training on the login node.