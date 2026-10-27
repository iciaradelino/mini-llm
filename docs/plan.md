# Project plan

Tick items off as they are done. Milestones: first GPU results by **Session 25**,
deliverables before **Session 29**.

## Phase 0 — Organize
- [ ] 1. Split roles (`docs/roles.md`)
- [ ] 2. Ask the instructor the three questions (`docs/roles.md`)
- [x] 3. Create the repository skeleton; record cluster facts in `env/cluster.md`

## Phase 1 — Data and code (CPU only)
- [ ] 4. Corpus script: download, strip Gutenberg headers/licence, hold out 1–2 works (cpu partition)
- [ ] 5. Train BPE tokenizer (~8k, train only), encode to `.bin`, make tiny subset
- [ ] 6. Training script from nanoGPT: CLI args, DDP, per-phase timing (skip ~20 warm-up), append to `results/runs.csv`
- [ ] 7. CPU smoke test on cpu partition; verify `module load pytorch/2.8.0` in a job
- [x] 8. Correctness rule in README

## Phase 2 — First GPU results (Session 25)
- [ ] 9. Submit first bundled GPU job as soon as CPU test passes; save `squeue --start` output if blocked
- [ ] 10. CPU/OpenMP thread sweep: 1,2,4,8,16,32,64 threads × 3 repeats

## Phase 3 — GPU experiments
- [ ] Job A — 1 GPU haskell: fp32 ×3, seq-len 128–1024, bf16 ×3, profiler
- [ ] Job B — 1 GPU julia: fp32 ×3, bf16 ×3
- [ ] Job C — 2 GPU haskell: DDP ×3, P2P off ×3, profiler
- [ ] Job D — cross-node DDP ×3 (SKIPPED with reason if never scheduled)
- [ ] Job E — CPU thread sweep (= step 10)

## Phase 4 — Bottleneck and optimization
- [ ] 11. Analyse profiler output (A, C); save table/screenshot to `results/profiles/`
- [ ] 12. One optimization with predicted effect; before/after on same workload; apply correctness rule

## Phase 5 — Final model and analysis
- [ ] 13. Long run (1–3 h) with best config; samples from "En un lugar de la Mancha"
- [ ] 14. Analysis script: medians, speedup, efficiency, all plots

## Phase 6 — Deliverables (before Session 29)
- [ ] 15. Technical report (5–7 pages, nine sections, technology matrix)
- [ ] 16. EuroHPC scale-up plan (1–2 pages, Leonardo/LUMI, node-hours)
- [ ] 17. REPRODUCE.md complete; all logs and raw CSV committed
- [ ] 18. Presentation (5 slides, 5 minutes)

## Risks and fallbacks
- GPU queue blocks us → CPU experiments still give baseline, scaling and sensitivity results; document the queue as a measured blocker.
- Job D never scheduled → record as SKIPPED with reason.
- Never run training on the login node.
