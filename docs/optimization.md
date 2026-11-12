# Optimization (step 12): `torch.compile`

> This file is committed **before** any measurement of the optimization. The
> prediction below must not be edited afterwards; results go in a separate
> section at the end, in a later commit.

## What changes

Train with `--compile` (wraps the model in `torch.compile`). Nothing else
changes: same model, data, seed, global batch, learning-rate schedule and
precision. Validation and checkpoints use the plain model.

## Why it should help

The baseline model is small (13.9M parameters, 6 layers, width 384) and one step
is short. A step runs many small GPU kernels (layer norms, GELU, residual
additions, attention projections, in forward and backward). On a small model the
GPU can spend a noticeable part of the step waiting for the next kernel to be
launched, and each small operation reads and writes GPU memory on its own.
`torch.compile` fuses chains of small operations into fewer kernels, so it should
reduce both the number of launches and the memory traffic.

This is a hypothesis from reasoning, not from measurements: the Job A profiler
output has not been read yet. If the profiler shows that the step is dominated by
the large matrix multiplications, the expected gain is smaller (see below).

## Prediction

Single GPU on haskell (RTX 6000 Ada), fp32, baseline model, global batch 64,
seq 256, `--max_steps 500`, seed 1337:

| Quantity | Prediction |
|----------|------------|
| `t_step_ms` (median) | **10–30 % lower** than the same run without `--compile` (central guess: about 20 %) |
| `tokens_per_s` | 11–43 % higher (the same change, expressed as a rate) |
| `t_fwd_bwd_ms` | carries almost all of the gain |
| `t_data_ms`, `t_optim_ms` | unchanged (within noise) |
| `val_loss_500` | within 0.5 % of the uncompiled run (the 2 % rule should pass easily) |
| `peak_mem_mb` | similar (within ±10 %); compilation usually lowers it, but this is uncertain |
| first steps | much slower (compile time, tens of seconds), excluded by the 20-step timing warm-up |
| `wall_s` (total run time) | **higher** for the compiled run: every compiled run pays the full cold compile time (caches are disabled and each run has its own empty cache directory) and, at 500 steps, this probably exceeds the time saved per step. The saving only pays off beyond roughly `compile time / saving per step` steps |

On the cpu partition I expect a smaller gain than on GPU (the step there is
dominated by matrix multiplications: about 98 % of the step in the 200-step
check, job 28297), so I will not use CPU to judge this optimization.

## How it will be measured

- Same job, same node, same GPU, runs one after another and **interleaved**
  (baseline, compiled, baseline, compiled, baseline, compiled), so slow drift in
  the node affects both variants equally and is not mistaken for the effect.
- Settings: `--max_steps 500`, default `--lr`, `--min_lr`, `--warmup_steps`,
  `--seed 1337`, global batch 64, seq 256.
- Compiled runs use an `--experiment` label containing `_compile` (the repeat
  index comes last, e.g. `opt_fp32_compile_r1`; see `docs/interfaces.md`) and
  `--job opt`.
- Compare the **median over the 3 repeats** of `t_step_ms` and `tokens_per_s`.
- Also report the total run time, `wall_s`, of every run, so the compile cost
  shows up. Every run starts with an empty Inductor/Triton cache, so the compile
  cost is cold in each compiled run (the 30-step `compile_check` that runs first
  does not warm anything for them). It is in the `SUMMARY` line of the Slurm log and in the per-run
  summary JSON (it is not a `runs.csv` column).
- Correctness: `src/check_correctness.py` on the compiled rows against the fp32
  single-GPU baseline.

## What would refute the prediction

- The gain in `t_step_ms` is below 5 %, or the compiled run is slower.
- The gain is smaller than the spread of the 3 baseline repeats (their
  max − min `t_step_ms`): then it cannot be told apart from noise.
- `val_loss_500` differs by more than 2 % from the baseline (`valid = no`).
- Compilation fails inside the container (no C compiler or triton), in which case
  the result is "could not run", reported with the error log.

If the gain is smaller than predicted, the likely reasons to check in the profiler
are: the step is dominated by large matmuls, or the GPU was already kept busy
without launch gaps.

## Results

*(to be filled in after the measurements, below this line, without editing the
sections above)*

### Measurement

Job 28862 (`slurm/job_opt_compile.sbatch`): 1 GPU on haskell (RTX 6000 Ada), fp32,
baseline model, global batch 64, seq 256, `--max_steps 500`, seed 1337. Three
interleaved pairs (baseline, compiled), every compiled run with a cold cache. All
six runs finished (`failed runs: 0`). Raw data: `results/runs.csv` (`job = opt`),
`slurm/logs/job_opt_28862*`.

| Run | `t_step_ms` | `t_fwd_bwd_ms` | `tokens_per_s` | `peak_mem_mb` | `val_loss_500` | `wall_s` |
|-----|-------------|----------------|----------------|---------------|----------------|----------|
| `opt_fp32_r1` | 80.02 | 77.79 | 204743 | 4580 | 4.9729 | 43.15 |
| `opt_fp32_compile_r1` | 79.86 | 77.63 | 205163 | 4179 | 4.9724 | 48.07 |
| `opt_fp32_r2` | 80.53 | 78.27 | 203455 | 4580 | 4.9721 | 43.67 |
| `opt_fp32_compile_r2` | 79.93 | 77.67 | 204989 | 4179 | 4.9737 | 48.17 |
| `opt_fp32_r3` | 80.72 | 78.49 | 202984 | 4580 | 4.9727 | 43.77 |
| `opt_fp32_compile_r3` | 79.94 | 77.69 | 204958 | 4179 | 4.9713 | 48.24 |
| **median, baseline** | **80.53** | 78.27 | 203455 | 4580 | 4.9727 | 43.67 |
| **median, compiled** | **79.93** | 77.67 | 204989 | 4179 | 4.9724 | 48.17 |

`t_data_ms` (about 0.51) and `t_optim_ms` (about 1.7) are the same in both variants.
The `wall_s` values are from the `SUMMARY` lines of the Slurm log.

### Against the prediction

| Quantity | Predicted | Measured | |
|----------|-----------|----------|---|
| `t_step_ms` (median) | 10–30 % lower | **0.75 % lower** (80.53 → 79.93 ms) | missed |
| `tokens_per_s` | 11–43 % higher | 0.75 % higher | missed |
| `t_fwd_bwd_ms` | carries the gain | carries all of it (−0.60 ms of −0.60 ms) | held |
| `t_data_ms`, `t_optim_ms` | unchanged | unchanged | held |
| `val_loss_500` | within 0.5 % | within 0.03 % of the baseline median; `valid = yes` for all six runs | held |
| `peak_mem_mb` | within ±10 % | 8.8 % lower (4580 → 4179 MB) | held |
| `wall_s` | higher | 4.5 s higher (43.67 → 48.17 s, +10 %) | held |

**The prediction is refuted** by two of the criteria set above: the gain is below
5 %, and it is smaller than the spread of the three baseline runs (0.60 ms against
0.69 ms, max − min). The three compiled runs are all slightly faster than the three
baseline runs and are very consistent with each other (spread 0.08 ms), so the
effect is probably real, but it is about 0.75 %, not the 10–30 % predicted.

The compile cost is about 5 s per run (the `wall_s` difference, which also contains
the small per-step saving). Saving 0.6 ms per step, it would take roughly 8000 steps
to pay for itself.

### Interpretation

This is a hypothesis, not yet checked against the profiler. One step moves 16384
tokens through 13.9M parameters in 78 ms, about 17 TFLOP/s in fp32. That is close to
what the large matrix multiplications (attention and MLP projections, and the
8192-wide output layer) can sustain on this GPU in fp32, so the step is probably
compute-bound in them. `torch.compile` fuses the small operations (layer norm, GELU,
residual additions), which are a small share of the step here, so there is little for
it to win. This is the case the "Why it should help" section warned about. The lower
peak memory is consistent with fusion removing intermediate activations.

Not tested: bf16, where the matrix multiplications are much faster and the small
operations weigh more, and larger models. Either would be a different optimization
with its own prediction written before running it.

### Runs that did not produce results

Two earlier submissions of the same job failed in the 30-step `compile_check` and are
kept as evidence (`slurm/logs/job_opt_28658.*`, `job_opt_28778.*`). Job 28658:
`--device cuda` without an index crashed in `torch.cuda.set_device` (fixed in
PR #27). Job 28778: Triton found no C compiler in the container (fixed in PR #28 by
binding `/cvmfs` and setting `CC`). Queue waits before each run are in
`slurm/logs/queue_*.txt`.
