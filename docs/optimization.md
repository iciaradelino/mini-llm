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
| `wall_s` (total run time) | **higher** for the compiled run: the compile time is paid once and, at 500 steps, probably exceeds the time saved per step. The saving only pays off beyond roughly `compile time / saving per step` steps |

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
  shows up. It is in the `SUMMARY` line of the Slurm log and in the per-run
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
