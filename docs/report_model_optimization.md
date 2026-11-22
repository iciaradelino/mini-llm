# Model, training loop and optimization

Draft of the report section "Model and optimization". Numbers come from
`results/runs.csv` (job `opt`, Slurm job 32275, repetition on the current corpus); the full prediction, measurement
and per-run table are in `docs/optimization.md`.

## 1. Model

The model is a GPT-style decoder-only transformer adapted from nanoGPT and written
in plain PyTorch (`src/train.py`).

| Parameter | Value |
|-----------|-------|
| Layers / heads / width | 6 / 6 / 384 |
| Context length | 256 tokens |
| Vocabulary | 8192 (byte-level BPE trained on the training split only) |
| Parameters | 13.87 M (input embedding and output layer share weights) |
| Block | pre-LayerNorm, causal self-attention (`scaled_dot_product_attention`), 4x GELU MLP, no dropout |

The corpus is a set of public-domain Spanish books from Project Gutenberg. It is
tokenized once into `uint16` `.bin` files and read through a memory map, so the data
loader costs about 0.5 ms per step.

## 2. Training loop

- AdamW (betas 0.9 / 0.95, weight decay 0.1 on matrices only), gradient clipping at
  1.0, cosine learning-rate schedule from 6e-4 to 6e-5 with 100 warm-up steps.
- Global batch of 64 sequences (16384 tokens per step). Under DDP the per-GPU batch
  is 64 divided by the number of processes, so every configuration trains on the same
  tokens per step.
- Precision is fp32 by default, or bf16 autocast. TF32 is not enabled.
- Fixed seed (1337): the same seed gives the same initialization and data order.
- Every run times its own phases per step (data, forward and backward, optimizer,
  communication). The first 20 steps are excluded from the medians. Each run appends
  one row to `results/runs.csv`.
- Validation uses fixed windows, so validation losses are comparable across runs.

## 3. Correctness rule

A run counts as valid if its validation loss after 500 steps (`val_loss_500`) is
within 2 % of the single-GPU fp32 baseline with the same seed, global batch and
number of steps. `src/check_correctness.py` applies the rule and fills the `valid`
column. An optimized run is judged against the baseline and never becomes one.

## 4. Optimization: `torch.compile`

**What and why.** We trained with `--compile`, changing nothing else. The hypothesis
was that on a small model a step is many small GPU kernels (layer norm, GELU,
residual additions), and that fusing them would cut launch overhead and memory
traffic.

**Prediction (written and committed before any measurement).** Median step time 10-30 %
lower (central guess 20 %), almost all of it in forward and backward;
`val_loss_500` within 0.5 %; peak memory within 10 %; total run time *higher*,
because every run pays the compile cost.

**Method.** One Slurm job on one RTX 6000 Ada (haskell): three baseline runs and
three compiled runs, interleaved, 500 steps each, same seed and batch. Each compiled
run started with an empty compilation cache, so its run time includes the full compile
cost.

**Result.**

| | Baseline (median of 3) | `--compile` (median of 3) | Change |
|---|---|---|---|
| Step time | 80.59 ms | 79.90 ms | **-0.86 %** |
| Throughput | 203 299 tokens/s | 205 069 tokens/s | +0.87 % |
| Peak GPU memory | 4580 MB | 4179 MB | -8.8 % |
| `val_loss_500` | 4.9265 | 4.9261 | within 0.03 % |
| Total run time | 43.8 s | 48.5 s | +4.7 s (+11 %) |

All six runs are valid under the 2 % rule. A first job (on an earlier, smaller
corpus) gave the same picture: 0.75 % faster, same memory (`docs/optimization.md`).
The table above is the repetition on the current corpus (job 32275).

**The prediction was refuted.** The gain is 0.86 %, far below the 10-30 % predicted
(the first job gave 0.75 %). The compiled runs were all slightly faster than all
baseline runs and very consistent with each other, so a small real effect is likely,
but it is not a useful speedup. The
predictions on memory, correctness, which phase changes and the higher run time held.
Compilation costs about 5 s per run; at 0.7 ms saved per step it would take roughly
7000 steps to pay back.

**Interpretation (not yet confirmed with the profiler).** A step performs roughly
1.4 TFLOP in 78 ms, about 17 TFLOP/s in fp32. This suggests the step is dominated by
the large matrix multiplications, which `torch.compile` does not speed up, while the
small operations it fuses are a minor part of the step. The profiler output for the
single-GPU job will show whether this is right.

**Limits.** One GPU model, one job, three repeats, 500 steps. We did not test bf16,
where matrix multiplications are faster and small operations weigh more, nor larger
models. Either would be a new optimization and would need its own prediction before
measuring.

## 5. Practical notes

- On this cluster `torch.compile` needs a C compiler that the PyTorch container does
  not provide. The job binds `/cvmfs` into the container and points `CC` at the host's
  gcc (`slurm/job_opt_compile.sbatch`). Without this, compilation fails with
  `Failed to find C compiler` (job 28778).
- GPU jobs waited in the queue for hours on a node whose memory was almost fully
  allocated by other users while a GPU was free (`slurm/logs/queue_*.txt`).
