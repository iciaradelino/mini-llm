# Reproducing the results

Every command below is copy-pasteable on the SciTech cluster. Submit everything
with `sbatch` from the repository root; **never run training on the login node**.
Phase 1 (sections 0–3) takes under 3 minutes of compute in three CPU jobs and
ends with a 50-step training run on the tiny subset.

## 0. Environment

### Get the code

```bash
git clone https://github.com/iciaradelino/mini-llm.git
cd mini-llm
```

All commands assume you stay in the repository root. `slurm/logs/` already exists
in the clone; Slurm writes every log there. The exact commit used for the results
in the report will be tagged and recorded here at the end of the project.

### Software

```bash
module load pytorch/2.8.0
torch-python -m pip install --user -r env/requirements.txt
```

`pytorch/2.8.0` is the only module needed. It runs PyTorch inside a container and
provides the `torch-python` launcher; every Slurm script calls `torch-python`, not
plain `python`, so the module's interpreter is used. torch is never pip-installed.

| Item | Version / value | Source |
|------|-----------------|--------|
| Module | `pytorch/2.8.0` (only module loaded) | `env/modules.md`, job 26995 |
| Python | 3.11.13 | jobs 26101, 26995 |
| PyTorch | 2.8.0+cu128 | job 26995 |
| CUDA | 12.8 (bundled with the PyTorch build) | `+cu128` suffix |
| NCCL | bundled with PyTorch; recorded by the first GPU job (`torch.cuda.nccl.version()`) | TODO after job A |
| numpy | 2.3.2 (provided by the module) | job 26101 |
| tokenizers | 0.23.3 | `env/requirements.txt` |
| pandas / matplotlib | 3.1.0 / 3.11.2 (analysis only) | `env/requirements.txt` |
| Compiler / flags | none: the project is pure Python; PyTorch and its CUDA kernels come prebuilt in the module | — |
| Slurm account / QoS | none set; assigned by the cluster | assignment brief §7 |

Cluster hardware and partitions: `env/cluster.md`.

### Fixed settings

| Setting | Value |
|---------|-------|
| Random seed | 1337 in every run (`train.py --seed`, default) |
| Model | 6 layers, 6 heads, width 384, 13.87M parameters, vocabulary 8,192 |
| Data | 81 Project Gutenberg works, 12 held out for validation (`data/README.md`) |
| Pipeline contract | `docs/interfaces.md` (file formats, `train.py` flags, `runs.csv` rules) |

### Environment variables set by the Slurm scripts

| Variable | Value | Script |
|----------|-------|--------|
| `OMP_NUM_THREADS` | `$SLURM_CPUS_PER_TASK` (halved per rank in 2-process CPU DDP) | `train_check`, `ddp_cpu_check`, `cpu_smoke`, `job_A` |
| `TOKENIZERS_PARALLELISM` | `true` | `tokenize` |
| `RAYON_NUM_THREADS` | `$SLURM_CPUS_PER_TASK` | `tokenize` |
| `NCCL_P2P_DISABLE` | `1` only in the P2P-off runs of job C | job C |

## 1. Data (cpu partition)

```bash
sbatch slurm/prep_data.sbatch
```

| Resources | Runtime | Log |
|-----------|---------|-----|
| cpu, 1 task, 4 cores, 8 GB, 30 min limit | ~30 s | `slurm/logs/prep_data_<jobid>.out` |

`src/prepare_corpus.py` (standard library only) downloads 81 public-domain works from
Project Gutenberg's mirror, `https://gutenberg.pglaf.org/cache/epub/<id>/pg<id>.txt`
(Gutenberg asks automated downloads to use a mirror), into `data/raw/` (cached:
existing files are not downloaded again), strips the Gutenberg header, footer and
licence text, normalises to UTF-8 and writes:

- `data/clean/train/` — 69 works, 30.13 MB
- `data/clean/val/` — 12 works held out whole, 3.13 MB:
  *Poesías completas* (Machado), *Misericordia* (Galdós), *La Tribuna* (Pardo Bazán),
  *Cañas y barro* (Blasco Ibáñez), *Cuentos y diálogos* (Valera), *Don Álvaro*
  (Duque de Rivas), *Sab* (Gómez de Avellaneda), *El Buscón* (Quevedo),
  *Fuente Ovejuna* (Lope de Vega), *Niebla* (Unamuno), *Luces de bohemia*
  (Valle-Inclán), *Azul...* (Darío)
- `data/clean/catalog.csv` — one row per work with its literary movement and genre

The full list of works with Gutenberg IDs, movement, genre and split is in `data/README.md`.
Compute nodes reach gutenberg.org (verified in jobs 25843 and 26989). If a node has
no internet, copy the 81 files into `data/raw/` first and the script uses them.

Check: the log ends with `Train Corpus Size: 30.13 MB`, `Val Corpus Size: 3.13 MB`
and `catalog.csv (81 works)`.

## 2. Tokenizer and encoding (cpu partition)

Needs `data/clean/` from step 1.

```bash
sbatch slurm/tokenize.sbatch
```

| Resources | Runtime | Log |
|-----------|---------|-----|
| cpu, 1 task, 4 cores, 8 GB, 30 min limit | ~15 s | `slurm/logs/tokenize_<jobid>.out` |

The job runs, in order:

```bash
torch-python src/train_tokenizer.py --vocab_size 8192   # BPE on data/clean/train only
torch-python src/encode.py --tiny_tokens 1500000         # .bin files + meta.json
torch-python src/test_roundtrip.py                       # decode(encode(text)) == text
sha256sum data/tokenized/*.bin data/tokenized/tokenizer.json
```

Outputs in `data/tokenized/`:

| File | Tokens |
|------|--------|
| `train.bin` | 9,069,382 |
| `val.bin` | 962,497 |
| `tiny.bin` | 1,500,000 (prefix of `train.bin`) |
| `tokenizer.json`, `meta.json` | — |

Format: flat little-endian uint16, no header; load with
`np.memmap("data/tokenized/train.bin", dtype=np.uint16, mode="r")`.

Check: the round-trip test passes, and the token counts and SHA-256 hashes in the log
match `meta.json`. Reruns produce byte-identical files.

## 3. CPU smoke test (cpu partition)

Needs `data/raw/` from step 1 (the pipeline stops if fewer than 81 raw files exist).

```bash
sbatch slurm/cpu_smoke.sbatch
```

| Resources | Runtime | Log |
|-----------|---------|-----|
| cpu, 1 task, 8 cores, 16 GB, 45 min limit | ~1.5 min | `slurm/logs/cpu_smoke_<jobid>.out`, `slurm/logs/env_snapshot_<jobid>.txt` |

The job loads the module, saves `module list`, the Python version and `pip freeze`
to `slurm/logs/env_snapshot_<jobid>.txt`, then runs `scripts/pipeline_tiny.sh`:
prepare corpus → train tokenizer → encode → memmap check → 50 training steps on
`tiny.bin`. The training command it runs is:

```bash
torch-python src/train.py --data_dir data/tokenized_smoke --dataset tiny --device cpu \
    --seed 1337 --max_steps 50 --global_batch 16 --warmup_steps 5 \
    --eval_interval 25 --eval_iters 5 --log_interval 10 --num_threads 8 \
    --job smoke --runs_csv results/smoke_runs.csv \
    --experiment smoke_single_<jobid> --summary_json slurm/logs/smoke_summary_<jobid>.json
```

Smoke outputs go to `data/tokenized_smoke/` and `results/smoke_runs.csv` (both
git-ignored), so the test never touches `data/tokenized/` or `results/runs.csv`.

Expected result (reference: job 26995):
- the log ends with `[PIPELINE OK]`;
- training loss falls from 9.07 to about 6.6, validation loss at step 50 is about 6.79;
- one row with `status = ok` is appended to `results/smoke_runs.csv`.

To also test 2-process DDP with the gloo backend:

```bash
sbatch --export=ALL,RUN_DDP=1 slurm/cpu_smoke.sbatch
```

### Correctness check

`src/check_correctness.py` fills the `valid` column of `results/runs.csv` (rule in
`README.md`, matching in `docs/interfaces.md` §5). It uses the standard library
only and takes milliseconds, so it can run on the login node:

```bash
python3 tests/test_check_correctness.py     # expected: PASSED: 18 rows ...
python3 src/check_correctness.py            # after every job that adds rows
```

## 4. Experiments

| Job | Script | Seeds | Notes |
|-----|--------|-------|-------|
| A | `slurm/job_A_haskell_1gpu.sbatch` | 1337 | template; GPU partition TODO |
| B | TODO | 1337 | |
| C | TODO | 1337 | |
| D | TODO | 1337 | |
| E | TODO | 1337 | |

After each job: `python3 src/check_correctness.py`, then commit `results/runs.csv`
and the job's logs.

## 5. Analysis

```bash
# TODO: python src/analyze.py results/runs.csv
```

## 6. Final model and samples

```bash
# TODO
```

## Job log index

Every submitted job (job ID, script, dates, outcome). Raw logs are in `slurm/logs/`.

| Job ID | Script | Node | Started | Outcome |
|--------|--------|------|---------|---------|
| 25843 | `slurm/prep_data.sbatch` | julia | 2026-10-30 17:50 | completed, 30 s; first corpus build |
| 26101 | `slurm/tokenize.sbatch` | haskell | 2026-11-01 14:20 | completed, 11 s; round-trip test passed |
| 26102 | `slurm/train_check.sbatch` | fortran | 2026-11-01 14:22 | completed, 3.5 min; 200 steps, val loss 6.67 → 6.06 |
| 26989 | `slurm/prep_data.sbatch` | fortran | 2026-11-04 12:07 | completed, 30 s; corpus rebuilt, 18.30 MB clean |
| 26995 | `slurm/cpu_smoke.sbatch` | fortran | 2026-11-04 12:13 | completed, 1 min 12 s; `[PIPELINE OK]` |
| 28436 | `slurm/prep_data.sbatch` | fortran | 2026-11-11 08:57 | completed, <1 s (raw files cached); 81-work corpus, 33.26 MB clean |
| 28437 | `slurm/tokenize.sbatch` | fortran | 2026-11-11 08:57 | completed, 18 s; 81-work corpus, round-trip test passed |