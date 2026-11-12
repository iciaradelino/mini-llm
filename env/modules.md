# Modules and software versions

Source of truth for versions: `slurm/logs/env_snapshot_27921.txt`, written by
`slurm/cpu_smoke.sbatch` from **inside** a Slurm job (`module list`, `python --version`
and `pip freeze`). The first snapshot (job 26995) was never committed, so it is not
used as a source.

## Modules

| Module        | Loaded with                 | Verified in a batch job                                                              |
|---------------|-----------------------------|--------------------------------------------------------------------------------------|
| pytorch/2.8.0 | `module load pytorch/2.8.0` | Slurm jobs 26995 (2026-11-04) and 27921 (2026-11-09), `cpu` partition, node `fortran` |

`module list` inside the job shows only `pytorch/2.8.0`. The module runs PyTorch in a
container and provides the `torch-python` launcher (Python 3.11.13). All scripts call
`torch-python`, not plain `python`, so they use the module's interpreter.

## Versions (job 27921)

| Package    | Version     | Source |
|------------|-------------|--------|
| python     | 3.11.13     | `env_snapshot_27921.txt` (`python --version`) |
| torch      | 2.8.0+cu128 | `env_snapshot_27921.txt` (`torch==2.8.0+cu128`) |
| numpy      | 2.3.2       | `env_snapshot_27921.txt` (listed as a wheel path `numpy-2.3.2-...whl`, because it comes with the module) |
| tokenizers | 0.23.3      | `env_snapshot_27921.txt` |
| pandas     | 3.1.0       | `env_snapshot_27921.txt` |
| matplotlib | 3.11.2      | `env_snapshot_27921.txt` |

CUDA 12.8 comes with the PyTorch build (`+cu128`). The NCCL version is recorded by
the first GPU job.

## Setup on a new account

torch is never pip-installed; the extra packages go on top of the module:

```bash
module load pytorch/2.8.0
torch-python -m pip install --user -r env/requirements.txt
```

## Verification runs

| Job   | Script                   | Result |
|-------|--------------------------|--------|
| 26989 | `slurm/prep_data.sbatch` | COMPLETED 0:0, 30 raw works, 25 train + 5 val, 18.30 MB clean |
| 26995 | `slurm/cpu_smoke.sbatch` | COMPLETED 0:0 in 00:01:13, 50 steps, loss 9.07 -> 6.57, `results/smoke_runs.csv` row status ok |
| 27921 | `slurm/cpu_smoke.sbatch` | COMPLETED, `[PIPELINE OK]` from a fresh clone; environment snapshot committed |
| 28436 | `slurm/prep_data.sbatch` | COMPLETED, 81 works, 69 train + 12 val, 33.26 MB clean |
| 28437 | `slurm/tokenize.sbatch`  | COMPLETED, 81-work corpus, round-trip test passed |