# Modules and software versions

Source of truth for versions: `slurm/logs/env_snapshot_26995.txt`, written by
`slurm/cpu_smoke.sbatch` from **inside** a Slurm job (`module list` + `pip freeze`).

## Modules

| Module        | Loaded with                 | Verified in a batch job                                                      |
|---------------|-----------------------------|------------------------------------------------------------------------------|
| pytorch/2.8.0 | `module load pytorch/2.8.0` | Slurm job 26995 (`cpu` partition, node `fortran`, COMPLETED 0:0, 2026-11-04) |

`module list` inside the job showed only `pytorch/2.8.0`. The module provides the
`torch-python` launcher (Python 3.11.13). All scripts call `torch-python`, not plain
`python`, so they use the module's interpreter.

## Versions (job 26995)

| Package    | Version     | Source                                                                                               |
|------------|-------------|------------------------------------------------------------------------------------------------------|
| python     | 3.11.13     | env_snapshot_26995.txt                                                                               |
| torch      | 2.8.0+cu128 | env_snapshot_26995.txt                                                                               |
| numpy      | 2.3.2       | `slurm/logs/train_check_26102.out` (not listed as `numpy==` in the freeze; it comes from the module) |
| tokenizers | 0.23.3      | env_snapshot_26995.txt                                                                               |
| pandas     | 3.1.0       | env_snapshot_26995.txt                                                                               |
| matplotlib | 3.11.2      | env_snapshot_26995.txt                                                                               |

## Setup on a new account

torch is never pip-installed; the extra packages go on top of the module:

```bash
module load pytorch/2.8.0
torch-python -m pip install --user -r env/requirements.txt
```

## Verification runs

| Job   | Script                   | Result                                                                                         |
|-------|--------------------------|------------------------------------------------------------------------------------------------|
| 26989 | `slurm/prep_data.sbatch` | COMPLETED 0:0, 30 raw works, 25 train + 5 val, 18.30 MB clean                                  |
| 26995 | `slurm/cpu_smoke.sbatch` | COMPLETED 0:0 in 00:01:13, 50 steps, loss 9.07 -> 6.57, `results/smoke_runs.csv` row status ok |