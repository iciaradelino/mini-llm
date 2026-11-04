# Modules and software versions

Source of truth for versions: `slurm/logs/env_snapshot_<jobid>.txt`, written by
`slurm/cpu_smoke.sbatch` from **inside** a Slurm job (`module list` + `pip freeze`).
Copy the final values from the passing smoke job into the table below.

## Modules

| Module          | Loaded with                 | Verified in a batch job |
|-----------------|-----------------------------|-------------------------|
| pytorch/2.8.0   | `module load pytorch/2.8.0` | TODO: job ID of the passing cpu_smoke job |

The module provides the `torch-python` launcher. All scripts call `torch-python`
(not plain `python`) so they use the module's interpreter.

## Versions seen so far

| Package | Version | Where it was observed |
|---------|---------|-----------------------|
| torch   | 2.8.0+cu128 | `slurm/logs/train_check_26102.out` (node `fortran`, cpu partition) |
| numpy   | 2.3.2       | same log |
| python  | TODO        | env_snapshot of the cpu_smoke job |
| tokenizers | TODO     | env_snapshot / `data/tokenized*/meta.json` (`versions`) |
| pandas, matplotlib | TODO | env_snapshot |

## Extra packages

Installed on top of the module with `pip install --user -r env/requirements.txt`
(torch is never pip-installed). After the smoke job passes, pin the versions from
the env snapshot in `env/requirements.txt`.

## Setup on a new account

```bash
module load pytorch/2.8.0
pip install --user -r env/requirements.txt
```
