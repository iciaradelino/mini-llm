# Reproducing the results

> Fill in each section as the corresponding step is completed. Every command here
> must be copy-pasteable on the cluster. Never run training on the login node.

## 0. Environment

```bash
module load pytorch/2.8.0
```

| Item             | Version / value |
|------------------|-----------------|
| PyTorch module   | pytorch/2.8.0   |
| CUDA             | TODO            |
| NCCL             | TODO            |
| Python           | TODO            |
| Extra packages   | see `env/requirements.txt` |
| Slurm account    | TODO (gpu-prior+? — confirm with instructor) |

Cluster details: `env/cluster.md`.

## 1. Data (cpu partition)

```bash
# TODO: sbatch slurm/prepare_data.sbatch
```

- Sources: TODO (list of public-domain works + URLs)
- Validation hold-out: TODO (one or two complete works)

## 2. Tokenizer and encoding (cpu partition)

```bash
# TODO
```

- BPE vocabulary size: ~8,000, trained on training texts only
- Outputs: `data/tokenized/train.bin`, `data/tokenized/val.bin`, tiny test subset

## 3. CPU smoke test

```bash
# TODO: sbatch slurm/cpu_smoke.sbatch
```

## 4. Experiments

| Job | Script                 | Seeds | Notes |
|-----|------------------------|-------|-------|
| A   | TODO                   | TODO  |       |
| B   | TODO                   | TODO  |       |
| C   | TODO                   | TODO  |       |
| D   | TODO                   | TODO  |       |
| E   | TODO                   | TODO  |       |

## 5. Analysis

```bash
# TODO: python src/analyze.py results/runs.csv
```

## 6. Final model and samples

```bash
# TODO
```

## Job log index

Record every submitted job here (job ID, script, date, outcome). Raw logs live in
`slurm/logs/`.

| Job ID | Script | Submitted | Started | Outcome |
|--------|--------|-----------|---------|---------|
|        |        |           |         |         |
