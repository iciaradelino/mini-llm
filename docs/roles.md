# Team roles

Everyone helps with the analysis, report and slides at the end.

| Person | Name | Area | Steps | Main files |
|--------|------|------|-------|------------|
| 1 | Nicolás Grass | Corpus | 4 | `src/prepare_corpus.py`, `slurm/prep_data.sbatch`, `data/README.md` |
| 2 | Olivia Espiñeira | Tokenizer and encoding | 5 | `src/train_tokenizer.py`, `src/encode.py`, `src/test_roundtrip.py`, `slurm/tokenize.sbatch` |
| 3 | Laura González | Model and training loop | 6a | `src/train.py`, `slurm/train_check.sbatch` |
| 4 | Icíar Adeliño | DDP and instrumentation | 6b | `src/ddp_utils.py`, `src/timing.py`, `src/runlog.py`, `results/README.md`, `slurm/ddp_cpu_check.sbatch` |
| 5 | Rodrigo Blanco | Slurm, environment, CPU testing | 7, 9 | `scripts/pipeline_tiny.sh`, `slurm/cpu_smoke.sbatch`, `slurm/job_*.sbatch`, `env/` |
| 6 | Lara Iglesias | Interfaces, correctness, reproducibility | 8, 17 | `docs/interfaces.md`, `src/check_correctness.py`, `REPRODUCE.md` |


## Questions for the instructor

1. Is the route approved: PyTorch + CUDA, PyTorch DDP, with the CPU/OpenMP
   comparison as the additional experiment?
2. Should we use the `gpu-prior+` account?
3. How are groups expected to cope with the GPU queue?

| Question | Answer | Date |
|----------|--------|------|
| 1        |        |      |
| 2        |        |      |
| 3        |        |      |
