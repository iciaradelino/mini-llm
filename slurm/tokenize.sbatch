#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/tokenize_%j.out
#SBATCH --error=slurm/logs/tokenize_%j.err

# Step 5: train BPE tokenizer on data/clean/train, encode train/val/tiny to .bin,
# then run the round-trip test. Submit from the repo root:
#     sbatch slurm/tokenize.sbatch
# Requires data/clean/ from slurm/prep_data.sbatch.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "=================================================="
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Start:       $(date)"
echo "Working dir: $(pwd)"
echo "=================================================="

module load pytorch/2.8.0          # PyTorch lives in an Apptainer container: run Python with torch-python
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

torch-python -c "import sys, numpy, tokenizers; print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| tokenizers', tokenizers.__version__)"

test -d data/clean/train || { echo "[ERROR] data/clean/train missing - run slurm/prep_data.sbatch first"; exit 1; }

torch-python src/train_tokenizer.py --vocab_size 8192
torch-python src/encode.py --tiny_tokens 1500000
torch-python src/test_roundtrip.py

sha256sum data/tokenized/*.bin data/tokenized/tokenizer.json
echo "=================================================="
echo "Finished: $(date)"
echo "=================================================="
