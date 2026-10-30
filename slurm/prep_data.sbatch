#!/bin/bash
#SBATCH --job-name=prep_data
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/prep_data_%j.out
#SBATCH --error=slurm/logs/prep_data_%j.err

# Create log directory if it doesn't exist
mkdir -p slurm/logs

# Print execution metadata
echo "=================================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Running on:    $(hostname)"
echo "Start time:    $(date)"
echo "Working dir:   $(pwd)"
echo "=================================================="

# Activate virtual environment if available
if [ -f "env/bin/activate" ]; then
    source env/bin/activate
fi

# Run corpus preparation script
python3 src/prepare_corpus.py

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="