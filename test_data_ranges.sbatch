#!/bin/bash
#SBATCH --job-name=test_uchi_ranges
#SBATCH --output=logs/test_uchi_ranges_%j.out
#SBATCH --error=logs/test_uchi_ranges_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=cpu
#SBATCH --account=bcqs-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Create logs directory if it doesn't exist
mkdir -p logs

# Load conda module if available
module load anaconda3_cpu 2>/dev/null || \
module load miniconda3 2>/dev/null || \
module load conda 2>/dev/null || \
echo "No conda module found, using system conda..."

# Initialize conda for bash
eval "$(conda shell.bash hook)" 2>/dev/null || true

# Activate the LIIF environment
echo "Activating conda environment: liif_env"
conda activate liif_env

# Verify environment is working
echo "Verifying environment..."
python -c "import torch, numpy, h5py; print('Environment check passed!')"

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Print Python information
echo "Python version:"
python --version

echo ""
echo "Testing U_CHI data distribution ranges..."
echo "This will show what ranges the model will see during training/inference"
echo ""

# Run the test script
python test_data_ranges.py

echo ""
echo "Job completed at: $(date)"
echo "Check output above for data range analysis" 