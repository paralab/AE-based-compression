#!/bin/bash
#SBATCH --job-name=setup_liif_env
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup_env_%j.out
#SBATCH --error=logs/setup_env_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Setting up LIIF conda environment on cluster..."

# Try different conda modules
module load anaconda3_cpu 2>/dev/null || \
module load miniconda3 2>/dev/null || \
module load conda 2>/dev/null || \
echo "No conda module found, trying system conda..."

# Initialize conda
eval "$(conda shell.bash hook)" 2>/dev/null || true

# Run the setup script
bash setup_conda_env.sh

echo "Environment setup completed!"
echo "You can now use the updated sbatch scripts." 