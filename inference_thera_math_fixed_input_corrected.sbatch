#!/bin/bash
#SBATCH --job-name=thera_fixed_corrected
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/thera_fixed_corrected_%j.out
#SBATCH --error=logs/thera_fixed_corrected_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting LIIF Thera Fixed Input (CORRECTED) - All Scales (2x,4x,8x,16x,32x)..."

# Load conda module
module load anaconda3_cpu 2>/dev/null || \
module load miniconda3 2>/dev/null || \
module load conda 2>/dev/null || \
echo "No conda module found, using system conda..."

# Initialize conda
eval "$(conda shell.bash hook)" 2>/dev/null || true

# Activate the LIIF environment
conda activate liif_env

# Verify environment is working
echo "Verifying environment..."
python -c "import torch, yaml, tensorboardX, imageio; print('Environment check passed!')"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

echo "Running CORRECTED Thera inference with proper downsampling (GT→64×64 LR)..."
echo "Testing scales: 2x, 4x, 8x, 16x, 32x"

# Run corrected inference - proper downsampling relationship, all scales
python inference_thera_math_fixed_input.py \
    --thera_model save/thera_math_training/epoch-best.pth \
    --output_dir results/thera_fixed_input_corrected_all_scales \
    --scale_factors 2 4 8 16 32 \
    --test_functions 2,2 2,8 5,5 8,2 8,8

echo "Fixed input size inference (CORRECTED - All Scales) completed!" 