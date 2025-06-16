#!/bin/bash
#SBATCH --job-name=thera_inf_32x_diag
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/thera_inf_32x_diag_%j.out
#SBATCH --error=logs/thera_inf_32x_diag_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting LIIF Thera Fixed Input (CORRECTED) - All Scales (2x,4x,6x,8x)..."
echo "Training: k∈{2,3,4,5,6} diagonal only, scale_max=8x"
echo "Testing: Diagonal combinations (2,2), (3,3), (4,4), (5,5), (6,6)"
echo "         + Superposition test: sin(4πx)sin(4πy) + sin(12πx)sin(12πy)"

# Load conda module if available
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
echo "Testing scales: 2x, 4x, 6x, 8x"
echo "Testing functions: diagonal combinations + superposition"

# Run corrected inference - diagonal combinations + superposition, all scales
python inference_thera_math_fixed_input.py \
    --thera_model save/thera_math_training_8x_diagonal/epoch-best.pth \
    --output_dir results/thera_fixed_input_corrected_diagonal_8x \
    --scale_factors 2 4 6 8 \
    --test_functions 2,2 3,3 4,4 5,5 6,6 super,0

echo "Fixed input size inference (CORRECTED - All Scales) completed!" 