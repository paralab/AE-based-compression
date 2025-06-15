#!/bin/bash
#SBATCH --job-name=thera_math_inference
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/thera_math_inference_%j.out
#SBATCH --error=logs/thera_math_inference_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting LIIF Thera Math Inference (Fixed Version)..."

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

echo "Running Thera Math inference with improved script (includes 2x scaling)..."

# Run inference - only Thera model, skip LIIF comparison, includes 2x scaling
python inference_thera_math.py \
    --thera_model save/thera_math_training/epoch-best.pth \
    --output_dir results/thera_math_inference_fixed \
    --scale_factors 2 4 8 16 \
    --test_functions 2,2 2,8 5,5 8,2 8,8

echo "Inference completed!" 