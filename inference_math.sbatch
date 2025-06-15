#!/bin/bash
#SBATCH --job-name=math_inference
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/math_inference_%j.out
#SBATCH --error=logs/math_inference_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules
module load cuda/11.8
module load python/3.10

# Install additional dependencies if needed
pip install --user tensorboardx pyyaml imageio

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run inference on mathematical functions
python inference_math.py \
    --model save/math_function_training/epoch-best.pth \
    --output_dir results/math_inference \
    --scale_factors 4 8 16 \
    --test_functions 2,2 2,8 5,5 8,2 8,8

echo "Mathematical function inference completed!" 