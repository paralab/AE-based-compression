#!/bin/bash
#SBATCH --job-name=liif_test_bench
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/test_benchmark_%j.out
#SBATCH --error=logs/test_benchmark_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules
module load python/pytorch/2.2.0

# Install additional dependencies if needed
pip install --user tensorboardx pyyaml imageio

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="save/test_run/epoch-best.pth"

echo "=========================================="
echo "LIIF Benchmark Evaluation"
echo "Model: $MODEL_PATH"
echo "=========================================="

# Test on DIV2K validation set
echo "=== Testing on DIV2K validation set ==="
bash scripts/test-div2k.sh $MODEL_PATH 0

echo ""
echo "=========================================="
echo "Benchmark evaluation completed!"
echo "==========================================" 