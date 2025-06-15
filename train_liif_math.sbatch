#!/bin/bash
#SBATCH --job-name=liif_math_train
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_liif_math_%j.out
#SBATCH --error=logs/train_liif_math_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (CUDA already loaded by default)
module load python/pytorch/2.2.0

# Install additional dependencies if needed
pip install --user tensorboardx pyyaml imageio

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

echo "Starting LIIF training on mathematical functions sin(2πk1*x)*sin(2πk2*y)"
echo "k1, k2 values: {2, 3, 4, 5, 6, 7, 8, 9}"
echo "Data generation is integrated into the training process"

# Run training
python train_liif.py --config configs/train-div2k/train_edsr-baseline-liif_math.yaml --gpu 0 --name math_function_training 