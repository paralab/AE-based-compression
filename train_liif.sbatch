#!/bin/bash
#SBATCH --job-name=liif_train
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_liif_%j.out
#SBATCH --error=logs/train_liif_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (CUDA already loaded by default)
module load python/pytorch/2.2.0

# Install additional dependencies if needed
pip install --user tensorboardx pyyaml imageio

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Run training
python train_liif.py --config configs/train-div2k/train_edsr-baseline-liif_modified.yaml --gpu 0 