#!/bin/bash
#SBATCH --job-name=swae_3d_pure
#SBATCH --output=logs/swae_3d_pure_%j.out
#SBATCH --error=logs/swae_3d_pure_%j.err
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcqs-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Create logs directory
mkdir -p logs

# Use current Python environment (pyenv)

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "Working Directory: $(pwd)"

# Print GPU information
nvidia-smi

# Run training with paper-specified parameters
python train_swae_3d_pure.py \
    --k-values 1 2 3 4 \
    --resolution 40 \
    --block-size 8 \
    --latent-dim 16 \
    --lambda-reg 1.0 \
    --batch-size 16 \
    --epochs 200 \
    --lr 1e-4 \
    --train-split 0.8 \
    --num-workers 4 \
    --device auto \
    --save-dir ./save/swae_3d_pure_$(date +%Y%m%d_%H%M%S) \
    --eval-interval 10 \
    --save-interval 20

echo "Training completed at $(date)" 