#!/bin/bash
#SBATCH --job-name=thera_math_train
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/thera_math_train_%j.out
#SBATCH --error=logs/thera_math_train_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting LIIF Thera Math Training..."

# Load conda module (required for cluster)
module load anaconda3_cpu 2>/dev/null || \
module load miniconda3 2>/dev/null || \
module load conda 2>/dev/null || \
echo "No conda module found, trying system conda..."

# Initialize conda for this shell session
eval "$(conda shell.bash hook)" 2>/dev/null || true

# Activate the LIIF environment
echo "Activating conda environment: liif_env"
conda activate liif_env

# Verify environment is working
echo "Verifying CUDA-enabled PyTorch environment..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Number of GPUs: {torch.cuda.device_count()}')

import yaml, tensorboardX, imageio
print('All required packages imported successfully!')

if torch.cuda.is_available():
    print(f'Current CUDA device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name()}')
else:
    raise RuntimeError('CUDA not available - environment setup failed')
"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

echo "Starting training with Thera Math configuration..."

# Run training
python train_liif.py \
    --config configs/train-div2k/train_edsr-baseline-thera_math.yaml \
    --name thera_math_training \
    --gpu 0

echo "Training completed!" 