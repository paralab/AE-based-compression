#!/bin/bash
#SBATCH --job-name=thera_math_8x_diag
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/thera_math_8x_diag_%j.out
#SBATCH --error=logs/thera_math_8x_diag_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting Thera training with updated parameters:"
echo "- k values: {2, 3, 4, 5, 6} (diagonal combinations only: k1=k2)"
echo "- Scale range: 1x to 8x (reduced from 32x)"
echo "- Resolution: 384x384 (adjusted for 8x scaling)"
echo "- Functions: sin(2πkx)sin(2πky) for k∈{2,3,4,5,6}"
echo "- Training combinations: (2,2), (3,3), (4,4), (5,5), (6,6)"

# Load conda module if available
module load anaconda3_cpu 2>/dev/null || \
module load miniconda3 2>/dev/null || \
module load conda 2>/dev/null || \
echo "No conda module found, using system conda..."

# Initialize conda for bash
eval "$(conda shell.bash hook)" 2>/dev/null || true

# Activate the LIIF environment
echo "Activating conda environment: liif_env"
conda activate liif_env

# Verify environment is working
echo "Verifying environment..."
python -c "import torch, yaml, tensorboardX, imageio; print('Environment check passed!')"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run training
python train_liif.py \
    --config configs/train-div2k/train_edsr-baseline-thera_math.yaml \
    --name thera_math_training_8x_diagonal \
    --gpu 0

echo "Training completed!" 