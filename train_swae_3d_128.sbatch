#!/bin/bash
#SBATCH --job-name=swae_3d_128
#SBATCH --output=logs/swae_3d_128_%j.out
#SBATCH --error=logs/swae_3d_128_%j.err
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcqs-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

# Create logs directory
mkdir -p logs

echo "Starting SWAE 3D training for 128x128x128 resolution:"
echo "- 3D mathematical functions: sin(2πkx)sin(2πky)sin(2πkz)"
echo "- k values: {2, 3, 4, 5, 6} (diagonal combinations: k1=k2=k3)"
echo "- Resolution: 128x128x128 (vs 40x40x40 baseline)"
echo "- Blocks per sample: 16^3 = 4,096 blocks (vs 125 for 40x40x40)"
echo "- Same compression: 32:1 per 8x8x8 block"
echo "- Overall compression: ~32:1 for full volume"
echo "- Computational challenge: ~32x more blocks per sample"

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
python -c "import torch, numpy; print('Environment check passed!')"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Set CUDA device
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

# Run training with parameters optimized for 128x128x128
python train_swae_3d_128.py \
    --k-values 2 3 4 5 6 \
    --resolution 128 \
    --block-size 8 \
    --latent-dim 16 \
    --lambda-reg 1.0 \
    --batch-size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --train-split 0.8 \
    --num-workers 4 \
    --device auto \
    --save-dir ./save/swae_3d_128_$(date +%Y%m%d_%H%M%S) \
    --eval-interval 10 \
    --save-interval 10

echo "128x128x128 SWAE 3D training completed at $(date)" 