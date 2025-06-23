#!/bin/bash
#SBATCH --job-name=swae_thera_3d_improved
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:00
#SBATCH --output=logs/swae_thera_3d_improved_%j.out
#SBATCH --error=logs/swae_thera_3d_improved_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting IMPROVED 3D SWAE+Thera training:"
echo "- 3D mathematical functions: sin(2πkx)sin(2πky)sin(2πkz)"
echo "- k values: {2, 3, 4, 5, 6} (diagonal combinations only: k1=k2=k3)"
echo "- Input resolution: 20x20x20"
echo "- Target resolution: 40x40x40 (2x super-resolution)"
echo "- IMPROVEMENTS:"
echo "  * Higher learning rate: 5e-4 (vs 1e-4)"
echo "  * Smaller batch size: 2 (vs 4)"
echo "  * Smaller model: 16,32,64 channels"
echo "  * Fewer sample points: 2000 (vs 4000)"
echo "  * Better initialization and gradient clipping"
echo "  * More frequent validation"

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

# Run training with improved configuration
python train_liif_3d.py \
    --config configs/train-3d/train_swae_thera_3d_improved.yaml \
    --name swae_thera_3d_improved \
    --gpu 0

echo "Improved 3D SWAE+Thera training completed!" 