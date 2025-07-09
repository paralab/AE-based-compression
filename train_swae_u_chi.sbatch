#!/bin/bash
#SBATCH --job-name=swae_u_chi_7x7x7_poslog_fixed
#SBATCH --output=logs/swae_u_chi_%j.out
#SBATCH --error=logs/swae_u_chi_%j.err
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
python -c "import torch, numpy, matplotlib; print('Environment check passed!')"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

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

echo "Starting SWAE training on U_CHI dataset (7x7x7 blocks)"
echo "Dataset: /u/tawal/BSSN-Extracted-Data/tt_q01/"
echo "Data splits: 80% train (~23K), 15% val (~4K), 5% test (~1.5K)"
echo "🔒 Test set (5%) held out for final unbiased evaluation"
echo "Compression ratio: 21.4:1 (343 -> 16 dimensions)"
echo "Normalization: pos_log (FIXED per-sample implementation)"
echo "⚠️  CRITICAL FIX: Per-sample log transformation for correct statistical properties"
echo "   Expected: mean~4.5, std~1.1 (was incorrectly mean~-0.66, std~0.6)"

# Run U_CHI SWAE training with FIXED per-sample pos_log normalization
# ADJUSTED HYPERPARAMETERS for new data distribution:
# - Reduced learning rate from 2e-4 to 1e-4 (more stable for new distribution)
# - Increased gradient clipping from 1.0 to 2.0 (handle larger gradients) 
# - Reduced batch size from 128 to 64 (more stable training)
# - Keep lambda_reg at 0.9 (should still work well)
# - Using deeper architecture with channels [32,64,128,256]
# - Early stopping with 40 epochs patience
python train_swae_u_chi.py \
    --data-folder /u/tawal/BSSN-Extracted-Data/tt_q01/ \
    --normalize \
    --normalize-method pos_log \
    --latent-dim 16 \
    --lambda-reg 0.9 \
    --encoder-channels "32,64,128,256" \
    --batch-size 64 \
    --epochs 1000 \
    --lr 1e-4 \
    --train-split 0.8 \
    --num-workers 8 \
    --device auto \
    --save-dir ./save/swae_u_chi_poslog_FIXED_$(date +%Y%m%d_%H%M%S) \
    --eval-interval 10 \
    --save-interval 25 \
    --early-stopping-patience 40 \
    --early-stopping-min-delta 1e-4

echo "Training completed at $(date)"
echo "Check results in save directory"
echo "🎯 Model trained with FIXED per-sample pos_log normalization"
echo "🎯 Expected improvements: Proper statistical properties, better reconstruction quality, and deeper architecture" 