#!/bin/bash
#SBATCH --job-name=swae_u_chi_7x7x7_poslog
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
echo "Dataset: /u/tawal/0620-NN-based-compression-thera/tt_q01/"
echo "Total samples: ~29K (23K train, 6K val)"
echo "Compression ratio: 21.4:1 (343 -> 16 dimensions)"
echo "Normalization: pos_log (corrected implementation)"

# Run U_CHI SWAE training with corrected pos_log normalization
python train_swae_u_chi.py \
    --data-folder /u/tawal/0620-NN-based-compression-thera/tt_q01/ \
    --normalize \
    --normalize-method pos_log \
    --latent-dim 16 \
    --lambda-reg 0.9 \
    --batch-size 128 \
    --epochs 1000 \
    --lr 2e-4 \
    --train-split 0.8 \
    --num-workers 8 \
    --device auto \
    --save-dir ./save/swae_u_chi_poslog_corrected_$(date +%Y%m%d_%H%M%S) \
    --eval-interval 10 \
    --save-interval 25

echo "Training completed at $(date)"
echo "Check results in save directory"
echo "Model trained with corrected pos_log normalization (exact match with visualize_uchi_log_scale.py)" 