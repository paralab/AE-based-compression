#!/bin/bash
#SBATCH --job-name=swae_log_loss
#SBATCH --output=logs/swae_log_loss_%j.out
#SBATCH --error=logs/swae_log_loss_%j.err
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcqs-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Print GPU info
nvidia-smi

echo "Starting Dual-Model SWAE Training with Log-Space Loss"
echo "===================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""
echo "Model 1 (poslog): Standard variables with log-space loss"
echo "Model 2 (asinh): Problematic variables with log-space loss"
echo "This prevents vanishing gradients when MSE becomes very small"
echo ""

# Run dual-model training with log-space loss
python train_swae_5x5x5_problematic_variables.py \
    --data-folder "/u/tawal/BSSN-Extracted-Data/tt_q01/" \
    --save-dir "./save/swae_dual_log_loss" \
    --arch mlp \
    --batch-size 256 \
    --epochs 1000 \
    --lr 2e-4 \
    --latent-dim 16 \
    --lambda-reg 1.0 \
    --early-stopping-patience 40 \
    --eval-interval 5 \
    --save-interval 10 \
    --num-workers 8 \
    --use-log-loss \
    --loss-scale 1000.0

echo ""
echo "Training completed!"
echo "Models saved in: ./save/swae_dual_log_loss/"
echo "  - poslog model: ./save/swae_dual_log_loss/poslog/"
echo "  - asinh model: ./save/swae_dual_log_loss/asinh/"