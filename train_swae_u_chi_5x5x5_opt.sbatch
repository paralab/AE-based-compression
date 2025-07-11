#!/bin/bash
#SBATCH --job-name=swae_u_chi_5x5x5_latent4_poslog_fixed
#SBATCH --output=logs/swae_u_chi_5x5x5_opt_%j.out
#SBATCH --error=logs/swae_u_chi_5x5x5_opt_%j.err
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcqs-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Print GPU info
nvidia-smi

echo "Starting SWAE 3D U_CHI training (5x5x5 OPTIMIZED version)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Run training - MLP architecture with optimizations
python train_swae_u_chi_5x5x5_opt.py \
    --data-folder "/u/tawal/BSSN-Extracted-Data/tt_q01/" \
    --save-dir "./save/swae_u_chi_5x5x5_opt_mlp" \
    --arch mlp \
    --batch-size 256 \
    --epochs 1000 \
    --lr 2e-4 \
    --latent-dim 8 \
    --lambda-reg 1.0 \
    --normalize-method pos_log \
    --early-stopping-patience 40 \
    --eval-interval 5 \
    --save-interval 10 \
    --num-workers 8

echo "Training completed!"
echo "Model saved in: ./save/swae_u_chi_5x5x5_opt_mlp" 