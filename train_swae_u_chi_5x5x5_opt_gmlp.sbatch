#!/bin/bash
#SBATCH --job-name=swae_u_chi_5x5x5_opt_gmlp
#SBATCH --output=logs/swae_u_chi_5x5x5_opt_gmlp_%j.out
#SBATCH --error=logs/swae_u_chi_5x5x5_opt_gmlp_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Print GPU info
nvidia-smi

echo "Starting SWAE 3D U_CHI training (5x5x5 OPTIMIZED version - gMLP)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Run training - gMLP architecture with optimizations
python train_swae_u_chi_5x5x5_opt.py \
    --data-folder "/u/tawal/BSSN-Extracted-Data/tt_q01/" \
    --save-dir "./save/swae_u_chi_5x5x5_opt_gmlp" \
    --arch gmlp \
    --batch-size 256 \
    --epochs 80 \
    --lr 2e-4 \
    --latent-dim 16 \
    --lambda-reg 1.2 \
    --normalize-method pos_log \
    --early-stopping-patience 20 \
    --eval-interval 5 \
    --save-interval 10 \
    --num-workers 8

echo "Training completed!"
echo "Model saved in: ./save/swae_u_chi_5x5x5_opt_gmlp" 