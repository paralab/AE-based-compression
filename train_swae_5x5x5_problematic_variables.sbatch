#!/bin/bash
#SBATCH --job-name=swae_dual_transform
#SBATCH --output=logs/swae_dual_transform_%j.out
#SBATCH --error=logs/swae_dual_transform_%j.err
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcqs-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Print GPU info
nvidia-smi

echo "Starting Dual-Model SWAE Training"
echo "================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""
echo "Model 1 (poslog): Standard variables (U_ALPHA, U_K, U_GT0, U_GT1, etc.)"
echo "Model 2 (asinh): Problematic variables (U_B2, U_SYMAT2, U_GT2, U_SYMGT2, U_SYMAT4, U_SYMAT3)"
echo ""

# Run dual-model training with robust loss for problematic variables
python train_swae_5x5x5_problematic_variables.py \
    --data-folder "/u/tawal/BSSN-Extracted-Data/tt_q01/" \
    --save-dir "./save/swae_dual_transform_robust" \
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
    --use-robust-loss \
    --robust-loss-type log_cosh

echo ""
echo "Training completed!"
echo "Models saved in: ./save/swae_dual_transform_robust/"
echo "  - poslog model: ./save/swae_dual_transform_robust/poslog/"
echo "  - asinh model (with log-cosh robust loss): ./save/swae_dual_transform_robust/asinh/" 