#!/bin/bash
#SBATCH --job-name=swae_adaptive
#SBATCH --output=logs/swae_adaptive_%j.out
#SBATCH --error=logs/swae_adaptive_%j.err
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

echo "Starting SWAE Training with Adaptive Scaling"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""
echo "This approach scales each variable to a learnable range [-10, 10]"
echo "regardless of their original scale (including 10^-8 values)"
echo ""

# Run training with adaptive scaling
python train_swae_adaptive_scaling.py \
    --data-folder "/u/tawal/BSSN-Extracted-Data/tt_q01/" \
    --save-dir "./save/swae_adaptive_scaling" \
    --arch mlp \
    --batch-size 256 \
    --epochs 200 \
    --lr 2e-4 \
    --latent-dim 16 \
    --lambda-reg 1.0 \
    --target-range 10.0 \
    --eval-interval 10 \
    --num-workers 8

echo ""
echo "Training completed!"
echo "Models saved in: ./save/swae_adaptive_scaling/"
echo "Scaling parameters saved in: ./scaling_params.json"