#!/bin/bash
#SBATCH --job-name=thera_math_train
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/thera_math_train_%j.out
#SBATCH --error=logs/thera_math_train_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules
module load cuda/11.8
module load python/3.10

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run training
python train_liif.py \
    --config configs/train-div2k/train_edsr-baseline-thera_math.yaml \
    --name thera_math_training \
    --gpu 0

echo "Training completed!" 