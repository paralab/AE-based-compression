#!/bin/bash
#SBATCH --job-name=swae_all_vars_inf_5x5x5_opt
#SBATCH --output=logs/swae_all_vars_inf_5x5x5_opt_%j.out
#SBATCH --error=logs/swae_all_vars_inf_5x5x5_opt_%j.err
#SBATCH --partition=gpuA40x4
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

echo "Starting SWAE 3D All Variables inference (5x5x5 OPTIMIZED version)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Default model path - can be overridden when submitting
MODEL_PATH=${MODEL_PATH:-"./save/swae_all_vars_5x5x5_opt_mlp/best_model.pth"}
ARCH=${ARCH:-"mlp"}
OUTPUT_DIR=${OUTPUT_DIR:-"test_all_vars_test_5x5x5_opt_results"}
USE_FLOAT8=${USE_FLOAT8:-"false"}

echo "Model path: $MODEL_PATH"
echo "Architecture: $ARCH"
echo "Output Directory: $OUTPUT_DIR"
echo "Use Float8 quantization: $USE_FLOAT8"

# Run inference with speed benchmarking
if [ "$USE_FLOAT8" = "true" ]; then
    FLOAT8_FLAG="--use-float8"
else
    FLOAT8_FLAG=""
fi

python inference_swae_all_variables_validation_5x5x5_opt.py \
    --data-folder "/u/tawal/BSSN-Extracted-Data/tt_q/" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --arch "$ARCH" \
    --num-samples 50 \
    --num-vti-samples 10 \
    --batch-size 32 \
    --normalize-method pos_log \
    --device auto \
    --exclude-vars \
    $FLOAT8_FLAG

echo "Inference completed!"
echo "Results saved in: $OUTPUT_DIR"
