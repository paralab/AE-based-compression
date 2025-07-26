#!/bin/bash
#SBATCH --job-name=swae_validation_hdf5_save
#SBATCH --output=logs/swae_validation_hdf5_save_%j.out
#SBATCH --error=logs/swae_validation_hdf5_save_%j.err
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

echo "Starting SWAE 3D All Variables Validation with HDF5 Output"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Default model path - can be overridden when submitting
MODEL_PATH=${MODEL_PATH:-"./save/swae_all_vars_5x5x5_opt_mlp/best_model.pth"}
ARCH=${ARCH:-"mlp"}
OUTPUT_DIR=${OUTPUT_DIR:-"validation_results_with_hdf5"}
USE_FLOAT8=${USE_FLOAT8:-"false"}
USE_FLOAT16=${USE_FLOAT16:-"false"}
SAMPLES_PER_VAR=${SAMPLES_PER_VAR:-"all"}

echo "Model path: $MODEL_PATH"
echo "Architecture: $ARCH"
echo "Output directory: $OUTPUT_DIR"
echo "Use Float8 quantization: $USE_FLOAT8"
echo "Use Float16 precision: $USE_FLOAT16"
echo "Samples per variable: $SAMPLES_PER_VAR"

# Run inference with speed benchmarking
if [ "$USE_FLOAT8" = "true" ]; then
    PRECISION_FLAG="--use-float8"
elif [ "$USE_FLOAT16" = "true" ]; then
    PRECISION_FLAG="--use-float16"
else
    PRECISION_FLAG=""
fi

python "compression_and_reconstruction.py" \
    --data-folder "/u/tawal/BSSN-Extracted-Data/tt_q/" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --arch "$ARCH" \
    --samples-per-variable "$SAMPLES_PER_VAR" \
    --num-vti-samples 10 \
    --batch-size 32 \
    --normalize-method pos_log \
    --device auto \
    --exclude-vars \
    $PRECISION_FLAG

echo "Inference completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "HDF5 files: all_variables_reconstructed.h5 (5x5x5) and all_variables_encoded.h5 (all variables)"
