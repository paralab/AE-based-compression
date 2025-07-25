#!/bin/bash
#SBATCH --job-name=dual_inference
#SBATCH --output=logs/dual_inference_%j.out
#SBATCH --error=logs/dual_inference_%j.err
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcqs-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Print GPU info
nvidia-smi

echo "Starting Dual Model Inference"
echo "============================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

# Set paths
INPUT_FILE="/u/tawal/BSSN-Extracted-Data/tt_q08/bssn_gr_11200_extracted.hdf5"
OUTPUT_FILE="./reconstructed/bssn_gr_11200_log_loss_reconstructed.hdf5"
LATENT_FILE="./reconstructed/bssn_gr_11200_log_loss_latent_codes.npz"

# Model checkpoints - using the log-loss models
POSLOG_MODEL="./save/swae_dual_log_loss/poslog/best_model.pth"
ASINH_MODEL="./save/swae_dual_log_loss/asinh/best_model.pth"

# Create output directory
mkdir -p ./reconstructed

echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Poslog model: $POSLOG_MODEL"
echo "Asinh model: $ASINH_MODEL"
echo ""

# Run inference
python inference_dual_models_hdf5.py \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --poslog-checkpoint "$POSLOG_MODEL" \
    --asinh-checkpoint "$ASINH_MODEL" \
    --latent-file "$LATENT_FILE" \
    --device cuda \
    --batch-size 256 \
    --arch mlp

echo ""
echo "Inference completed!"
echo "Reconstructed file: $OUTPUT_FILE"
echo "Latent codes: $LATENT_FILE"

# Create visualization plots
echo ""
echo "Creating visualization plots..."
echo "=============================="

# Set plot output directory
PLOT_DIR="./reconstruction_plots_log_loss"
mkdir -p $PLOT_DIR

# Run visualization
python visualize_dual_reconstruction.py \
    --original-file "$INPUT_FILE" \
    --reconstructed-file "$OUTPUT_FILE" \
    --output-dir "$PLOT_DIR" \
    --sample-indices 0 10 50 100 \
    --slice-index 3

# Create focused plots for problematic variables
echo ""
echo "Creating focused plots for problematic variables..."
mkdir -p "${PLOT_DIR}/problematic_vars"

python visualize_dual_reconstruction.py \
    --original-file "$INPUT_FILE" \
    --reconstructed-file "$OUTPUT_FILE" \
    --output-dir "${PLOT_DIR}/problematic_vars" \
    --variables U_B2 U_SYMAT2 U_GT2 U_SYMGT2 U_SYMAT4 U_SYMAT3 \
    --sample-indices 0 25 75 125 \
    --slice-index 3

echo ""
echo "All processing completed!"
echo "========================"
echo "Reconstructed file: $OUTPUT_FILE"
echo "Latent codes: $LATENT_FILE"
echo "Plots saved to: $PLOT_DIR"