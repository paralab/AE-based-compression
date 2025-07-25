#!/bin/bash
#SBATCH --job-name=vis_dual_recon
#SBATCH --output=logs/vis_dual_recon_%j.out
#SBATCH --error=logs/vis_dual_recon_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting Dual Model Reconstruction Visualization"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

# Set paths
ORIGINAL_FILE="/u/tawal/BSSN-Extracted-Data/tt_q08/bssn_gr_11200_extracted.hdf5"
RECONSTRUCTED_FILE="./reconstructed/bssn_gr_11200_dual_reconstructed.hdf5"
OUTPUT_DIR="./reconstruction_plots"

echo "Original file: $ORIGINAL_FILE"
echo "Reconstructed file: $RECONSTRUCTED_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create plots for all variables
echo "Creating plots for all U_ variables..."
python visualize_dual_reconstruction.py \
    --original-file "$ORIGINAL_FILE" \
    --reconstructed-file "$RECONSTRUCTED_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --sample-indices 0 10 50 100 \
    --slice-index 3

echo ""
echo "Creating focused plots for problematic variables..."
# Create additional plots focusing on problematic variables
python visualize_dual_reconstruction.py \
    --original-file "$ORIGINAL_FILE" \
    --reconstructed-file "$RECONSTRUCTED_FILE" \
    --output-dir "${OUTPUT_DIR}/problematic_vars" \
    --variables U_B2 U_SYMAT2 U_GT2 U_SYMGT2 U_SYMAT4 U_SYMAT3 \
    --sample-indices 0 25 75 125 \
    --slice-index 3

echo ""
echo "Visualization completed!"
echo "Plots saved to: $OUTPUT_DIR"