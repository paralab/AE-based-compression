#!/bin/bash
# Script to run SWAE compression and reconstruction on folders

echo "Running SWAE Compression & Reconstruction on folder"

# Set the data folder
DATA_FOLDER="/u/tawal/BSSN-Extracted-Data/tt_q/"

# Set output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="compression_reconstruction_results_${TIMESTAMP}"

# Default model (MLP architecture)
MODEL_PATH="./save/swae_all_vars_5x5x5_opt_mlp/best_model.pth"
ARCH="mlp"
SAMPLES_PER_VAR="all"  # Process all samples

echo "=================================="
echo "SWAE Compression & Reconstruction"
echo "=================================="
echo "Data folder: $DATA_FOLDER"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL_PATH"
echo "Architecture: $ARCH"
echo "Samples per variable: $SAMPLES_PER_VAR"
echo ""

# Submit the compression and reconstruction job
echo "Submitting SLURM job for folder processing..."
sbatch --export=MODEL_PATH=$MODEL_PATH,OUTPUT_DIR=$OUTPUT_DIR,ARCH=$ARCH,SAMPLES_PER_VAR=$SAMPLES_PER_VAR compression_and_reconstruction.sbatch

echo ""
echo "Job submitted! Check logs/swae_validation_hdf5_save_*.out for progress"
echo ""
echo "Output files will be saved as:"
echo "  ${OUTPUT_DIR}/all_variables_reconstructed.h5 - All reconstructed data"
echo "  ${OUTPUT_DIR}/all_variables_encoded.h5 - All encoded latent vectors"
echo "  ${OUTPUT_DIR}/<var_name>/*.png - Comparison plots (first 10 samples per variable)"
echo ""
echo "Note: Processing ALL samples from all HDF5 files in chronological order"
echo ""
echo "Output file structure:"
echo "  Reconstructed file:"
echo "    - var_data: reconstructed 7x7x7 blocks"
echo "    - Original metadata and structure preserved"
echo "  Encoded file:"
echo "    - <var_name>/latent_vectors: compressed representations"
echo "    - Per-variable reconstruction metrics"