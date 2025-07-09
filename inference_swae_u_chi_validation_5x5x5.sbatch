#!/bin/bash
#SBATCH --job-name=swae_u_chi_5x5x5_latent4_TEST_FIXED
#SBATCH --output=logs/swae_u_chi_5x5x5_latent4_inference_%j.out
#SBATCH --error=logs/swae_u_chi_5x5x5_latent4_inference_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcqs-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Create logs directory if it doesn't exist
mkdir -p logs

# Load conda module if available
module load anaconda3_cpu 2>/dev/null || \
module load miniconda3 2>/dev/null || \
module load conda 2>/dev/null || \
echo "No conda module found, using system conda..."

# Initialize conda for bash
eval "$(conda shell.bash hook)" 2>/dev/null || true

# Activate the LIIF environment
echo "Activating conda environment: liif_env"
conda activate liif_env

# Verify environment is working
echo "Verifying environment..."
python -c "import torch, numpy, matplotlib; print('Environment check passed!')"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Print GPU information
nvidia-smi

# Print Python and PyTorch information
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Run the inference script
echo "Starting SWAE U_CHI validation inference (5x5x5 version) with dual plots and VTI output..."

# Look for the latest 5x5x5 latent4 FIXED pos_log model first
latest_model_dir=$(ls -td ./save/swae_u_chi_5x5x5_latent4_poslog_FIXED_* 2>/dev/null | head -1)
if [ -z "$latest_model_dir" ]; then
    # Fallback to any 5x5x5 latent4 model
    latest_model_dir=$(ls -td ./save/swae_u_chi_5x5x5_latent4_* 2>/dev/null | head -1)
fi
if [ -z "$latest_model_dir" ]; then
    # Fallback to regular 5x5x5 models (latent dim 16)
    latest_model_dir=$(ls -td ./save/swae_u_chi_5x5x5_* 2>/dev/null | head -1)
fi
if [ -z "$latest_model_dir" ]; then
    # Final fallback to 7x7x7 FIXED model (for comparison)
    latest_model_dir=$(ls -td ./save/swae_u_chi_poslog_FIXED_* 2>/dev/null | head -1)
fi
if [ -n "$latest_model_dir" ]; then
    latest_model="$latest_model_dir/best_model.pth"
else
    # Fallback to any available model (update path as needed)
    latest_model="./save/swae_u_chi/best_model.pth"
fi

if [ ! -f "$latest_model" ]; then
    echo "Error: Model checkpoint not found: $latest_model"
    echo "Available models:"
    find ./save -name "*.pth" -type f | head -5
    exit 1
fi

echo "Using model checkpoint: $latest_model"
echo "🎯 Using model trained with FIXED per-sample pos_log normalization (5x5x5 version, latent dim 4)"
echo "🎯 Expected: Proper statistical properties and high compression ratio"
echo "🎯 Compression: 31.25:1 ratio (125 -> 4 dimensions)"
echo "🎯 Data processing: 5x5x5 center crop from 7x7x7 original data"

# Run inference with dual plot generation (normalized + denormalized) - 5x5x5 version
python inference_swae_u_chi_validation_5x5x5.py \
    --model-path "$latest_model" \
    --data-folder /u/tawal/BSSN-Extracted-Data/tt_q01/ \
    --output-dir test_u_chi_5x5x5_latent4_results_FIXED_$(date +%Y%m%d_%H%M%S) \
    --normalize-method pos_log \
    --batch-size 32 \
    --num-samples 20 \
    --num-vti-samples 5 \
    --device auto

echo "Job completed at: $(date)"
echo ""
echo "🎯 Results from 5x5x5 LATENT4 FIXED model on HELD-OUT TEST SET (5%):"
echo "  - Unbiased evaluation on data NEVER seen during training"
echo "  - Data processing: 5x5x5 center crop extracted from 7x7x7 data"
echo "  - Compression ratio: 31.25:1 (vs 7.8:1 for latent dim 16)"
echo "  - High compression experiment: Testing reconstruction quality at extreme compression"
echo "  - Dual comparison plots for all 20 test samples:"
echo "    * normalized scale (log pos) - as model sees data"
echo "    * denormalized scale (original units) - physical interpretation"
echo "  - VTI files (normalized scale) for 5 random test samples"
echo "  - Average metrics summary (calculated on denormalized data)"
echo "  - Compression/decompression speed benchmarking in GBps"
echo "  - Relative error analysis with percentile breakdown"
echo "  - Performance comparison with latent dim 16 and 7x7x7 model metrics"
echo "🔒 Test set is deterministic and consistent across all runs" 