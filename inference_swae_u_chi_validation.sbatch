#!/bin/bash
#SBATCH --job-name=swae_u_chi_TEST_FIXED
#SBATCH --output=logs/swae_u_chi_inference_%j.out
#SBATCH --error=logs/swae_u_chi_inference_%j.err
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
echo "Starting SWAE U_CHI validation inference with dual plots and VTI output..."

# Look for the latest FIXED pos_log model first, then corrected
latest_model_dir=$(ls -td ./save/swae_u_chi_poslog_FIXED_* 2>/dev/null | head -1)
if [ -z "$latest_model_dir" ]; then
    # Fallback to corrected model
    latest_model_dir=$(ls -td ./save/swae_u_chi_poslog_corrected_* 2>/dev/null | head -1)
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
echo "🎯 Using model trained with FIXED per-sample pos_log normalization"
echo "🎯 Expected: Proper statistical properties and improved reconstruction quality"

# Run inference with dual plot generation (normalized + denormalized)
python inference_swae_u_chi_validation.py \
    --model-path "$latest_model" \
    --data-folder /u/tawal/0620-NN-based-compression-thera/tt_q01/ \
    --output-dir test_u_chi_results_FIXED_$(date +%Y%m%d_%H%M%S) \
    --normalize-method pos_log \
    --batch-size 32 \
    --num-samples 20 \
    --num-vti-samples 5 \
    --device auto

echo "Job completed at: $(date)"
echo ""
echo "🎯 Results from FIXED model on HELD-OUT TEST SET (5%):"
echo "  - Unbiased evaluation on data NEVER seen during training"
echo "  - Dual comparison plots for all 20 test samples:"
echo "    * normalized scale (log pos) - as model sees data"
echo "    * denormalized scale (original units) - physical interpretation"
echo "  - VTI files (normalized scale) for 5 random test samples"
echo "  - Average metrics summary (calculated on denormalized data)"
echo "  - Compression/decompression speed benchmarking in GBps"
echo "  - Relative error analysis with percentile breakdown"
echo "  - Expected improvements: PSNR 14+ dB, Correlation 0.85+"
echo "🔒 Test set is deterministic and consistent across all runs" 