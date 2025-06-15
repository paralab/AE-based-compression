#!/bin/bash
#SBATCH --job-name=liif_demo
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/demo_inference_%j.out
#SBATCH --error=logs/demo_inference_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p outputs

# Load necessary modules
module load python/pytorch/2.2.0

# Install additional dependencies if needed
pip install --user tensorboardx pyyaml imageio

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

echo "Running LIIF inference with trained model..."
echo "Model: save/test_run/epoch-best.pth"
echo "Input: load/div2k/DIV2K_valid_HR/0855.png"

# Demo inference at different resolutions
echo "=== Generating super-resolution at 1024x1024 (2x) ==="
python demo.py --input load/div2k/DIV2K_valid_HR/0855.png \
               --model save/test_run/epoch-best.pth \
               --resolution 1024,1024 \
               --output outputs/demo_output_1024x1024.png \
               --gpu 0

echo "=== Generating super-resolution at 2048x2048 (4x) ==="
python demo.py --input load/div2k/DIV2K_valid_HR/0855.png \
               --model save/test_run/epoch-best.pth \
               --resolution 2048,2048 \
               --output outputs/demo_output_2048x2048.png \
               --gpu 0

echo "=== Generating super-resolution at 4096x4096 (8x) ==="
python demo.py --input load/div2k/DIV2K_valid_HR/0855.png \
               --model save/test_run/epoch-best.pth \
               --resolution 4096,4096 \
               --output outputs/demo_output_4096x4096.png \
               --gpu 0

echo "=== Testing different input image ==="
python demo.py --input load/div2k/DIV2K_valid_HR/0802.png \
               --model save/test_run/epoch-best.pth \
               --resolution 1536,1536 \
               --output outputs/demo_output_0802_1536x1536.png \
               --gpu 0

echo "=== Inference completed! ==="
echo "Generated outputs:"
ls -la outputs/demo_output*.png

echo "Original image info:"
python -c "from PIL import Image; img = Image.open('load/div2k/DIV2K_valid_HR/0855.png'); print(f'Original 0855.png size: {img.size}')"

echo "Super-resolution results info:"
python -c "
from PIL import Image
import os

files = [f for f in os.listdir('outputs/') if f.startswith('demo_output')]
for f in sorted(files):
    img = Image.open(f'outputs/{f}')
    print(f'{f}: {img.size}')" 