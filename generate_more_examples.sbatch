#!/bin/bash
#SBATCH --job-name=liif_more_examples
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/more_examples_%j.out
#SBATCH --error=logs/more_examples_%j.err

# Create directories
mkdir -p logs
mkdir -p outputs/more_examples

# Load necessary modules
module load python/pytorch/2.2.0

# Install additional dependencies if needed
pip install --user tensorboardx pyyaml imageio

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

echo "=== GENERATING MORE LIIF INFERENCE EXAMPLES ==="
echo "Model: save/test_run/epoch-best.pth"
echo "Target: 8 new inference examples with different resolutions"
echo ""

# Generate diverse examples with different resolutions
echo "1/8: Generating 0803 at 512x512..."
python demo.py --input load/div2k/DIV2K_valid_HR/0803.png \
               --model save/test_run/epoch-best.pth \
               --resolution 512,512 \
               --output outputs/more_examples/0803_512x512.png \
               --gpu 0

echo "2/8: Generating 0803 at 1280x1280..."
python demo.py --input load/div2k/DIV2K_valid_HR/0803.png \
               --model save/test_run/epoch-best.pth \
               --resolution 1280,1280 \
               --output outputs/more_examples/0803_1280x1280.png \
               --gpu 0

echo "3/8: Generating 0804 at 768x768..."
python demo.py --input load/div2k/DIV2K_valid_HR/0804.png \
               --model save/test_run/epoch-best.pth \
               --resolution 768,768 \
               --output outputs/more_examples/0804_768x768.png \
               --gpu 0

echo "4/8: Generating 0804 at 1600x1600..."
python demo.py --input load/div2k/DIV2K_valid_HR/0804.png \
               --model save/test_run/epoch-best.pth \
               --resolution 1600,1600 \
               --output outputs/more_examples/0804_1600x1600.png \
               --gpu 0

echo "5/8: Generating 0805 at 640x640..."
python demo.py --input load/div2k/DIV2K_valid_HR/0805.png \
               --model save/test_run/epoch-best.pth \
               --resolution 640,640 \
               --output outputs/more_examples/0805_640x640.png \
               --gpu 0

echo "6/8: Generating 0805 at 1920x1080 (widescreen)..."
python demo.py --input load/div2k/DIV2K_valid_HR/0805.png \
               --model save/test_run/epoch-best.pth \
               --resolution 1920,1080 \
               --output outputs/more_examples/0805_1920x1080.png \
               --gpu 0

echo "7/8: Generating 0806 at 800x600..."
python demo.py --input load/div2k/DIV2K_valid_HR/0806.png \
               --model save/test_run/epoch-best.pth \
               --resolution 800,600 \
               --output outputs/more_examples/0806_800x600.png \
               --gpu 0

echo "8/8: Generating 0807 at 1333x1333..."
python demo.py --input load/div2k/DIV2K_valid_HR/0807.png \
               --model save/test_run/epoch-best.pth \
               --resolution 1333,1333 \
               --output outputs/more_examples/0807_1333x1333.png \
               --gpu 0

echo ""
echo "=== INFERENCE COMPLETED! ==="
echo "Generated files:"
ls -la outputs/more_examples/

echo ""
echo "File sizes and resolutions:"
python -c "
import os
from PIL import Image
files = [f for f in os.listdir('outputs/more_examples/') if f.endswith('.png')]
print(f'{'Filename':<25} {'Resolution':<15} {'Size (MB)':<10}')
print('-' * 50)
total_size = 0
for filename in sorted(files):
    path = f'outputs/more_examples/{filename}'
    img = Image.open(path)
    size_mb = os.path.getsize(path) / (1024*1024)
    print(f'{filename:<25} {img.size[0]}×{img.size[1]:<15} {size_mb:<10.1f}')
    total_size += size_mb
print('-' * 50)
print(f'{'TOTAL':<25} {'':<15} {total_size:<10.1f}')
"

echo ""
echo "✅ Successfully generated 8 additional LIIF inference examples!"
echo "   Demonstrating arbitrary resolution capabilities"
echo "   From 512×512 to 1920×1080 and 1600×1600" 