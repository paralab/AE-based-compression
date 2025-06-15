#!/bin/bash

echo "Setting up conda environment for LIIF project..."

# Load conda module if available
module load anaconda3_cpu 2>/dev/null || module load miniconda3 2>/dev/null || echo "Loading conda manually..."

# Initialize conda for bash if not already done
eval "$(conda shell.bash hook)" 2>/dev/null || true

# Create conda environment
ENV_NAME="liif_env"
echo "Creating conda environment: $ENV_NAME"

# Remove existing environment if it exists
conda env remove -n $ENV_NAME -y 2>/dev/null || true

# Create new environment with Python 3.10
conda create -n $ENV_NAME python=3.10 -y

# Activate the environment
conda activate $ENV_NAME

echo "Installing PyTorch with CUDA support..."
# Install PyTorch with CUDA support - using pip for more reliable CUDA installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing other dependencies..."
# Install other required packages via conda
conda install -y \
    pyyaml \
    numpy \
    pillow \
    matplotlib \
    tqdm \
    tensorboard

# Install additional packages via pip
pip install \
    tensorboardX \
    imageio \
    imageio-ffmpeg

echo "Testing imports..."
# Test all critical imports
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Number of GPUs: {torch.cuda.device_count()}')

import yaml
print(f'PyYAML imported successfully')

import tensorboardX
print('TensorboardX imported successfully')

import imageio
print('ImageIO imported successfully')

import numpy
import PIL
import matplotlib
import tqdm
print('All other imports successful!')

# Test CUDA functionality
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f'Current CUDA device: {device}')
    print(f'Device name: {torch.cuda.get_device_name(device)}')
    
    # Test creating a tensor on GPU
    x = torch.randn(3, 3).cuda()
    print('Successfully created tensor on GPU!')
else:
    print('WARNING: CUDA not available!')
"

echo "Environment setup complete!"
echo "Environment name: $ENV_NAME"
echo "To activate manually: conda activate $ENV_NAME" 