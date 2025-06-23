#!/bin/bash

echo "3D SWAE+Thera Super-Resolution Implementation"
echo "============================================="
echo ""

# Check if we're on a compute node or login node
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected - ready for training"
    GPU_AVAILABLE=true
else
    echo "⚠ No GPU detected - running tests on CPU only"
    GPU_AVAILABLE=false
fi

echo ""
echo "Step 1: Testing implementation..."
python test_3d_implementation.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All tests passed!"
    
    if [ "$GPU_AVAILABLE" = true ]; then
        echo ""
        echo "Step 2: Starting training..."
        echo "Running: python train_liif_3d.py --config configs/train-3d/train_swae_thera_3d.yaml --name swae_thera_3d_training --gpu 0"
        echo ""
        python train_liif_3d.py \
            --config configs/train-3d/train_swae_thera_3d.yaml \
            --name swae_thera_3d_training \
            --gpu 0
    else
        echo ""
        echo "To start training on a GPU node, run:"
        echo "sbatch train_swae_thera_3d.sbatch"
        echo ""
        echo "Or for interactive training:"
        echo "python train_liif_3d.py --config configs/train-3d/train_swae_thera_3d.yaml --name swae_thera_3d_training --gpu 0"
    fi
else
    echo ""
    echo "❌ Tests failed! Please check the error messages above."
    exit 1
fi 