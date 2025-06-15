#!/bin/bash

echo "=================================================="
echo "LIIF Quickstart Guide - After Training Completes"
echo "=================================================="
echo ""

# Check if training has completed
if [ -f "save/_train_edsr-baseline-liif_test_test_run/epoch-best.pth" ]; then
    MODEL_PATH="save/_train_edsr-baseline-liif_test_test_run/epoch-best.pth"
    echo "✓ Training completed! Model found at: $MODEL_PATH"
    echo ""
    
    echo "1. Quick demo inference (super-resolution):"
    echo "   python demo.py --input load/div2k/DIV2K_valid_HR/0801.png --model $MODEL_PATH --resolution 1024,1024 --output outputs/demo_output.png --gpu 0"
    echo ""
    
    echo "2. Test on DIV2K validation set:"
    echo "   bash scripts/test-div2k.sh $MODEL_PATH 0"
    echo ""
    
    echo "3. Test on benchmark datasets:"
    echo "   bash scripts/test-benchmark.sh $MODEL_PATH 0"
    echo ""
    
    echo "4. Submit inference job to cluster:"
    echo "   sbatch demo_inference.sbatch"
    echo ""
    
    echo "5. Submit benchmark testing job:"
    echo "   sbatch test_benchmark.sbatch"
    echo ""
    
else
    echo "⏳ Training not yet completed."
    echo "Check job status with: squeue -u $USER"
    echo "Monitor training progress with: tail -f logs/train_liif_test_*.out"
    echo ""
    echo "When training completes, the model will be saved to:"
    echo "   save/_train_edsr-baseline-liif_test_test_run/epoch-best.pth"
    echo ""
fi

echo "==================================================" 