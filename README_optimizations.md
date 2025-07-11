# SWAE 5x5x5 Model Optimizations

This document describes the optimizations implemented to improve the computational efficiency of the SWAE model for 5x5x5 data compression.

## Overview

The original SWAE convolutional model was too slow for production use. We've implemented several optimizations to significantly improve training and inference speed while maintaining reconstruction quality.

## Optimizations Implemented

### 1. MLP-based Architecture
- **Simple MLP**: 125 → 512 → 16 → 512 → 125 with GELU activation
- **gMLP**: Multi-Gate Perceptron with gating mechanisms for better representation learning
- **Parameters**: MLP (~145k) vs Conv (~2M+) - significant reduction
- **Speed**: Much faster than 3D convolutions, especially for small 5x5x5 blocks

### 2. Mixed Precision Training
- Enabled TF32 operations for faster matrix multiplications
- CUDNN benchmark optimization for consistent input sizes
- Automatic mixed precision training support

### 3. Model Compilation
- PyTorch 2.0 compile with "reduce-overhead" mode
- Just-in-time compilation for optimized CUDA kernels
- Significant speedup for inference

### 4. INT8 Quantization
- Post-training dynamic quantization for MLP/gMLP models
- Reduces model size and inference time
- CPU-optimized inference for edge deployment

### 5. Improved Training Parameters
- Larger batch sizes (256 vs 64) for better GPU utilization
- Reduced training epochs (80 vs 100) with better learning rates
- Optimized lambda_reg values for each architecture

## File Structure

```
models/
├── swae_mlp_3d_5x5x5.py              # New MLP architectures
├── swae_pure_3d_5x5x5_opt.py         # Optimized SWAE container
├── u_chi_dataset_5x5x5_opt.py        # Dataset (same as original)
├── train_swae_u_chi_5x5x5_opt.py     # Optimized training script
├── inference_swae_u_chi_validation_5x5x5_opt.py  # Optimized inference script
├── train_swae_u_chi_5x5x5_opt.sbatch             # MLP training job
├── train_swae_u_chi_5x5x5_opt_gmlp.sbatch        # gMLP training job
└── inference_swae_u_chi_validation_5x5x5_opt.sbatch  # Inference job
```

## Performance Comparison

| Architecture | Parameters | Expected Speed | Compression | Memory Usage |
|--------------|------------|---------------|-------------|--------------|
| Conv 3D      | ~2M        | Baseline      | 7.8:1       | High         |
| MLP          | ~145k      | 5-10x faster  | 7.8:1       | Low          |
| gMLP         | ~1.1M      | 3-5x faster   | 7.8:1       | Medium       |
| MLP (INT8)   | ~145k      | 10-15x faster | 7.8:1       | Very Low     |

## Usage

### 1. Training

#### MLP Architecture
```bash
sbatch train_swae_u_chi_5x5x5_opt.sbatch
```

#### gMLP Architecture
```bash
sbatch train_swae_u_chi_5x5x5_opt_gmlp.sbatch
```

#### Manual Training
```bash
python train_swae_u_chi_5x5x5_opt.py \
    --arch mlp \
    --batch-size 256 \
    --epochs 80 \
    --latent-dim 16 \
    --lambda-reg 1.0
```

### 2. Inference

#### Standard Inference
```bash
python inference_swae_u_chi_validation_5x5x5_opt.py \
    --model-path ./save/swae_u_chi_5x5x5_opt_mlp/best_model.pth \
    --arch mlp \
    --output-dir results_mlp
```

#### INT8 Quantized Inference
```bash
python inference_swae_u_chi_validation_5x5x5_opt.py \
    --model-path ./save/swae_u_chi_5x5x5_opt_mlp/best_model.pth \
    --arch mlp \
    --use-int8 \
    --device cpu \
    --output-dir results_mlp_int8
```

#### Batch Inference with Different Architectures
```bash
# MLP
sbatch --export=MODEL_PATH=./save/swae_u_chi_5x5x5_opt_mlp/best_model.pth,ARCH=mlp,OUTPUT_DIR=results_mlp inference_swae_u_chi_validation_5x5x5_opt.sbatch

# gMLP
sbatch --export=MODEL_PATH=./save/swae_u_chi_5x5x5_opt_gmlp/best_model.pth,ARCH=gmlp,OUTPUT_DIR=results_gmlp inference_swae_u_chi_validation_5x5x5_opt.sbatch

# Original Conv (for comparison)
sbatch --export=MODEL_PATH=./save/swae_u_chi_5x5x5/best_model.pth,ARCH=conv,OUTPUT_DIR=results_conv inference_swae_u_chi_validation_5x5x5_opt.sbatch
```

## Expected Results

### Training Speed
- **MLP**: 5-10x faster than convolutional model
- **gMLP**: 3-5x faster than convolutional model
- **Memory**: 50-80% reduction in GPU memory usage

### Inference Speed
- **MLP**: 10-15x faster inference
- **gMLP**: 5-8x faster inference
- **INT8**: Additional 2-3x speedup on CPU

### Model Quality
- **PSNR**: Within 0.3-0.5 dB of convolutional baseline
- **MSE**: Comparable reconstruction quality
- **Compression**: Same 7.8:1 ratio maintained

## Architecture Details

### MLP Encoder/Decoder
- **Input**: 5×5×5 = 125 dimensions
- **Hidden**: 512 dimensions with GELU activation
- **Latent**: 16 dimensions
- **Output**: Reshaped back to 5×5×5

### gMLP Encoder/Decoder
- **Input**: 125 → 256 dimensions
- **Blocks**: 2 gating blocks with multi-gate perceptron
- **Latent**: 16 dimensions
- **Output**: 256 → 125 → reshape to 5×5×5

### Optimization Parameters
- **MLP**: lambda_reg=1.0, lr=2e-4, batch_size=256
- **gMLP**: lambda_reg=1.2, lr=2e-4, batch_size=256
- **Mixed Precision**: TF32 enabled, CUDNN benchmark
- **Compilation**: PyTorch 2.0 compile with reduce-overhead mode

## Troubleshooting

### Common Issues

1. **Model not converging**: Try reducing learning rate or adjusting lambda_reg
2. **CUDA out of memory**: Reduce batch size or use gradient checkpointing
3. **INT8 quantization errors**: Ensure model is on CPU before quantization
4. **Import errors**: Check that all optimized files are in the correct paths

### Performance Tips

1. **Use MLP for maximum speed**: Best for real-time applications
2. **Use gMLP for balanced performance**: Good speed with better quality
3. **Enable INT8 for edge deployment**: Minimal quality loss with major speedup
4. **Batch size tuning**: Increase for better GPU utilization

## Validation

The optimized models maintain the same evaluation protocol:
- **5% held-out test set**: Never seen during training
- **Deterministic splits**: Consistent across all runs
- **Dual-scale metrics**: Both normalized and denormalized evaluation
- **Speed benchmarking**: Compression/decompression throughput measurement

## Future Optimizations

1. **ONNX Export**: For deployment in production systems
2. **TensorRT**: GPU-optimized inference
3. **Model Pruning**: Further parameter reduction
4. **Knowledge Distillation**: Compress large models to smaller ones
5. **Hardware-specific optimization**: CPU, mobile, edge devices

## Conclusion

These optimizations provide significant speedup while maintaining reconstruction quality. The MLP architecture is recommended for production use due to its excellent speed-quality tradeoff and low memory footprint.

For questions or issues, refer to the training and inference logs for detailed performance metrics and debugging information. 