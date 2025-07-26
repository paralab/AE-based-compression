# Neural Network-based Compression with Autoencoders (SWAE)

This repository contains implementations of Sliced Wasserstein Autoencoders (SWAE) for compressing scientific data, specifically focusing on gravitational wave simulation data from the BSSN formalism.

## Overview

The project implements various SWAE architectures to compress 5x5x5 subvolumes of simulation variables. The primary goal is to achieve efficient compression while maintaining high reconstruction quality, enabling fast data transfer across different network interconnects.

## Architecture Implementations

### 1. CNN SWAE Architecture

The convolutional neural network implementation with optimized architecture:

**Encoder Architecture:**
```
Input (5×5×5×1) → Conv3D(32) → ReLU → Conv3D(64) → ReLU → 
Conv3D(128) → ReLU → Conv3D(256) → ReLU → Flatten → 
Linear(2048→16) → Latent(16D)
```

**Decoder Architecture:**
```
Latent(16D) → Linear(16→2048) → Reshape(2×2×2×256) → 
ConvTranspose3D(128) → ReLU → ConvTranspose3D(64) → ReLU → 
ConvTranspose3D(32) → ReLU → ConvTranspose3D(1) → 
Crop → Output(5×5×5×1)
```

**Key Features:**
- Progressive channel expansion: 32→64→128→256
- 3×3×3 kernels throughout
- Stride-2 transposed convolutions for upsampling
- Total parameters: ~3.2M
- FLOPs: 57.3M (Encoder) + 24.9M (Decoder)

### 2. MLP SWAE Architecture

The Multi-Layer Perceptron implementation for ultra-low latency:

**Architecture:**
```
Encoder: Input(125) → Linear(512) → ReLU → Linear(256) → ReLU → Linear(16) → Latent(16D)
Decoder: Latent(16D) → Linear(256) → ReLU → Linear(512) → ReLU → Linear(125) → Output(125)
```

**Key Features:**
- Fully connected architecture (no convolutions)
- Total parameters: ~398K (8x fewer than CNN)
- FLOPs: 796K (72x fewer than CNN)
- Supports INT8/Float8 quantization for 16x speedup
- Ideal for latency-sensitive applications

**Training Scripts:** 
- Use `--arch mlp` flag with any training script
- Optimized versions have `_opt` suffix

### 3. All Variables Model

Extended implementation that handles all 24 BSSN variables with a single model:

**Training Pipeline:**
```
All U_* Variables → Per-sample Normalization → Shared SWAE Model (CNN/MLP) → 
Latent Space (16D) → Reconstruction → Denormalization → Output
```

**Supported Variables:** 
- All 24 BSSN formalism variables (U_ALPHA, U_B0-B2, U_BETA0-2, U_CHI, U_GT0-2, U_K, U_SYMAT0-5, U_SYMGT0-5)
- Uses the same architecture but trained on diverse data distribution
- Maintains per-sample normalization for handling different value ranges

**Training Script:** `train_swae_5x5x5_all_variables.py`  
**Inference Script:** `inference_swae_all_variables_validation_5x5x5_opt.py`

## Performance Metrics

### Latest Results (All 24 Variables)

**Overall Performance:**
- **Average MSE:** 0.004907
- **Average PSNR:** 47.81 dB
- **Average MAE:** 0.0330
- **Mean Relative Error:** 0.91%
- **Compression Ratio:** 15.6:1 (5×5×5 → 8 latent dimensions)

**Per-Variable Performance (Best to Worst):**
```
Variable    MSE        PSNR (dB)   MAE        Mean Rel Err   Samples
U_CHI       0.000734   55.88       0.017151   0.34%          358
U_ALPHA     0.000932   55.61       0.018036   0.34%          359
U_BETA0     0.002279   inf         0.026260   0.37%          364
U_SYMGT0    0.003279   50.49       0.031647   0.49%          352
U_SYMAT3    0.003563   49.31       0.033936   0.88%          345
U_SYMAT1    0.003613   48.73       0.034668   4.31%          363
U_SYMAT2    0.003738   46.94       0.034485   0.98%          349
U_B0        0.004242   inf         0.035187   0.42%          314
U_GT1       0.004353   inf         0.030563   0.53%          311
U_BETA2     0.004463   inf         0.027832   0.35%          344
U_SYMAT5    0.004623   48.47       0.035736   2.49%          334
U_BETA1     0.004795   inf         0.031555   0.44%          346
U_B2        0.005043   inf         0.040131   0.44%          329
U_SYMGT3    0.005184   49.89       0.035706   0.54%          338
U_SYMGT5    0.005375   48.55       0.038483   0.54%          334
U_SYMGT4    0.006050   47.09       0.040497   0.57%          343
U_SYMAT0    0.006081   48.74       0.039001   2.82%          334
U_GT0       0.006222   inf         0.030762   0.53%          349
U_SYMGT1    0.006325   48.63       0.038300   0.58%          322
U_GT2       0.006702   inf         0.031708   0.52%          329
U_B1        0.006714   inf         0.038605   0.47%          362
U_K         0.006718   48.35       0.029282   0.38%          348
U_SYMGT2    0.008354   47.90       0.038483   0.54%          362
U_SYMAT4    0.008797   47.49       0.036743   1.72%          373
```

### Compression Specifications
- **Input:** 5×5×5 = 125 float32 values (500 bytes)
- **Compressed:** 8 latent dimensions (32 bytes)
- **Compression Ratio:** 15.6:1

### Latency Analysis (from computation_costs.py)

The following table shows total latency (computation + transfer) for different network interconnects:

| Link | TX Raw [μs] | TX Latent [μs] | RAW Total [μs] | Conv SWAE Total [μs] | MLP SWAE FLOAT8 Total [μs] |
|------|-------------|-----------------|----------------|---------------------|---------------------------|
| NVLink 3 (100 GB/s) | 0.04 | 0.0013 | 0.04 | 4.2196 | 0.0048 |
| NVLink 4 (200 GB/s) | 0.02 | 0.0006 | 0.02 | 4.219 | 0.0041 |
| PCIe 4x16 (32 GB/s) | 0.125 | 0.004 | 0.125 | 4.2224 | 0.0075 |
| IB 200 Gb (25 GB/s) | 0.16 | 0.0051 | 0.16 | 4.2235 | 0.0086 |
| Eth 40 Gb (5 GB/s) | 0.8 | 0.0256 | 0.8 | 4.244 | 0.0291 |
| Eth 10 Gb (1.25 GB/s) | 3.2 | 0.1024 | 3.2 | 4.3208 | 0.1059 |
| Eth 1 Gb (0.125 GB/s) | 32.0 | 1.024 | 32.0 | 5.2424 | 1.0275 |

**Key Insights:**
- MLP SWAE with Float8 quantization achieves 875x speedup over Conv SWAE on NVLink
- For slower networks (Ethernet), compression becomes critical for reducing transfer time
- Conv SWAE compute time dominates for fast interconnects, while MLP SWAE is network-bound

## Usage

### Training

1. **Single Variable (U_chi only):**
```bash
sbatch train_swae_u_chi_5x5x5.sbatch
```

2. **All Variables with Optimized Architecture:**
```bash
sbatch train_swae_5x5x5_all_variables.sbatch
```

3. **Custom Architecture Training:**
```bash
python train_swae_5x5x5_all_variables.py \
    --arch mlp \
    --batch-size 64 \
    --epochs 100 \
    --lr 2e-4
```

### Inference

1. **Single Variable Inference:**
```bash
sbatch inference_swae_u_chi_validation_5x5x5.sbatch
```

2. **All Variables Inference:**
```bash
python inference_swae_all_variables_validation_5x5x5_opt.py \
    --model-path ./save/swae_all_vars_5x5x5_opt_mlp/best_model.pth \
    --output-dir results \
    --use-float8  # For Float8 quantization
```

### Compression and Reconstruction

To compress an entire folder of HDF5 files and save both reconstructed data and embeddings:

```bash
./compress_folder.sh
```

This will:
- Process all HDF5 files in `/u/tawal/BSSN-Extracted-Data/tt_q/`
- Generate comparison plots for each variable
- Save `all_variables_reconstructed.h5` with 7x7x7 reconstructed data
- Save `all_variables_encoded.h5` with 16-dimensional latent embeddings
- Create timestamped output directory with all results

### Quantization Options

For MLP architectures, enable faster inference with:
- `--use-float8`: Float8 quantization (16x theoretical speedup with better accuracy than INT8)
  - Available in inference scripts with the `--use-float8` flag
  - Maintains better numerical precision than INT8 while achieving similar speedups
  - Particularly effective for the MLP architecture
- `--enable-int8`: INT8 quantization (16x theoretical speedup)
  - Legacy option for older scripts

Example usage:
```bash
python inference_swae_all_variables_validation_5x5x5_opt.py \
    --model-path ./save/swae_all_vars_5x5x5_opt_mlp/best_model.pth \
    --use-float8 \
    --arch mlp
```

## Architecture Search Experiment

An architecture search experiment is available in:
```
/u/tawal/0722-NN-based-compression-AE/architecture_search_experiment/
```

This directory contains code for systematically exploring different network architectures to find optimal configurations for compression quality vs. speed trade-offs.

## Data Requirements

The models expect HDF5 files containing simulation data with the following structure:
- 3D volumetric data from BSSN simulations
- Variables ending with `U_chi` pattern
- Data is automatically partitioned into 5×5×5 subvolumes during training

## Normalization Methods

Supports multiple normalization strategies:
- `pos_log`: Positive logarithmic transformation (default, handles positive values)
- `minmax`: Min-max normalization to [0, 1]
- `zscore`: Z-score normalization
- `none`: No normalization

## Model Outputs

- **Checkpoint files:** Saved in `./save/` directory
- **Inference results:** 
  - VTI files for visualization in ParaView
  - Comparison plots (normalized and denormalized scales)
  - Comprehensive metrics including PSNR, MSE, and relative errors
- **Speed benchmarks:** Compression/decompression throughput in GB/s

## Key Features

1. **Multi-scale evaluation:** Results shown in both normalized (model scale) and original physical units
2. **Deterministic data splits:** Fixed 80/15/5 train/val/test split with seed=42
3. **Early stopping:** Prevents overfitting with configurable patience
4. **Comprehensive metrics:** PSNR, MSE, correlation, and relative error analysis
5. **Production-ready optimizations:** TF32, mixed precision, and model compilation support

## Directory Structure

```
.
├── README.md                                    # Project documentation
├── compress_folder.sh                           # Script to compress all HDF5 files in a folder
├── compression_and_reconstruction.py            # Main compression/reconstruction script
├── compression_and_reconstruction.sbatch        # SLURM job for compression/reconstruction
├── computation_costs.py                         # Analyzes latency for different network interconnects
├── utils.py                                     # Utility functions
│
├── models/                                      # Model architectures
│   ├── __init__.py
│   ├── models.py                               # Model factory and utilities
│   ├── swae_pure_3d_5x5x5.py                  # CNN SWAE base implementation
│   ├── swae_pure_3d_5x5x5_opt.py              # Optimized CNN and MLP SWAE
│   └── swae_mlp_3d_5x5x5.py                   # MLP-specific implementations
│
├── datasets/                                    # Dataset loaders
│   ├── __init__.py
│   ├── datasets.py                             # Base dataset classes
│   ├── all_variables_dataset_5x5x5_opt.py      # All 24 variables dataset
│   ├── u_chi_dataset.py                        # Original 7x7x7 single variable
│   └── u_chi_dataset_5x5x5_opt.py             # 5x5x5 single variable dataset
│
├── train_swae_5x5x5_all_variables.py           # Main training script
├── train_swae_5x5x5_all_variables.sbatch       # SLURM job for training
│
├── inference_swae_all_variables_validation_5x5x5_opt.py    # Validation inference
├── inference_swae_all_variables_validation_5x5x5_opt.sbatch # SLURM job for validation
│
├── save/                                        # Model checkpoints
│   └── swae_all_vars_5x5x5_opt_conv/          # Pre-trained models
│       └── best_model.pth
│
├── compression_reconstruction_results_*/        # Results directories
│   ├── all_variables_reconstructed.h5         # Reconstructed data (7x7x7 padded)
│   ├── all_variables_encoded.h5               # Compressed latent codes
│   ├── average_metrics.txt                    # Performance metrics
│   └── U_*/                                   # Per-variable visualizations
│       ├── sample_*_comparison_slices_*.png   # Comparison plots
│       └── vti_files/                         # VTK files for ParaView
│
└── SWAE-3D-Architecture/                        # Architecture visualization
    └── swae_architecture_diagram.py            # Generates architecture diagrams
```

## Key Scripts

### Training
- `train_swae_5x5x5_all_variables.py`: Main training script supporting both CNN and MLP architectures
- `train_swae_5x5x5_all_variables.sbatch`: SLURM submission script

### Inference and Compression
- `compression_and_reconstruction.py`: Process folders of HDF5 files, generate plots and save compressed/reconstructed data
- `compress_folder.sh`: Convenience script to compress all HDF5 files in a folder and save both:
  - `all_variables_reconstructed.h5`: Reconstructed 7x7x7 data in BSSN format
  - `all_variables_encoded.h5`: Compressed latent embeddings (16D vectors)
- `inference_swae_all_variables_validation_5x5x5_opt.py`: Validation with comprehensive metrics

### Analysis
- `computation_costs.py`: Analyze compression performance across different network interconnects
- `exact_flops_analysis/calculate_swae_flops.py`: Detailed FLOP analysis for different architectures

## Citation

If you use this code in your research, please cite the relevant papers on Sliced Wasserstein Autoencoders and the specific application to gravitational wave data compression.