# 3D SWAE+Thera Implementation for Super-Resolution

This document describes the implementation of a 3D super-resolution system using Sliced-Wasserstein Autoencoder (SWAE) and Thera neural heat fields, based on the paper "Exploring Autoencoder-based Error-bounded Compression for Scientific Data".

## Overview

The implementation combines:
1. **SWAE (Sliced-Wasserstein Autoencoder)** as the encoder for feature extraction
2. **Thera Neural Heat Fields** as the decoder for implicit neural representation
3. **LIIF (Local Implicit Image Function)** framework extended to 3D
4. **3D Mathematical Functions** as training data: `sin(2πk₁x)sin(2πk₂y)sin(2πk₃z)`

## Architecture Components

### 1. 3D Mathematical Function Dataset (`datasets/math_function_3d.py`)
- Generates 3D functions of the form `sin(2πk₁x)sin(2πk₂y)sin(2πk₃z)`
- Supports diagonal combinations (k₁=k₂=k₃) for simplified training
- Resolution: 40×40×40 for base functions
- k values: {2, 3, 4, 5, 6}

### 2. SWAE 3D Encoder (`models/swae.py`)
- **Architecture**: 3D Convolutional layers with batch normalization
- **Input**: 40×40×40 volume → **Output**: 5×5×5×128 features
- **Loss Function**: Reconstruction loss + Sliced-Wasserstein distance
- **Key Features**:
  - Deterministic encoding/decoding (unlike VAE)
  - Efficient O(n log n) computation of sliced-Wasserstein distance
  - Suitable for scientific data compression

### 3. Thera 3D Decoder (`models/thera_3d.py`)
- **Thermal Activation**: ξ(z, ν, κ, t) = sin(z) · exp(-|ν|²κt)
- **Input**: Features + 3D coordinates + cell info + scale parameter
- **Output**: Scalar values for 3D function reconstruction
- **Parameters**:
  - Hidden dimension: 256
  - Number of frequencies: 64
  - Thermal diffusivity κ: 1.0

### 4. LIIF 3D Framework (`models/liif_3d.py`)
- **Extension of LIIF to 3D**: Handles 3D coordinates and volumes
- **Local Ensemble**: 8-point interpolation (2³ corners of 3D cube)
- **Features**: 
  - 3D coordinate encoding
  - 3D cell size information
  - Scale-aware processing

## Training Configuration

### Data Pipeline
- **Input Resolution**: 20×20×20 (downsampled)
- **Target Resolution**: 40×40×40 (2× super-resolution)
- **Sample Points**: 4000 coordinate-value pairs per volume
- **Batch Size**: 4 (due to 3D memory requirements)

### Model Configuration
```yaml
model:
  name: liif-3d
  args:
    encoder_spec:
      name: swae-encoder-3d
      args:
        input_channels: 1
        hidden_channels: [32, 64, 128]
    imnet_spec:
      name: thera-3d-simple
      args:
        out_dim: 1
        hidden_dim: 256
        num_frequencies: 64
        kappa: 1.0
```

### Training Parameters
- **Optimizer**: Adam (lr=1e-4)
- **Epochs**: 500
- **Learning Rate Schedule**: MultiStepLR with γ=0.5 at [100, 200, 300, 400]
- **Loss**: L1 Loss between predicted and ground truth values

## Key Advantages of SWAE over Other Autoencoders

Based on the paper findings:

1. **Deterministic**: Unlike VAEs, SWAE produces consistent latent codes
2. **Better Reconstruction**: Shows less reconstruction loss on scientific data
3. **Efficient Training**: O(n log n) vs O(n²) for Wasserstein Autoencoder
4. **Scientific Data Suitability**: Particularly effective for structured scientific datasets

## Files Created/Modified

### New Files
- `models/swae.py` - SWAE implementation
- `models/thera_3d.py` - 3D Thera neural heat fields
- `models/liif_3d.py` - 3D LIIF framework
- `datasets/math_function_3d.py` - 3D mathematical function dataset
- `train_liif_3d.py` - 3D training script
- `test_3d_implementation.py` - Comprehensive test suite
- `configs/train-3d/train_swae_thera_3d.yaml` - Training configuration
- `train_swae_thera_3d.sbatch` - SLURM training script

### Modified Files
- `datasets/wrappers.py` - Added 3D downsampling wrapper
- `datasets/__init__.py` - Added 3D dataset imports
- `models/__init__.py` - Added 3D model imports

## Usage

### 1. Test Implementation
```bash
python test_3d_implementation.py
```

### 2. Start Training
```bash
sbatch train_swae_thera_3d.sbatch
```

### 3. Monitor Training
```bash
# Check logs
tail -f logs/swae_thera_3d_*.out

# View tensorboard (if available)
tensorboard --logdir=save/swae_thera_3d_training
```

## Expected Results

The system should learn to:
1. **Encode** 3D mathematical functions into compact latent representations
2. **Super-resolve** from 20×20×20 to 40×40×40 resolution
3. **Preserve** frequency characteristics of the original functions
4. **Generalize** to unseen combinations of k values

## Future Enhancements

1. **3D Feature Unfolding**: Implement proper 3D neighborhood feature extraction
2. **Higher Scale Factors**: Extend to 4× or 8× super-resolution
3. **More Complex Functions**: Add non-diagonal k combinations
4. **Real Scientific Data**: Apply to actual 3D scientific datasets
5. **Compression Analysis**: Evaluate compression ratios and reconstruction quality

## Mathematical Foundation

### SWAE Loss Function
```
L(φ, ψ) = (1/M) Σ c(xₘ, ψ(φ(xₘ))) + (λ/LM) ΣΣ c(θₗ·z̃ᵢ[m], θₗ·φ(xⱼ[m]))
```

Where:
- φ: encoder, ψ: decoder
- c(x,y) = ||x-y||₂: L2 distance
- θₗ: random projections on unit sphere
- λ: regularization weight

### Thera Thermal Activation
```
ξ(z, ν, κ, t) = sin(z) · exp(-|ν|²κt)
```

Where:
- z: frequency-transformed input
- ν: frequency magnitudes
- κ: thermal diffusivity
- t: scale parameter (S²)

This implementation provides a solid foundation for 3D super-resolution using the SWAE+Thera combination, specifically designed for scientific data applications. 