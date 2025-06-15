# Mathematical Function Implementation for LIIF

## Overview
This implementation adapts the LIIF (Learning Continuous Image Representation with Local Implicit Image Function) model to train on mathematical functions of the form `sin(2πk1*x)*sin(2πk2*y)` where k1 and k2 are chosen from {2, 3, 4, 5, 6, 7, 8, 9}.

## Key Modifications

### 1. New Dataset: `MathFunctionDataset`
**File**: `datasets/math_function.py`
- Generates mathematical functions `sin(2πk1*x)*sin(2πk2*y)` on a 2D grid
- Supports all combinations of k1, k2 from the specified range
- Configurable resolution (default: 256x256)
- Memory caching for efficiency
- Single-channel output (1, H, W) instead of RGB (3, H, W)

### 2. New Wrapper: `MathFunctionDownsampled`
**File**: `datasets/wrappers.py`
- Adapts the super-resolution training paradigm for mathematical functions
- Creates low-resolution versions through bilinear downsampling
- Handles single-channel data instead of RGB
- Maintains the same coordinate sampling strategy as original LIIF
- Output format: `{'inp': LR_function, 'coord': coordinates, 'cell': cell_size, 'gt': function_values}`

### 3. Modified EDSR Model
**File**: `models/edsr.py`
- Added `n_colors` parameter to support single-channel input
- Default remains 3 for RGB compatibility
- Enables training on grayscale/single-channel data

### 4. New Configuration
**File**: `configs/train-div2k/train_edsr-baseline-liif_math.yaml`
- Dataset: `math-function` with k_values [2,3,4,5,6,7,8,9]
- Wrapper: `math-function-downsampled`
- Model: LIIF with single-channel EDSR encoder (`n_colors: 1`)
- Output dimension: 1 (scalar function values instead of RGB)
- Data normalization: No normalization (functions already in [-1,1])

### 5. Training Script
**File**: `train_liif_math.sbatch`
- SLURM job script for mathematical function training
- Integrated data generation (no separate data preparation needed)
- Uses same training infrastructure as original LIIF

## Technical Details

### Function Generation
```python
Z = sin(2π * k1 * X) * sin(2π * k2 * Y)
```
- X, Y: coordinate grids from -1 to 1
- k1, k2: frequency parameters from {2,3,4,5,6,7,8,9}
- Output range: [-1, 1]

### Architecture Changes
- **Input channels**: 3 → 1 (RGB → single channel)
- **Output dimension**: 3 → 1 (RGB → scalar value)
- **Encoder**: EDSR-baseline with single-channel input
- **Decoder**: MLP with 1D output

### Training Configuration
- **Resolution**: 256x256 for high-resolution functions
- **Input size**: 48x48 patches
- **Scale factor**: Up to 4x upsampling
- **Sample points**: 2304 coordinate-value pairs per patch
- **Batch size**: 16
- **Learning rate**: 1e-4
- **Epochs**: 1000

## Expected Performance

### Why This Should Work Well
1. **Function Smoothness**: Mathematical sine functions are smooth and continuous
2. **Perfect Ground Truth**: No noise or artifacts unlike natural images
3. **Regular Patterns**: Sine functions have predictable, learnable patterns
4. **Coordinate-based Learning**: LIIF's coordinate-based approach is ideal for continuous functions
5. **Scale Invariance**: Can learn arbitrary upsampling factors

### Advantages Over Image Super-Resolution
- **Cleaner Training Signal**: No image artifacts or noise
- **Infinite Resolution**: Mathematical functions have no inherent resolution limit
- **Predictable Patterns**: More regular than natural image textures
- **Exact Evaluation**: Can compute exact function values for any coordinate

## Usage

### Training
```bash
sbatch train_liif_math.sbatch
```

### Monitoring
```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/train_liif_math_*.out
```

### Results
Training results will be saved in `./save/math_function_training/`

## File Structure
```
liif_0613/
├── datasets/
│   ├── math_function.py          # New mathematical function dataset
│   └── wrappers.py               # Modified with math function wrapper
├── models/
│   └── edsr.py                   # Modified to support single-channel input
├── configs/train-div2k/
│   └── train_edsr-baseline-liif_math.yaml  # New config for math functions
├── train_liif_math.sbatch        # New training script
└── MATH_FUNCTION_IMPLEMENTATION.md  # This documentation
```

## Next Steps
1. Monitor training progress
2. Evaluate on test functions with different k1, k2 values
3. Test arbitrary scale factors (2x, 3x, 5x, etc.)
4. Visualize learned function reconstructions
5. Compare with analytical solutions for validation 