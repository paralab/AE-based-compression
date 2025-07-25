# Dual Model Training Implementation Summary

## Overview
Successfully implemented a dual-model training system for SWAE that trains two separate models simultaneously:
1. **Model 1 (poslog)**: For standard variables using poslog transformation
2. **Model 2 (asinh)**: For problematic variables using asinh transformation

## Problematic Variables Identified
Based on analysis, these variables perform poorly with poslog transformation:
- U_B2 (asinh with scale 0.001)
- U_SYMAT2 (asinh with scale 0.01)
- U_GT2 (asinh with scale 0.001)
- U_SYMGT2 (asinh with scale 0.01)
- U_SYMAT4 (asinh with scale 0.01)
- U_SYMAT3 (asinh with scale 0.01)

## Files Modified
1. **train_swae_5x5x5_problematic_variables.py**
   - Restructured to train two models simultaneously
   - Each model has its own optimizer and learning rate scheduler
   - Training functions handle both models in each epoch
   - Models saved in separate directories: `./save/swae_dual_transform/poslog/` and `./save/swae_dual_transform/asinh/`

2. **train_swae_5x5x5_problematic_variables.sbatch**
   - Updated job name to `swae_dual_transform`
   - Increased memory to 48G for dual models
   - Updated documentation to reflect dual-model approach

3. **datasets/problematic_variables_dataset_5x5x5.py**
   - Updated to use asinh transformation for all problematic variables
   - Based on transformation analysis results

## Running the Training
To start the dual-model training:
```bash
sbatch train_swae_5x5x5_problematic_variables.sbatch
```

## Expected Outputs
- Two trained models, each optimized for its specific variable group
- Separate validation metrics for each model
- Overall PSNR combining both models' performance
- Checkpoints and best models saved for both architectures

## Key Benefits
1. **Optimal Transformations**: Each variable group uses the transformation that works best for its characteristics
2. **Improved Performance**: Problematic variables should show significant PSNR improvements with asinh transformation
3. **Modular Design**: Easy to extend with additional variable groups or transformations
4. **Simultaneous Training**: Both models train in parallel, maintaining efficiency