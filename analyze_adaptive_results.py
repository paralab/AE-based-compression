#!/usr/bin/env python3
"""
Analyze the adaptive scaling reconstruction results
"""

import h5py
import numpy as np
import json
import matplotlib.pyplot as plt
import os

def analyze_reconstruction():
    """Analyze reconstruction quality and identify issues"""
    
    # Load original and reconstructed files
    original_file = "/u/tawal/BSSN-Extracted-Data/tt_q08/bssn_gr_11200_extracted.hdf5"
    reconstructed_file = "./reconstructed/bssn_gr_11200_adaptive_reconstructed.hdf5"
    
    print("Loading data...")
    with h5py.File(original_file, 'r') as f_orig, h5py.File(reconstructed_file, 'r') as f_recon:
        var_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                     for name in f_orig['vars'][:]]
        orig_data = f_orig['var_data'][:]
        recon_data = f_recon['var_data'][:]
    
    # Focus on problematic variables
    problematic_vars = ['U_B2', 'U_SYMAT2', 'U_GT2', 'U_SYMGT2', 'U_SYMAT4', 'U_SYMAT3']
    
    # Create output directory
    os.makedirs("adaptive_analysis", exist_ok=True)
    
    # Analyze each problematic variable
    results = []
    for var_name in problematic_vars:
        if var_name not in var_names:
            continue
            
        var_idx = var_names.index(var_name)
        orig = orig_data[var_idx, :, 1:6, 1:6, 1:6]  # Extract 5x5x5 center
        recon = recon_data[var_idx, :, 1:6, 1:6, 1:6]
        
        # Find constant/near-constant samples
        constant_samples = []
        for i in range(orig.shape[0]):
            sample = orig[i]
            sample_range = sample.max() - sample.min()
            sample_std = sample.std()
            
            if sample_range < 1e-12 or sample_std < 1e-12:
                constant_samples.append(i)
        
        # Calculate metrics
        mse = np.mean((orig - recon) ** 2)
        mae = np.mean(np.abs(orig - recon))
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # Analyze constant samples separately
        if constant_samples:
            const_orig = orig[constant_samples]
            const_recon = recon[constant_samples]
            const_mse = np.mean((const_orig - const_recon) ** 2)
            const_mae = np.mean(np.abs(const_orig - const_recon))
            const_psnr = 20 * np.log10(1.0 / np.sqrt(const_mse)) if const_mse > 0 else float('inf')
        else:
            const_mse = const_mae = const_psnr = None
        
        # Store results
        result = {
            'variable': var_name,
            'n_samples': orig.shape[0],
            'n_constant': len(constant_samples),
            'overall_mse': mse,
            'overall_mae': mae,
            'overall_psnr': psnr,
            'constant_mse': const_mse,
            'constant_mae': const_mae,
            'constant_psnr': const_psnr,
            'orig_min': orig.min(),
            'orig_max': orig.max(),
            'orig_mean': orig.mean(),
            'orig_std': orig.std()
        }
        results.append(result)
        
        print(f"\n{var_name}:")
        print(f"  Total samples: {result['n_samples']}")
        print(f"  Constant samples: {result['n_constant']} ({100*result['n_constant']/result['n_samples']:.1f}%)")
        print(f"  Overall PSNR: {result['overall_psnr']:.2f} dB")
        if const_psnr is not None:
            print(f"  Constant samples PSNR: {const_psnr:.2f} dB")
        print(f"  Original range: [{orig.min():.2e}, {orig.max():.2e}]")
        
        # Plot example constant field if any
        if constant_samples and len(constant_samples) > 0:
            idx = constant_samples[0]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original
            im0 = axes[0].imshow(orig[idx, 2, :, :], cmap='viridis')
            axes[0].set_title(f'Original (sample {idx})')
            plt.colorbar(im0, ax=axes[0])
            
            # Reconstructed
            im1 = axes[1].imshow(recon[idx, 2, :, :], cmap='viridis')
            axes[1].set_title('Reconstructed')
            plt.colorbar(im1, ax=axes[1])
            
            # Error
            error = orig[idx, 2, :, :] - recon[idx, 2, :, :]
            im2 = axes[2].imshow(error, cmap='RdBu_r')
            axes[2].set_title(f'Error (MSE={np.mean(error**2):.2e})')
            plt.colorbar(im2, ax=axes[2])
            
            plt.suptitle(f'{var_name} - Constant Field Example')
            plt.tight_layout()
            plt.savefig(f'adaptive_analysis/{var_name}_constant_example.png', dpi=150)
            plt.close()
    
    # Save results
    with open('adaptive_analysis/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nAnalysis complete. Results saved to adaptive_analysis/")


if __name__ == "__main__":
    analyze_reconstruction()