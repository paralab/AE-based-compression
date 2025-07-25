#!/usr/bin/env python
"""
Analyze why poslog transformation doesn't work well for specific variables
and propose alternative transformations.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import os
from tqdm import tqdm

def poslog_transform(data, epsilon=1e-8):
    """Apply positive-shift log transformation (poslog)."""
    data_min = data.min()
    if data_min <= 0:
        data_shifted = data - data_min + epsilon
    else:
        data_shifted = data.copy()
    return np.log(data_shifted)

def symmetric_log_transform(data):
    """Apply symmetric log transformation (handles positive and negative values)."""
    return np.sign(data) * np.log1p(np.abs(data))

def asinh_transform(data):
    """Apply inverse hyperbolic sine transformation."""
    return np.arcsinh(data)

def power_transform(data, lambda_param=0):
    """Apply Box-Cox-like power transformation."""
    if lambda_param == 0:
        return np.sign(data) * np.log1p(np.abs(data))
    else:
        return np.sign(data) * (np.power(np.abs(data) + 1, lambda_param) - 1) / lambda_param

def analyze_transformation_issues():
    """Analyze why poslog doesn't work well and test alternatives."""
    
    # Simulated data with properties similar to the problematic variables
    variables_properties = {
        'U_B2': {
            'mean': 0.00000200,
            'std': 0.00325078,
            'min': -0.10413070,
            'max': 0.02945773,
            'skewness': -9.286,
            'zero_pct': 2.66,
            'neg_pct': 34.43
        },
        'U_SYMAT2': {
            'mean': 0.0,
            'std': 0.06360482,
            'min': -1.32084740,
            'max': 1.32084740,
            'skewness': 0.0,
            'zero_pct': 0.0,
            'neg_pct': 50.0
        },
        'U_SYMGT2': {
            'mean': 0.0,
            'std': 0.00910068,
            'min': -0.08594614,
            'max': 0.08594614,
            'skewness': 0.0,
            'zero_pct': 2.66,
            'neg_pct': 48.67
        }
    }
    
    print("ANALYSIS OF POSLOG TRANSFORMATION ISSUES")
    print("=" * 80)
    
    for var_name, props in variables_properties.items():
        print(f"\n{var_name}:")
        print("-" * len(var_name))
        
        # Issues with poslog for this variable
        print("\nIssues with poslog transformation:")
        
        # Issue 1: Negative values
        if props['neg_pct'] > 0:
            print(f"  1. {props['neg_pct']:.1f}% negative values require shifting")
            print(f"     - Min value: {props['min']:.8f}")
            print(f"     - Shift required: {-props['min']:.8f}")
            
        # Issue 2: Near-zero values
        if abs(props['mean']) < 0.001:
            print(f"  2. Mean very close to zero ({props['mean']:.8f})")
            print("     - Log transformation amplifies noise near zero")
            
        # Issue 3: Symmetry
        if abs(props['skewness']) < 0.1 and props['neg_pct'] > 40:
            print(f"  3. Symmetric distribution ({props['neg_pct']:.1f}% negative)")
            print("     - Poslog breaks natural symmetry")
            
        # Issue 4: Scale
        if props['std'] < 0.01:
            print(f"  4. Very small scale (std: {props['std']:.8f})")
            print("     - Log transformation may over-compress")
            
        # Issue 5: Range
        range_val = props['max'] - props['min']
        if props['min'] < 0 and props['max'] > 0:
            print(f"  5. Data spans zero (range: {range_val:.8f})")
            print("     - Poslog shift changes relative magnitudes")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED ALTERNATIVES:")
    print("=" * 80)
    
    print("\n1. Symmetric Log Transformation (symlog):")
    print("   - Preserves symmetry around zero")
    print("   - Formula: sign(x) * log(1 + |x|)")
    print("   - Good for: U_SYMAT2, U_SYMGT2, U_SYMAT4")
    
    print("\n2. Inverse Hyperbolic Sine (asinh):")
    print("   - Handles positive and negative values naturally")
    print("   - Formula: asinh(x) = log(x + sqrt(x² + 1))")
    print("   - Good for: All variables, especially symmetric ones")
    
    print("\n3. Standardization Only:")
    print("   - Simple z-score normalization")
    print("   - Formula: (x - mean) / std")
    print("   - Good for: Variables with small ranges")
    
    print("\n4. Robust Scaling:")
    print("   - Uses median and IQR instead of mean/std")
    print("   - Formula: (x - median) / IQR")
    print("   - Good for: Variables with outliers (U_B2)")

def create_comparison_plots(data_folder, target_variables, output_dir="transformation_comparison"):
    """Create comparison plots for different transformations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load first file to get sample data
    file_pattern = os.path.join(data_folder, "bssn_gr_*_extracted.hdf5")
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        print(f"No files found in {data_folder}")
        return
    
    # Load sample data
    with h5py.File(files[0], 'r') as f:
        vars_list = f['vars'][:].astype(str)
        var_data = f['var_data'][:]
    
    for var_name in target_variables:
        if var_name in vars_list:
            var_idx = list(vars_list).index(var_name)
            data = var_data[var_idx][:, 1:6, 1:6, 1:6].flatten()
            
            # Create comparison plot
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'{var_name} - Transformation Comparison', fontsize=16)
            
            # Original
            ax = axes[0, 0]
            ax.hist(data, bins=100, alpha=0.7, density=True)
            ax.set_title('Original')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            
            # Poslog
            ax = axes[0, 1]
            try:
                data_poslog = poslog_transform(data)
                ax.hist(data_poslog, bins=100, alpha=0.7, density=True, color='orange')
                ax.set_title('Poslog Transform')
            except:
                ax.text(0.5, 0.5, 'Failed', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Poslog Transform (Failed)')
            
            # Symmetric log
            ax = axes[0, 2]
            data_symlog = symmetric_log_transform(data)
            ax.hist(data_symlog, bins=100, alpha=0.7, density=True, color='green')
            ax.set_title('Symmetric Log')
            
            # Asinh
            ax = axes[1, 0]
            data_asinh = asinh_transform(data)
            ax.hist(data_asinh, bins=100, alpha=0.7, density=True, color='red')
            ax.set_title('Asinh Transform')
            
            # Standardization
            ax = axes[1, 1]
            std_val = np.std(data)
            if std_val > 0:
                data_std = (data - np.mean(data)) / std_val
                ax.hist(data_std, bins=100, alpha=0.7, density=True, color='purple')
            else:
                ax.text(0.5, 0.5, 'Std = 0', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Standardization')
            
            # Robust scaling
            ax = axes[1, 2]
            q25, q75 = np.percentile(data, [25, 75])
            iqr = q75 - q25
            median = np.median(data)
            if iqr > 0:
                data_robust = (data - median) / iqr
            else:
                data_robust = data - median
            ax.hist(data_robust, bins=100, alpha=0.7, density=True, color='brown')
            ax.set_title('Robust Scaling')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{var_name}_transformations.png'), dpi=150)
            plt.close()
            
            print(f"Created transformation comparison plot for {var_name}")

def main():
    # First, analyze the issues
    analyze_transformation_issues()
    
    # Then create comparison plots
    data_folder = "/u/tawal/BSSN-Extracted-Data/tt_q01/"
    target_variables = ['U_B2', 'U_SYMAT2', 'U_GT2', 'U_SYMGT2', 'U_SYMAT4', 'U_SYMAT3']
    
    if os.path.exists(data_folder):
        print("\n\nCreating transformation comparison plots...")
        create_comparison_plots(data_folder, target_variables)
        print("\nAnalysis complete! Check 'transformation_comparison' folder for plots.")
    else:
        print(f"\nData folder not found: {data_folder}")
        print("Skipping plot generation.")

if __name__ == "__main__":
    main()