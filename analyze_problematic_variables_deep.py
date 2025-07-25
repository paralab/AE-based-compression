#!/usr/bin/env python3
"""
Deep analysis of problematic variables to understand why they fail
"""

import numpy as np
import h5py
import glob
import os
from collections import defaultdict

def analyze_variable_characteristics(data_folder, target_vars):
    """Analyze detailed characteristics of problematic variables"""
    
    hdf5_files = glob.glob(os.path.join(data_folder, "*.hdf5"))
    hdf5_files.sort()
    
    var_data = defaultdict(list)
    
    # Load data for analysis
    for file_path in hdf5_files[:5]:  # Just first 5 files for quick analysis
        print(f"Loading {file_path}")
        with h5py.File(file_path, 'r') as f:
            var_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                        for name in f['vars'][:]]
            data = f['var_data'][:]
            
            for var in target_vars:
                if var in var_names:
                    idx = var_names.index(var)
                    var_samples = data[idx][:100]  # First 100 samples
                    var_data[var].extend(var_samples.flatten())
    
    # Analyze each variable
    results = {}
    for var, values in var_data.items():
        values = np.array(values)
        
        # Remove NaN/inf
        finite_mask = np.isfinite(values)
        values_clean = values[finite_mask]
        
        if len(values_clean) == 0:
            continue
            
        # Detailed analysis
        analysis = {
            'count': len(values_clean),
            'mean': np.mean(values_clean),
            'std': np.std(values_clean),
            'min': np.min(values_clean),
            'max': np.max(values_clean),
            'median': np.median(values_clean),
            'q1': np.percentile(values_clean, 25),
            'q3': np.percentile(values_clean, 75),
            'iqr': np.percentile(values_clean, 75) - np.percentile(values_clean, 25),
            'zeros': np.sum(values_clean == 0),
            'near_zeros': np.sum(np.abs(values_clean) < 1e-10),
            'very_small': np.sum(np.abs(values_clean) < 1e-6),
            'negative_ratio': np.sum(values_clean < 0) / len(values_clean),
            'positive_ratio': np.sum(values_clean > 0) / len(values_clean),
            'typical_magnitude': np.median(np.abs(values_clean[values_clean != 0])) if np.any(values_clean != 0) else 0,
            'scale_ratio': np.max(np.abs(values_clean)) / (np.median(np.abs(values_clean[values_clean != 0])) + 1e-12) if np.any(values_clean != 0) else np.inf
        }
        
        results[var] = analysis
    
    return results

def suggest_transformation(analysis):
    """Suggest best transformation based on analysis"""
    
    suggestions = []
    
    # Check for zero-centered data
    if abs(analysis['mean']) < 0.1 * analysis['std']:
        suggestions.append("Data is zero-centered")
    
    # Check for many near-zero values
    if analysis['near_zeros'] > 0.5 * analysis['count']:
        suggestions.append("Many near-zero values - consider shifted transformation")
    
    # Check scale ratio
    if analysis['scale_ratio'] > 1000:
        suggestions.append("Extreme scale differences - consider adaptive scaling")
    
    # Suggest transformation
    if analysis['near_zeros'] > 0.3 * analysis['count']:
        transform = "Consider: sinh-arcsinh transformation or shifted log"
    elif analysis['negative_ratio'] > 0.3:
        transform = "Consider: asinh with larger scale or sign-preserving log"
    else:
        transform = "Consider: standard log or Box-Cox transformation"
    
    return suggestions, transform

def main():
    data_folder = "/u/tawal/BSSN-Extracted-Data/tt_q01/"
    problematic_vars = ['U_B2', 'U_SYMAT2', 'U_GT2', 'U_SYMGT2', 'U_SYMAT4', 'U_SYMAT3']
    
    print("="*80)
    print("DEEP ANALYSIS OF PROBLEMATIC VARIABLES")
    print("="*80)
    
    results = analyze_variable_characteristics(data_folder, problematic_vars)
    
    for var, analysis in results.items():
        print(f"\n{var}:")
        print(f"  Range: [{analysis['min']:.2e}, {analysis['max']:.2e}]")
        print(f"  Mean ± Std: {analysis['mean']:.2e} ± {analysis['std']:.2e}")
        print(f"  Median (Q1, Q3): {analysis['median']:.2e} ({analysis['q1']:.2e}, {analysis['q3']:.2e})")
        print(f"  Typical magnitude: {analysis['typical_magnitude']:.2e}")
        print(f"  Scale ratio (max/typical): {analysis['scale_ratio']:.1f}")
        print(f"  Zero values: {analysis['zeros']} ({100*analysis['zeros']/analysis['count']:.1f}%)")
        print(f"  Near-zero (< 1e-10): {analysis['near_zeros']} ({100*analysis['near_zeros']/analysis['count']:.1f}%)")
        print(f"  Very small (< 1e-6): {analysis['very_small']} ({100*analysis['very_small']/analysis['count']:.1f}%)")
        print(f"  Negative/Positive ratio: {analysis['negative_ratio']:.1%} / {analysis['positive_ratio']:.1%}")
        
        suggestions, transform = suggest_transformation(analysis)
        print(f"  Issues: {'; '.join(suggestions)}")
        print(f"  {transform}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("1. These variables have many values extremely close to zero")
    print("2. Relative error explodes when dividing by near-zero values")
    print("3. Standard transformations (log, asinh) may not be sufficient")
    print("4. Need transformation that handles near-zero values gracefully")
    print("\nRECOMMENDATIONS:")
    print("1. Use MSE loss (not relative error) for training")
    print("2. Apply stronger pre-scaling before transformation")
    print("3. Consider sinh-arcsinh or Yeo-Johnson transformations")
    print("4. Evaluate using absolute error metrics for near-zero values")

if __name__ == "__main__":
    main()