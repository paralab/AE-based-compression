#!/usr/bin/env python
"""
Implement recommended transformations for specific variables and generate comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import os
from tqdm import tqdm
from scipy import stats

# Transformation functions
def poslog_transform(data, epsilon=1e-8):
    """Standard poslog transformation (for comparison)."""
    data_min = data.min()
    if data_min <= 0:
        data_shifted = data - data_min + epsilon
    else:
        data_shifted = data.copy()
    return np.log(data_shifted), data_min

def symmetric_log_transform(data, c=1.0):
    """Symmetric log transformation that preserves sign."""
    return np.sign(data) * np.log1p(np.abs(data) / c) * c

def asinh_transform(data, scale=1.0):
    """Inverse hyperbolic sine transformation."""
    return np.arcsinh(data / scale)

def standardize(data):
    """Standard z-score normalization."""
    return (data - np.mean(data)) / np.std(data)

def robust_scale(data):
    """Robust scaling using median and IQR."""
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    median = np.median(data)
    if iqr > 0:
        return (data - median) / iqr
    else:
        return data - median

# Reverse transformation functions
def reverse_symmetric_log(transformed, c=1.0):
    """Reverse symmetric log transformation."""
    return np.sign(transformed) * (np.expm1(np.abs(transformed) / c) * c)

def reverse_asinh(transformed, scale=1.0):
    """Reverse asinh transformation."""
    return np.sinh(transformed) * scale

def reverse_poslog(transformed, data_min, epsilon=1e-8):
    """Reverse poslog transformation."""
    return np.exp(transformed) - epsilon + data_min

def load_variable_data(data_folder, var_name, max_files=5):
    """Load data for a specific variable from first few files."""
    file_pattern = os.path.join(data_folder, "bssn_gr_*_extracted.hdf5")
    files = sorted(glob.glob(file_pattern))[:max_files]
    
    all_data = []
    for file_path in files:
        with h5py.File(file_path, 'r') as f:
            vars_list = f['vars'][:].astype(str)
            if var_name in vars_list:
                var_idx = list(vars_list).index(var_name)
                var_data = f['var_data'][:]
                # Extract 5x5x5 center crop
                data_5x5x5 = var_data[var_idx][:, 1:6, 1:6, 1:6]
                all_data.append(data_5x5x5.flatten())
    
    return np.concatenate(all_data) if all_data else np.array([])

def plot_transformation_comparison(var_name, data, transformations, output_dir):
    """Create comparison plots for different transformations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{var_name} - Transformation Comparison', fontsize=16)
    
    # Original distribution
    ax = axes[0, 0]
    counts, bins, _ = ax.hist(data, bins=100, alpha=0.7, color='blue', density=True)
    ax.set_title('Original Distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # Add statistics
    stats_text = f'Mean: {np.mean(data):.4f}\nStd: {np.std(data):.4f}\nSkew: {stats.skew(data):.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot each transformation
    plot_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    
    for idx, (name, transformed_data, color) in enumerate(transformations[:5]):
        row, col = plot_positions[idx]
        ax = axes[row, col]
        
        # Remove NaN and inf values for plotting
        valid_data = transformed_data[np.isfinite(transformed_data)]
        
        if len(valid_data) > 0:
            ax.hist(valid_data, bins=100, alpha=0.7, color=color, density=True)
            ax.set_title(name)
            ax.set_xlabel('Transformed Value')
            ax.set_ylabel('Density')
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
            
            # Add statistics
            stats_text = f'Mean: {np.mean(valid_data):.4f}\nStd: {np.std(valid_data):.4f}\nSkew: {stats.skew(valid_data):.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name} (Failed)')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'{var_name}_transformations.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot for {var_name} to {output_path}")

def analyze_transformation_effectiveness(var_name, data, transformed_data, transform_name):
    """Analyze how effective a transformation is."""
    # Remove NaN and inf values
    valid_transformed = transformed_data[np.isfinite(transformed_data)]
    
    if len(valid_transformed) == 0:
        return None
    
    return {
        'variable': var_name,
        'transform': transform_name,
        'original_skew': stats.skew(data),
        'transformed_skew': stats.skew(valid_transformed),
        'original_kurtosis': stats.kurtosis(data),
        'transformed_kurtosis': stats.kurtosis(valid_transformed),
        'percent_finite': 100 * len(valid_transformed) / len(transformed_data),
        'range_compression': (np.max(valid_transformed) - np.min(valid_transformed)) / (np.max(data) - np.min(data))
    }

def main():
    # Configuration
    data_folder = "/u/tawal/BSSN-Extracted-Data/tt_q01/"
    output_dir = "variable_distributions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Variable-specific transformation recommendations
    variable_configs = {
        'U_B2': {
            'recommended': 'asinh',
            'scale': 0.001  # Scale based on std
        },
        'U_SYMAT2': {
            'recommended': 'symmetric_log',
            'scale': 0.1
        },
        'U_GT2': {
            'recommended': 'asinh',
            'scale': 0.01
        },
        'U_SYMGT2': {
            'recommended': 'symmetric_log',
            'scale': 0.01
        },
        'U_SYMAT4': {
            'recommended': 'symmetric_log',
            'scale': 0.1
        },
        'U_SYMAT3': {
            'recommended': 'asinh',
            'scale': 0.1
        }
    }
    
    # Store results
    all_results = []
    
    print("Loading data and applying transformations...")
    
    for var_name, config in variable_configs.items():
        print(f"\nProcessing {var_name}...")
        
        # Load data
        data = load_variable_data(data_folder, var_name)
        
        if data.size == 0:
            print(f"No data found for {var_name}")
            continue
        
        print(f"Loaded {data.size:,} values for {var_name}")
        
        # Apply transformations
        transformations = []
        
        # 1. Poslog (for comparison)
        poslog_data, data_min = poslog_transform(data)
        transformations.append(('Poslog (Current)', poslog_data, 'orange'))
        
        # 2. Recommended transformation
        if config['recommended'] == 'asinh':
            recommended_data = asinh_transform(data, scale=config['scale'])
            transformations.append((f'Asinh (scale={config["scale"]})', recommended_data, 'green'))
        elif config['recommended'] == 'symmetric_log':
            recommended_data = symmetric_log_transform(data, c=config['scale'])
            transformations.append((f'Symmetric Log (c={config["scale"]})', recommended_data, 'green'))
        
        # 3. Alternative transformations
        transformations.append(('Standardization', standardize(data), 'purple'))
        transformations.append(('Robust Scaling', robust_scale(data), 'brown'))
        
        # 4. Additional asinh with different scale
        alt_scale = config['scale'] * 10
        transformations.append((f'Asinh (scale={alt_scale})', asinh_transform(data, scale=alt_scale), 'cyan'))
        
        # Create comparison plots
        plot_transformation_comparison(var_name, data, transformations, output_dir)
        
        # Analyze effectiveness
        for name, transformed_data, _ in transformations:
            result = analyze_transformation_effectiveness(var_name, data, transformed_data, name)
            if result:
                all_results.append(result)
    
    # Save analysis results
    results_file = os.path.join(output_dir, 'transformation_analysis.txt')
    with open(results_file, 'w') as f:
        f.write("TRANSFORMATION EFFECTIVENESS ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        for var_name in variable_configs.keys():
            var_results = [r for r in all_results if r['variable'] == var_name]
            if not var_results:
                continue
                
            f.write(f"\n{var_name}:\n")
            f.write("-" * len(var_name) + "\n")
            f.write(f"{'Transform':<30} {'Orig Skew':>10} {'New Skew':>10} {'Orig Kurt':>10} {'New Kurt':>10} {'Range Comp':>12}\n")
            f.write("-" * 82 + "\n")
            
            for result in var_results:
                f.write(f"{result['transform']:<30} {result['original_skew']:>10.3f} "
                       f"{result['transformed_skew']:>10.3f} {result['original_kurtosis']:>10.3f} "
                       f"{result['transformed_kurtosis']:>10.3f} {result['range_compression']:>12.3f}\n")
            
            # Identify best transformation
            best_for_skew = min(var_results, key=lambda x: abs(x['transformed_skew']))
            f.write(f"\nBest for reducing skewness: {best_for_skew['transform']}\n")
    
    print(f"\nAnalysis complete! Results saved to {results_file}")
    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    main()