#!/usr/bin/env python
"""
Analyze distribution of specific variables from BSSN dataset.
This version only loads and analyzes the requested variables.
Enhanced to test different transformations and find the best one for each variable.
"""

import h5py
import numpy as np
import glob
import os
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt

def load_specific_variables(data_folder, target_vars):
    """Load only specific variables from HDF5 files."""
    file_pattern = os.path.join(data_folder, "bssn_gr_*_extracted.hdf5")
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        raise ValueError(f"No HDF5 files found in {data_folder}")
    
    print(f"Found {len(files)} HDF5 files")
    
    # First, get the list of variables from the first file
    with h5py.File(files[0], 'r') as f:
        all_vars = list(f['vars'][:].astype(str))
        print(f"\nAvailable variables: {all_vars}")
        print(f"Target variables: {target_vars}")
    
    # Initialize dictionary to store data for each variable
    variable_data = {var: [] for var in target_vars if var in all_vars}
    missing_vars = [var for var in target_vars if var not in all_vars]
    
    if missing_vars:
        print(f"\nWARNING: These variables are not in the dataset: {missing_vars}")
    
    # Load data from all files
    for file_path in tqdm(files, desc="Loading files"):
        with h5py.File(file_path, 'r') as f:
            vars_list = f['vars'][:].astype(str)
            var_data = f['var_data'][:]  # Shape: (n_vars, n_blocks, 7, 7, 7)
            
            for var in variable_data.keys():
                if var in vars_list:
                    var_idx = list(vars_list).index(var)
                    # Extract 5x5x5 center crop from 7x7x7 data
                    data_7x7x7 = var_data[var_idx]  # Shape: (n_blocks, 7, 7, 7)
                    data_5x5x5 = data_7x7x7[:, 1:6, 1:6, 1:6]  # Center crop
                    variable_data[var].append(data_5x5x5)
    
    # Concatenate all data for each variable
    for var in list(variable_data.keys()):
        if variable_data[var]:
            variable_data[var] = np.concatenate(variable_data[var], axis=0)
            print(f"{var}: loaded {variable_data[var].shape[0]} blocks")
        else:
            print(f"Warning: No data found for {var}")
            del variable_data[var]
    
    return variable_data

def print_variable_metrics(variable_data, output_file=None):
    """Print detailed metrics for variables and optionally save to file."""
    import sys
    from datetime import datetime
    
    # Store original stdout
    original_stdout = sys.stdout
    
    # If output file is specified, open it
    if output_file:
        f = open(output_file, 'w')
        # Write to both console and file
        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
        sys.stdout = Tee(original_stdout, f)
    
    # Add timestamp
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print("DATA METRICS FOR SPECIFIC VARIABLES")
    print("="*80)
    
    for var_name, data in variable_data.items():
        if data.size > 0:
            data_flat = data.flatten()
            
            print(f"\n{var_name}:")
            print("-" * len(var_name))
            print(f"  Shape: {data.shape}")
            print(f"  Total elements: {data.size:,}")
            print(f"  Data type: {data.dtype}")
            print(f"\n  Statistical Metrics:")
            print(f"    Mean:     {np.mean(data_flat):15.8f}")
            print(f"    Std Dev:  {np.std(data_flat):15.8f}")
            print(f"    Variance: {np.var(data_flat):15.8f}")
            print(f"    Min:      {np.min(data_flat):15.8f}")
            print(f"    Max:      {np.max(data_flat):15.8f}")
            print(f"    Range:    {np.max(data_flat) - np.min(data_flat):15.8f}")
            
            # Percentiles
            percentiles = [1, 5, 25, 50, 75, 95, 99]
            print(f"\n  Percentiles:")
            for p in percentiles:
                val = np.percentile(data_flat, p)
                print(f"    {p:3d}%:     {val:15.8f}")
            
            # Additional metrics
            print(f"\n  Additional Metrics:")
            print(f"    Skewness: {np.mean(((data_flat - np.mean(data_flat)) / np.std(data_flat))**3):15.8f}")
            print(f"    Kurtosis: {np.mean(((data_flat - np.mean(data_flat)) / np.std(data_flat))**4) - 3:15.8f}")
            
            # Check for special values
            n_zeros = np.sum(data_flat == 0)
            n_negative = np.sum(data_flat < 0)
            n_positive = np.sum(data_flat > 0)
            n_nan = np.sum(np.isnan(data_flat))
            n_inf = np.sum(np.isinf(data_flat))
            
            print(f"\n  Value Distribution:")
            print(f"    Zeros:     {n_zeros:10d} ({100*n_zeros/data.size:6.2f}%)")
            print(f"    Negative:  {n_negative:10d} ({100*n_negative/data.size:6.2f}%)")
            print(f"    Positive:  {n_positive:10d} ({100*n_positive/data.size:6.2f}%)")
            print(f"    NaN:       {n_nan:10d} ({100*n_nan/data.size:6.2f}%)")
            print(f"    Inf:       {n_inf:10d} ({100*n_inf/data.size:6.2f}%)")
    
    print("\n" + "="*80)
    
    # Restore stdout and close file if needed
    if output_file:
        sys.stdout = original_stdout
        f.close()

# Transformation functions
def poslog_transform(data, epsilon=1e-8):
    """Standard poslog transformation."""
    data_min = data.min()
    if data_min <= 0:
        data_shifted = data - data_min + epsilon
    else:
        data_shifted = data.copy()
    return np.log(data_shifted), {'min': data_min, 'epsilon': epsilon}

def symmetric_log_transform(data, c=1.0):
    """Symmetric log transformation that preserves sign."""
    return np.sign(data) * np.log1p(np.abs(data) / c) * c, {'c': c}

def asinh_transform(data, scale=1.0):
    """Inverse hyperbolic sine transformation."""
    return np.arcsinh(data / scale), {'scale': scale}

def standardize(data):
    """Standard z-score normalization."""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / (std + 1e-8), {'mean': mean, 'std': std}

def robust_scale(data):
    """Robust scaling using median and IQR."""
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    median = np.median(data)
    if iqr > 0:
        return (data - median) / iqr, {'median': median, 'iqr': iqr}
    else:
        return data - median, {'median': median, 'iqr': 0}

def evaluate_transformation(original_data, transformed_data, transform_name):
    """
    Evaluate how good a transformation is for neural network training.
    Lower score is better.
    """
    # Remove any infinite or NaN values
    valid_mask = np.isfinite(transformed_data)
    if not np.any(valid_mask):
        return float('inf'), {}
    
    transformed_clean = transformed_data[valid_mask]
    
    # Calculate metrics
    metrics = {
        'skewness': abs(stats.skew(transformed_clean)),
        'kurtosis': abs(stats.kurtosis(transformed_clean)),
        'range': np.max(transformed_clean) - np.min(transformed_clean),
        'std': np.std(transformed_clean),
        'percent_finite': 100 * np.sum(valid_mask) / len(transformed_data)
    }
    
    # Score components (lower is better)
    score_components = {
        'skewness_score': metrics['skewness'] * 10,  # Penalize skewness heavily
        'kurtosis_score': max(0, metrics['kurtosis'] - 3) * 2,  # Penalize excess kurtosis
        'range_score': max(0, metrics['range'] - 10) * 0.5,  # Penalize very large ranges
        'stability_score': 100 - metrics['percent_finite'],  # Penalize infinite values
    }
    
    # Special handling for symmetric distributions
    if transform_name in ['symmetric_log', 'asinh'] and abs(metrics['skewness']) < 0.1:
        score_components['symmetry_bonus'] = -5  # Reward maintaining symmetry
    else:
        score_components['symmetry_bonus'] = 0
    
    # Calculate total score
    total_score = sum(score_components.values())
    
    return total_score, {**metrics, **score_components, 'total_score': total_score}

def test_transformations(variable_data, output_dir):
    """Test different transformations on each variable and find the best one."""
    
    print("\n" + "="*80)
    print("TESTING TRANSFORMATIONS")
    print("="*80)
    
    results = {}
    recommendations = {}
    
    for var_name, data in variable_data.items():
        data_flat = data.flatten()
        
        print(f"\nTesting transformations for {var_name}...")
        
        # Define transformations to test
        transformations = [
            ('poslog', lambda d: poslog_transform(d)),
            ('symmetric_log_0.01', lambda d: symmetric_log_transform(d, c=0.01)),
            ('symmetric_log_0.1', lambda d: symmetric_log_transform(d, c=0.1)),
            ('symmetric_log_1.0', lambda d: symmetric_log_transform(d, c=1.0)),
            ('asinh_0.001', lambda d: asinh_transform(d, scale=0.001)),
            ('asinh_0.01', lambda d: asinh_transform(d, scale=0.01)),
            ('asinh_0.1', lambda d: asinh_transform(d, scale=0.1)),
            ('asinh_1.0', lambda d: asinh_transform(d, scale=1.0)),
            ('standardize', lambda d: standardize(d)),
            ('robust_scale', lambda d: robust_scale(d)),
        ]
        
        var_results = []
        
        for transform_name, transform_func in transformations:
            try:
                transformed_data, params = transform_func(data_flat)
                score, metrics = evaluate_transformation(data_flat, transformed_data, transform_name.split('_')[0])
                
                var_results.append({
                    'name': transform_name,
                    'score': score,
                    'metrics': metrics,
                    'params': params
                })
            except Exception as e:
                print(f"  Error with {transform_name}: {e}")
        
        # Sort by score
        var_results.sort(key=lambda x: x['score'])
        results[var_name] = var_results
        
        # Get best transformation
        if var_results:
            best = var_results[0]
            recommendations[var_name] = best
            
            print(f"\nTop 3 transformations for {var_name}:")
            for i, result in enumerate(var_results[:3]):
                print(f"  {i+1}. {result['name']:20s} Score: {result['score']:8.2f}")
    
    # Save recommendations
    rec_file = os.path.join(output_dir, "transformation_recommendations.txt")
    with open(rec_file, 'w') as f:
        f.write("BEST TRANSFORMATIONS FOR EACH VARIABLE\n")
        f.write("="*80 + "\n\n")
        
        for var_name, best in recommendations.items():
            f.write(f"{var_name}:\n")
            f.write(f"  Best: {best['name']}\n")
            f.write(f"  Score: {best['score']:.2f}\n")
            f.write(f"  Parameters: {best['params']}\n")
            f.write(f"  Skewness after: {best['metrics']['skewness']:.3f}\n")
            f.write(f"  Kurtosis after: {best['metrics']['kurtosis']:.3f}\n\n")
    
    print(f"\nRecommendations saved to {rec_file}")
    
    # Create comparison plots
    create_transformation_plots(variable_data, results, output_dir)
    
    return recommendations

def create_transformation_plots(variable_data, results, output_dir):
    """Create plots comparing transformations for each variable."""
    
    for var_name, data in variable_data.items():
        data_flat = data.flatten()
        var_results = results[var_name]
        
        # Create figure with subplots
        n_plots = min(5, len(var_results) + 1)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        # Plot original
        ax = axes[0]
        ax.hist(data_flat[np.isfinite(data_flat)], bins=50, alpha=0.7, color='blue', density=True)
        ax.set_title(f'{var_name}\nOriginal')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        
        # Plot top transformations
        colors = ['green', 'orange', 'red', 'purple']
        for i, result in enumerate(var_results[:n_plots-1]):
            ax = axes[i+1]
            
            # Apply transformation
            transform_name = result['name']
            if 'poslog' in transform_name:
                transformed, _ = poslog_transform(data_flat)
            elif 'symmetric_log' in transform_name:
                c = float(transform_name.split('_')[-1]) if '_' in transform_name else 1.0
                transformed, _ = symmetric_log_transform(data_flat, c=c)
            elif 'asinh' in transform_name:
                scale = float(transform_name.split('_')[-1]) if '_' in transform_name else 1.0
                transformed, _ = asinh_transform(data_flat, scale=scale)
            elif transform_name == 'standardize':
                transformed, _ = standardize(data_flat)
            elif transform_name == 'robust_scale':
                transformed, _ = robust_scale(data_flat)
            
            # Plot
            valid_data = transformed[np.isfinite(transformed)]
            if len(valid_data) > 0:
                ax.hist(valid_data, bins=50, alpha=0.7, color=colors[i % len(colors)], density=True)
            
            ax.set_title(f'{transform_name}\nScore: {result["score"]:.1f}')
            ax.set_xlabel('Transformed Value')
        
        plt.suptitle(f'Transformation Comparison for {var_name}', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{var_name}_transformations.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved transformation plot for {var_name}")

def main():
    # Data folder path (adjust if needed)
    data_folder = "/u/tawal/BSSN-Extracted-Data/tt_q01/"
    
    # Check if folder exists
    if not os.path.exists(data_folder):
        # Try alternative path
        data_folder = "/u/tawal/BSSN Extracted Data/tt_q01/"
        if not os.path.exists(data_folder):
            print(f"Error: Data folder not found at either path")
            print("Please update the data_folder path in the script")
            return
    
    print(f"Loading data from: {data_folder}")
    
    # Define the specific variables to analyze
    target_variables = ['U_B2', 'U_SYMAT2', 'U_GT2', 'U_SYMGT2', 'U_SYMAT4', 'U_SYMAT3']
    
    # Load only the specific variables
    variable_data = load_specific_variables(data_folder, target_variables)
    
    # Create output directory if it doesn't exist
    output_dir = "variable_distributions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(output_dir, "specific_variables_metrics.txt")
    
    # Print metrics for the variables and save to file
    print_variable_metrics(variable_data, output_file)
    
    print(f"\nAnalysis complete! Metrics saved to {output_file}")
    
    # Test different transformations to find the best ones
    recommendations = test_transformations(variable_data, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("TRANSFORMATION RECOMMENDATIONS SUMMARY")
    print("="*80)
    for var_name, best in recommendations.items():
        print(f"{var_name}: Use {best['name']} (score: {best['score']:.2f})")

if __name__ == "__main__":
    main()