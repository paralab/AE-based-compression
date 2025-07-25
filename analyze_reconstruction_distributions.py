#!/usr/bin/env python3
"""
Analyze reconstruction distributions and save detailed metrics
"""

import h5py
import numpy as np
import os
from collections import defaultdict

# Define problematic variables
PROBLEMATIC_VARS = ['U_B2', 'U_SYMAT2', 'U_GT2', 'U_SYMGT2', 'U_SYMAT4', 'U_SYMAT3']


def analyze_variable_distribution(data, var_name):
    """Comprehensive analysis of variable distribution"""
    
    # Flatten data for analysis
    flat_data = data.flatten()
    
    # Remove NaN/inf if any
    finite_mask = np.isfinite(flat_data)
    clean_data = flat_data[finite_mask]
    
    if len(clean_data) == 0:
        return None
    
    # Basic statistics
    stats = {
        'count': len(clean_data),
        'mean': np.mean(clean_data),
        'std': np.std(clean_data),
        'min': np.min(clean_data),
        'max': np.max(clean_data),
        'median': np.median(clean_data),
        'q1': np.percentile(clean_data, 25),
        'q3': np.percentile(clean_data, 75),
        'iqr': np.percentile(clean_data, 75) - np.percentile(clean_data, 25),
        'skewness': calculate_skewness(clean_data),
        'kurtosis': calculate_kurtosis(clean_data),
    }
    
    # Zero analysis
    stats['exact_zeros'] = np.sum(clean_data == 0)
    stats['near_zeros_1e-10'] = np.sum(np.abs(clean_data) < 1e-10)
    stats['near_zeros_1e-8'] = np.sum(np.abs(clean_data) < 1e-8)
    stats['near_zeros_1e-6'] = np.sum(np.abs(clean_data) < 1e-6)
    stats['near_zeros_1e-4'] = np.sum(np.abs(clean_data) < 1e-4)
    
    # Sign analysis
    stats['positive_count'] = np.sum(clean_data > 0)
    stats['negative_count'] = np.sum(clean_data < 0)
    stats['positive_ratio'] = stats['positive_count'] / len(clean_data)
    stats['negative_ratio'] = stats['negative_count'] / len(clean_data)
    
    # Scale analysis
    abs_data = np.abs(clean_data[clean_data != 0])  # Exclude exact zeros
    if len(abs_data) > 0:
        stats['typical_magnitude'] = np.median(abs_data)
        stats['magnitude_mean'] = np.mean(abs_data)
        stats['magnitude_std'] = np.std(abs_data)
        stats['log10_magnitude_mean'] = np.mean(np.log10(abs_data + 1e-100))
        stats['log10_magnitude_std'] = np.std(np.log10(abs_data + 1e-100))
        stats['scale_range'] = np.log10(np.max(abs_data) / (np.min(abs_data) + 1e-100) + 1e-100)
    else:
        stats['typical_magnitude'] = 0
        stats['magnitude_mean'] = 0
        stats['magnitude_std'] = 0
        stats['log10_magnitude_mean'] = -100
        stats['log10_magnitude_std'] = 0
        stats['scale_range'] = 0
    
    # Percentile values
    percentiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    for p in percentiles:
        stats[f'p{p}'] = np.percentile(clean_data, p)
    
    # Dynamic range
    if stats['std'] > 0:
        stats['coefficient_of_variation'] = abs(stats['std'] / (stats['mean'] + 1e-100))
    else:
        stats['coefficient_of_variation'] = 0
    
    return stats


def calculate_skewness(data):
    """Calculate skewness"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)


def calculate_kurtosis(data):
    """Calculate kurtosis"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 4)


def format_stats_report(var_name, orig_stats, recon_stats):
    """Format statistics into readable report"""
    
    report = f"\n{'='*80}\n"
    report += f"Variable: {var_name}\n"
    report += f"Model Type: {'asinh' if var_name in PROBLEMATIC_VARS else 'poslog'}\n"
    report += f"{'='*80}\n\n"
    
    # Scale information
    report += "SCALE INFORMATION:\n"
    report += f"  Original range: [{orig_stats['min']:.9e}, {orig_stats['max']:.9e}]\n"
    report += f"  Reconstructed range: [{recon_stats['min']:.9e}, {recon_stats['max']:.9e}]\n"
    report += f"  Original typical magnitude: {orig_stats['typical_magnitude']:.9e}\n"
    report += f"  Reconstructed typical magnitude: {recon_stats['typical_magnitude']:.9e}\n"
    report += f"  Scale range (log10): Original={orig_stats['scale_range']:.2f}, Reconstructed={recon_stats['scale_range']:.2f}\n"
    report += "\n"
    
    # Distribution shape
    report += "DISTRIBUTION SHAPE:\n"
    report += f"  Mean: Original={orig_stats['mean']:.9e}, Reconstructed={recon_stats['mean']:.9e}\n"
    report += f"  Std: Original={orig_stats['std']:.9e}, Reconstructed={recon_stats['std']:.9e}\n"
    report += f"  Median: Original={orig_stats['median']:.9e}, Reconstructed={recon_stats['median']:.9e}\n"
    report += f"  Skewness: Original={orig_stats['skewness']:.4f}, Reconstructed={recon_stats['skewness']:.4f}\n"
    report += f"  Kurtosis: Original={orig_stats['kurtosis']:.4f}, Reconstructed={recon_stats['kurtosis']:.4f}\n"
    report += f"  Coefficient of Variation: Original={orig_stats['coefficient_of_variation']:.4f}, Reconstructed={recon_stats['coefficient_of_variation']:.4f}\n"
    report += "\n"
    
    # Zero analysis
    report += "ZERO/NEAR-ZERO ANALYSIS:\n"
    report += f"  Exact zeros: Original={orig_stats['exact_zeros']} ({100*orig_stats['exact_zeros']/orig_stats['count']:.2f}%), "
    report += f"Reconstructed={recon_stats['exact_zeros']} ({100*recon_stats['exact_zeros']/recon_stats['count']:.2f}%)\n"
    report += f"  Near-zeros (<1e-10): Original={orig_stats['near_zeros_1e-10']} ({100*orig_stats['near_zeros_1e-10']/orig_stats['count']:.2f}%), "
    report += f"Reconstructed={recon_stats['near_zeros_1e-10']} ({100*recon_stats['near_zeros_1e-10']/recon_stats['count']:.2f}%)\n"
    report += f"  Near-zeros (<1e-6): Original={orig_stats['near_zeros_1e-6']} ({100*orig_stats['near_zeros_1e-6']/orig_stats['count']:.2f}%), "
    report += f"Reconstructed={recon_stats['near_zeros_1e-6']} ({100*recon_stats['near_zeros_1e-6']/recon_stats['count']:.2f}%)\n"
    report += "\n"
    
    # Sign analysis
    report += "SIGN DISTRIBUTION:\n"
    report += f"  Positive ratio: Original={orig_stats['positive_ratio']:.2%}, Reconstructed={recon_stats['positive_ratio']:.2%}\n"
    report += f"  Negative ratio: Original={orig_stats['negative_ratio']:.2%}, Reconstructed={recon_stats['negative_ratio']:.2%}\n"
    report += "\n"
    
    # Key percentiles
    report += "KEY PERCENTILES:\n"
    report += "  Percentile    Original          Reconstructed     Difference\n"
    for p in [0.1, 1, 10, 50, 90, 99, 99.9]:
        orig_val = orig_stats[f'p{p}']
        recon_val = recon_stats[f'p{p}']
        diff = recon_val - orig_val
        report += f"  {p:5.1f}%      {orig_val:15.9e}   {recon_val:15.9e}   {diff:15.9e}\n"
    
    # Error analysis
    report += "\nERROR ANALYSIS:\n"
    abs_errors = []
    rel_errors = []
    
    # Calculate sample-wise errors (using indices for proper comparison)
    # Note: This is approximate since we're using flattened data
    n_samples = min(1000, orig_stats['count'])  # Sample for error calculation
    
    report += f"  MSE: {np.mean((orig_stats['mean'] - recon_stats['mean'])**2):.9e}\n"
    report += f"  Mean absolute difference: {abs(orig_stats['mean'] - recon_stats['mean']):.9e}\n"
    report += f"  Median absolute difference: {abs(orig_stats['median'] - recon_stats['median']):.9e}\n"
    report += f"  Std difference: {abs(orig_stats['std'] - recon_stats['std']):.9e}\n"
    
    return report


def main():
    # File paths
    original_file = "/u/tawal/BSSN-Extracted-Data/tt_q08/bssn_gr_11200_extracted.hdf5"
    reconstructed_file = "./reconstructed/bssn_gr_11200_log_loss_reconstructed.hdf5"
    output_file = "./reconstruction_distribution_analysis.txt"
    
    print("Loading data files...")
    
    # Load data
    with h5py.File(original_file, 'r') as f_orig:
        var_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                     for name in f_orig['vars'][:]]
        original_data = f_orig['var_data'][:]
    
    with h5py.File(reconstructed_file, 'r') as f_recon:
        reconstructed_data = f_recon['var_data'][:]
    
    # Filter to U_ variables
    u_var_indices = [(i, name) for i, name in enumerate(var_names) if name.startswith('U_')]
    
    # Analyze each variable
    full_report = "RECONSTRUCTION DISTRIBUTION ANALYSIS\n"
    full_report += f"Original file: {original_file}\n"
    full_report += f"Reconstructed file: {reconstructed_file}\n"
    full_report += f"Analysis date: {np.datetime64('now')}\n"
    full_report += "\n"
    
    # Summary statistics
    summary_stats = defaultdict(list)
    
    for var_idx, var_name in u_var_indices:
        print(f"Analyzing {var_name}...")
        
        # Get data
        orig_var_data = original_data[var_idx]
        recon_var_data = reconstructed_data[var_idx]
        
        # Analyze distributions
        orig_stats = analyze_variable_distribution(orig_var_data, var_name)
        recon_stats = analyze_variable_distribution(recon_var_data, var_name)
        
        if orig_stats and recon_stats:
            # Generate detailed report
            var_report = format_stats_report(var_name, orig_stats, recon_stats)
            full_report += var_report
            
            # Collect summary statistics
            summary_stats['variable'].append(var_name)
            summary_stats['model'].append('asinh' if var_name in PROBLEMATIC_VARS else 'poslog')
            summary_stats['orig_scale'].append(orig_stats['scale_range'])
            summary_stats['near_zero_ratio'].append(orig_stats['near_zeros_1e-6'] / orig_stats['count'])
            summary_stats['mean_diff'].append(abs(orig_stats['mean'] - recon_stats['mean']))
            summary_stats['std_diff'].append(abs(orig_stats['std'] - recon_stats['std']))
    
    # Add summary section
    full_report += "\n" + "="*80 + "\n"
    full_report += "SUMMARY OF CONCERNS:\n"
    full_report += "="*80 + "\n\n"
    
    # Identify problematic reconstructions
    concerns = []
    
    for i, var_name in enumerate(summary_stats['variable']):
        if summary_stats['near_zero_ratio'][i] > 0.3:
            concerns.append(f"{var_name}: High proportion of near-zero values ({summary_stats['near_zero_ratio'][i]:.1%})")
        if summary_stats['orig_scale'][i] > 10:
            concerns.append(f"{var_name}: Extreme scale range (10^{summary_stats['orig_scale'][i]:.1f})")
        if summary_stats['mean_diff'][i] > 1e-4:
            concerns.append(f"{var_name}: Large mean difference ({summary_stats['mean_diff'][i]:.2e})")
    
    if concerns:
        full_report += "IDENTIFIED CONCERNS:\n"
        for concern in concerns:
            full_report += f"  - {concern}\n"
    else:
        full_report += "No major concerns identified.\n"
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(full_report)
    
    print(f"\nAnalysis complete. Report saved to: {output_file}")
    
    # Also save a CSV for easier analysis
    csv_file = "./reconstruction_distribution_summary.csv"
    with open(csv_file, 'w') as f:
        f.write("Variable,Model,Scale_Range,Near_Zero_Ratio,Mean_Diff,Std_Diff\n")
        for i in range(len(summary_stats['variable'])):
            f.write(f"{summary_stats['variable'][i]},{summary_stats['model'][i]},")
            f.write(f"{summary_stats['orig_scale'][i]:.2f},{summary_stats['near_zero_ratio'][i]:.4f},")
            f.write(f"{summary_stats['mean_diff'][i]:.9e},{summary_stats['std_diff'][i]:.9e}\n")
    
    print(f"Summary CSV saved to: {csv_file}")


if __name__ == "__main__":
    main()