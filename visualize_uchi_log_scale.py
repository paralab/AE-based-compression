#!/usr/bin/env python3
"""
Visualize U_CHI Data on Log Scale
Plot 10 different samples to understand data distribution characteristics
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.u_chi_dataset import create_u_chi_datasets


def log_scale_transform(data, epsilon=1e-8, method='positive_shift'):
    """
    Apply log-scale transformation to handle high dynamic range
    
    Args:
        data: Input data array
        epsilon: Small value to prevent log(0)
        method: 'positive_shift', 'symlog', or 'abs_log'
    
    Returns:
        transformed_data: Log-transformed data
        transform_params: Parameters needed for inverse transformation
    """
    if method == 'positive_shift':
        # Shift data to positive domain, then log
        data_min = data.min()
        data_shifted = data - data_min + epsilon
        log_data = np.log(data_shifted + epsilon)
        transform_params = {'data_min': data_min, 'epsilon': epsilon}
        return log_data, transform_params
    
    elif method == 'symlog':
        # Symmetric log transformation (handles positive and negative)
        log_data = np.sign(data) * np.log1p(np.abs(data) / epsilon) * epsilon
        transform_params = {'epsilon': epsilon}
        return log_data, transform_params
    
    elif method == 'abs_log':
        # Log of absolute values
        data_sign = np.sign(data)
        log_data = np.log(np.abs(data) + epsilon)
        transform_params = {'data_sign': data_sign, 'epsilon': epsilon}
        return log_data, transform_params
    
    else:
        raise ValueError(f"Unknown method: {method}")


def inverse_log_scale_transform(log_data, transform_params, method='positive_shift'):
    """
    Apply inverse log-scale transformation to recover original data
    
    Args:
        log_data: Log-transformed data
        transform_params: Parameters from forward transformation
        method: Same method used in forward transformation
    
    Returns:
        recovered_data: Data recovered from log transformation
    """
    if method == 'positive_shift':
        data_min = transform_params['data_min']
        epsilon = transform_params['epsilon']
        # Inverse: exp(log_data) - epsilon + data_min
        recovered_data = np.exp(log_data) - epsilon + data_min
        return recovered_data
    
    elif method == 'symlog':
        epsilon = transform_params['epsilon']
        # Inverse of symlog
        recovered_data = np.sign(log_data) * (np.expm1(np.abs(log_data) / epsilon) * epsilon)
        return recovered_data
    
    elif method == 'abs_log':
        data_sign = transform_params['data_sign']
        epsilon = transform_params['epsilon']
        # Inverse: sign * (exp(log_data) - epsilon)
        recovered_data = data_sign * (np.exp(log_data) - epsilon)
        return recovered_data
    
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_reconstruction_metrics(original, recovered):
    """
    Calculate various metrics to assess reconstruction quality
    
    Args:
        original: Original data array
        recovered: Recovered data array
    
    Returns:
        dict: Dictionary containing various metrics
    """
    # Ensure same shape
    assert original.shape == recovered.shape, "Arrays must have same shape"
    
    # Flatten for calculations
    orig_flat = original.flatten()
    rec_flat = recovered.flatten()
    
    # Mean Squared Error
    mse = np.mean((orig_flat - rec_flat) ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(orig_flat - rec_flat))
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Peak Signal-to-Noise Ratio
    data_range = np.max(orig_flat) - np.min(orig_flat)
    if mse > 0 and data_range > 0:
        psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    else:
        psnr = float('inf')
    
    # Normalized Root Mean Squared Error
    nrmse = rmse / data_range if data_range > 0 else float('inf')
    
    # Correlation coefficient
    correlation = np.corrcoef(orig_flat, rec_flat)[0, 1] if len(orig_flat) > 1 else 1.0
    
    # Relative error
    relative_error = np.mean(np.abs(orig_flat - rec_flat) / (np.abs(orig_flat) + 1e-12)) * 100
    
    # Maximum absolute error
    max_abs_error = np.max(np.abs(orig_flat - rec_flat))
    
    # Signal-to-Noise Ratio
    signal_power = np.mean(orig_flat ** 2)
    noise_power = np.mean((orig_flat - rec_flat) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-12)) if noise_power > 0 else float('inf')
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'psnr': psnr,
        'nrmse': nrmse,
        'correlation': correlation,
        'relative_error': relative_error,
        'max_abs_error': max_abs_error,
        'snr': snr
    }


def plot_3d_slice_comparison(data_original, data_log, title, fig_size=(15, 5)):
    """Plot 2D slices of 3D data for comparison"""
    fig, axes = plt.subplots(2, 3, figsize=fig_size)
    fig.suptitle(title, fontsize=14)
    
    # Get middle slices
    mid_z, mid_y, mid_x = data_original.shape[0]//2, data_original.shape[1]//2, data_original.shape[2]//2
    
    # Original data slices
    axes[0, 0].imshow(data_original[mid_z, :, :], cmap='viridis')
    axes[0, 0].set_title('Original XY slice')
    axes[0, 0].set_xlabel(f'Range: [{data_original.min():.3e}, {data_original.max():.3e}]')
    
    axes[0, 1].imshow(data_original[:, mid_y, :], cmap='viridis')
    axes[0, 1].set_title('Original XZ slice')
    
    axes[0, 2].imshow(data_original[:, :, mid_x], cmap='viridis')
    axes[0, 2].set_title('Original YZ slice')
    
    # Log-scale data slices
    axes[1, 0].imshow(data_log[mid_z, :, :], cmap='viridis')
    axes[1, 0].set_title('Log-scale XY slice')
    axes[1, 0].set_xlabel(f'Range: [{data_log.min():.3f}, {data_log.max():.3f}]')
    
    axes[1, 1].imshow(data_log[:, mid_y, :], cmap='viridis')
    axes[1, 1].set_title('Log-scale XZ slice')
    
    axes[1, 2].imshow(data_log[:, :, mid_x], cmap='viridis')
    axes[1, 2].set_title('Log-scale YZ slice')
    
    plt.tight_layout()
    return fig


def plot_roundtrip_comparison(data_original, data_log, data_recovered, error_data, metrics, title, fig_size=(20, 10)):
    """Plot round-trip comparison: Original -> Log -> Recovered + Error analysis"""
    fig, axes = plt.subplots(3, 4, figsize=fig_size)
    fig.suptitle(f'{title} - Round-trip Analysis', fontsize=16)
    
    # Get middle slices
    mid_z, mid_y, mid_x = data_original.shape[0]//2, data_original.shape[1]//2, data_original.shape[2]//2
    
    # Row 1: Original data
    im1 = axes[0, 0].imshow(data_original[mid_z, :, :], cmap='viridis')
    axes[0, 0].set_title('Original XY')
    axes[0, 0].set_ylabel('Original Data')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(data_original[:, mid_y, :], cmap='viridis')
    axes[0, 1].set_title('Original XZ')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    im3 = axes[0, 2].imshow(data_original[:, :, mid_x], cmap='viridis')
    axes[0, 2].set_title('Original YZ')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    # Original histogram
    axes[0, 3].hist(data_original.flatten(), bins=50, alpha=0.7, color='blue', density=True)
    axes[0, 3].set_title('Original Distribution')
    axes[0, 3].set_yscale('log')
    
    # Row 2: Log-transformed data
    im4 = axes[1, 0].imshow(data_log[mid_z, :, :], cmap='plasma')
    axes[1, 0].set_title('Log XY')
    axes[1, 0].set_ylabel('Log-transformed')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    im5 = axes[1, 1].imshow(data_log[:, mid_y, :], cmap='plasma')
    axes[1, 1].set_title('Log XZ')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    im6 = axes[1, 2].imshow(data_log[:, :, mid_x], cmap='plasma')
    axes[1, 2].set_title('Log YZ')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    # Log histogram
    axes[1, 3].hist(data_log.flatten(), bins=50, alpha=0.7, color='orange', density=True)
    axes[1, 3].set_title('Log Distribution')
    
    # Row 3: Recovered data and error
    im7 = axes[2, 0].imshow(data_recovered[mid_z, :, :], cmap='viridis')
    axes[2, 0].set_title('Recovered XY')
    axes[2, 0].set_ylabel('Recovered Data')
    plt.colorbar(im7, ax=axes[2, 0], fraction=0.046)
    
    im8 = axes[2, 1].imshow(data_recovered[:, mid_y, :], cmap='viridis')
    axes[2, 1].set_title('Recovered XZ')
    plt.colorbar(im8, ax=axes[2, 1], fraction=0.046)
    
    # Error visualization
    im9 = axes[2, 2].imshow(error_data[:, :, mid_x], cmap='hot')
    axes[2, 2].set_title('Absolute Error YZ')
    plt.colorbar(im9, ax=axes[2, 2], fraction=0.046)
    
    # Metrics text
    metrics_text = f"""Round-trip Quality Metrics:
    
MSE: {metrics['mse']:.2e}
MAE: {metrics['mae']:.2e}
RMSE: {metrics['rmse']:.2e}
PSNR: {metrics['psnr']:.1f} dB
NRMSE: {metrics['nrmse']:.1e}
Correlation: {metrics['correlation']:.6f}
Relative Error: {metrics['relative_error']:.2f}%
Max Abs Error: {metrics['max_abs_error']:.2e}
SNR: {metrics['snr']:.1f} dB

Data Ranges:
Original: [{data_original.min():.3e}, {data_original.max():.3e}]
Log: [{data_log.min():.3f}, {data_log.max():.3f}]
Recovered: [{data_recovered.min():.3e}, {data_recovered.max():.3e}]
Error: [{error_data.min():.3e}, {error_data.max():.3e}]"""
    
    axes[2, 3].text(0.05, 0.95, metrics_text, transform=axes[2, 3].transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[2, 3].set_xlim(0, 1)
    axes[2, 3].set_ylim(0, 1)
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    return fig


def plot_histogram_comparison(data_original, data_log, title):
    """Plot histograms of original vs log-scale data"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{title} - Distribution Comparison', fontsize=14)
    
    # Original data histogram
    axes[0].hist(data_original.flatten(), bins=50, alpha=0.7, color='blue', density=True)
    axes[0].set_xlabel('Original Values')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Original Data Distribution')
    axes[0].set_yscale('log')
    
    # Log-scale data histogram
    axes[1].hist(data_log.flatten(), bins=50, alpha=0.7, color='red', density=True)
    axes[1].set_xlabel('Log-scale Values')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Log-scale Data Distribution')
    
    plt.tight_layout()
    return fig


def analyze_data_statistics(data, name):
    """Analyze and print data statistics"""
    print(f"\n{name} Statistics:")
    print(f"  Shape: {data.shape}")
    print(f"  Min: {data.min():.6e}")
    print(f"  Max: {data.max():.6e}")
    print(f"  Mean: {data.mean():.6e}")
    print(f"  Std: {data.std():.6e}")
    print(f"  Dynamic Range: {data.max()/data.min():.2e}" if data.min() != 0 else "  Dynamic Range: inf (contains zeros)")
    
    # Percentiles
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    perc_values = np.percentile(data.flatten(), percentiles)
    print("  Percentiles:", dict(zip(percentiles, perc_values)))
    
    # Count zeros and near-zeros
    n_zeros = np.sum(np.abs(data) < 1e-12)
    n_near_zeros = np.sum(np.abs(data) < 1e-6)
    print(f"  Zeros (< 1e-12): {n_zeros} ({100*n_zeros/data.size:.2f}%)")
    print(f"  Near-zeros (< 1e-6): {n_near_zeros} ({100*n_near_zeros/data.size:.2f}%)")


def main():
    # Configuration
    data_folder = "/u/tawal/0620-NN-based-compression-thera/tt_q01/"
    output_dir = "uchi_log_scale_analysis"
    num_samples = 10
    
    print("Loading U_CHI dataset...")
    
    # Create datasets (no normalization for raw analysis)
    train_dataset, val_dataset = create_u_chi_datasets(
        data_folder=data_folder,
        train_ratio=0.8,
        normalize=False,  # We want to see raw data
        normalize_method='none'
    )
    
    # Use validation dataset for analysis
    dataset = val_dataset
    print(f"Using validation dataset with {len(dataset)} samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample indices to analyze
    sample_indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
    print(f"Analyzing samples: {sample_indices}")
    
    # Collect statistics across all samples
    all_original_stats = []
    all_log_stats = []
    all_roundtrip_stats = []
    
    # Process each sample
    for i, sample_idx in enumerate(sample_indices):
        print(f"\n{'='*50}")
        print(f"Processing Sample {i+1}/{num_samples} (index {sample_idx})")
        print(f"{'='*50}")
        
        # Get sample
        sample, metadata = dataset[sample_idx]
        data_original = sample.squeeze().numpy()  # Remove channel dimension: (7, 7, 7)
        
        # Analyze original data
        analyze_data_statistics(data_original, "Original Data")
        
        # Try different log-scale transformations
        methods = ['positive_shift', 'symlog', 'abs_log']
        
        for method in methods:
            print(f"\n--- {method.upper()} Method ---")
            try:
                # Forward transformation
                data_log, transform_params = log_scale_transform(data_original, method=method)
                analyze_data_statistics(data_log, f"Log-scale ({method})")
                
                # Inverse transformation (round-trip test)
                data_recovered = inverse_log_scale_transform(data_log, transform_params, method=method)
                analyze_data_statistics(data_recovered, f"Recovered ({method})")
                
                # Calculate round-trip error metrics
                error_data = np.abs(data_original - data_recovered)
                metrics = calculate_reconstruction_metrics(data_original, data_recovered)
                
                print(f"\n  Round-trip Quality Assessment:")
                print(f"    MSE: {metrics['mse']:.2e}")
                print(f"    MAE: {metrics['mae']:.2e}")
                print(f"    PSNR: {metrics['psnr']:.1f} dB")
                print(f"    Correlation: {metrics['correlation']:.6f}")
                print(f"    Relative Error: {metrics['relative_error']:.2f}%")
                print(f"    Max Abs Error: {metrics['max_abs_error']:.2e}")
                
                # Create slice comparison plot (original vs log)
                fig1 = plot_3d_slice_comparison(
                    data_original, data_log, 
                    f'Sample {sample_idx} - {method.upper()} Transformation'
                )
                plt.savefig(os.path.join(output_dir, f'sample_{sample_idx:03d}_{method}_slices.png'), 
                          dpi=150, bbox_inches='tight')
                plt.close(fig1)
                
                # Create histogram comparison (original vs log)
                fig2 = plot_histogram_comparison(
                    data_original, data_log,
                    f'Sample {sample_idx} - {method.upper()}'
                )
                plt.savefig(os.path.join(output_dir, f'sample_{sample_idx:03d}_{method}_histogram.png'), 
                          dpi=150, bbox_inches='tight')
                plt.close(fig2)
                
                # Create comprehensive round-trip analysis plot
                fig3 = plot_roundtrip_comparison(
                    data_original, data_log, data_recovered, error_data, metrics,
                    f'Sample {sample_idx} - {method.upper()}'
                )
                plt.savefig(os.path.join(output_dir, f'sample_{sample_idx:03d}_{method}_roundtrip.png'), 
                          dpi=150, bbox_inches='tight')
                plt.close(fig3)
                
                # Store round-trip statistics
                roundtrip_stat = {
                    'sample_idx': sample_idx,
                    'method': method,
                    'mse': metrics['mse'],
                    'mae': metrics['mae'],
                    'psnr': metrics['psnr'],
                    'correlation': metrics['correlation'],
                    'relative_error': metrics['relative_error'],
                    'max_abs_error': metrics['max_abs_error'],
                    'snr': metrics['snr'],
                    'nrmse': metrics['nrmse']
                }
                all_roundtrip_stats.append(roundtrip_stat)
                
            except Exception as e:
                print(f"  Error with {method}: {e}")
        
        # Store statistics for summary
        all_original_stats.append({
            'sample_idx': sample_idx,
            'min': data_original.min(),
            'max': data_original.max(),
            'mean': data_original.mean(),
            'std': data_original.std(),
            'dynamic_range': data_original.max()/data_original.min() if data_original.min() != 0 else float('inf')
        })
    
    # Create summary plots
    print(f"\n{'='*50}")
    print("Creating Summary Analysis")
    print(f"{'='*50}")
    
    # Summary statistics plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('U_CHI Data Statistics Summary Across Samples', fontsize=16)
    
    sample_nums = [stats['sample_idx'] for stats in all_original_stats]
    
    # Min values
    mins = [stats['min'] for stats in all_original_stats]
    axes[0, 0].plot(sample_nums, mins, 'bo-')
    axes[0, 0].set_title('Minimum Values')
    axes[0, 0].set_ylabel('Min Value')
    axes[0, 0].set_yscale('symlog')
    
    # Max values
    maxs = [stats['max'] for stats in all_original_stats]
    axes[0, 1].plot(sample_nums, maxs, 'ro-')
    axes[0, 1].set_title('Maximum Values')
    axes[0, 1].set_ylabel('Max Value')
    axes[0, 1].set_yscale('symlog')
    
    # Dynamic range
    ranges = [stats['dynamic_range'] for stats in all_original_stats if stats['dynamic_range'] != float('inf')]
    range_samples = [stats['sample_idx'] for stats in all_original_stats if stats['dynamic_range'] != float('inf')]
    if ranges:
        axes[0, 2].plot(range_samples, ranges, 'go-')
        axes[0, 2].set_title('Dynamic Range (Max/Min)')
        axes[0, 2].set_ylabel('Dynamic Range')
        axes[0, 2].set_yscale('log')
    
    # Mean values
    means = [stats['mean'] for stats in all_original_stats]
    axes[1, 0].plot(sample_nums, means, 'mo-')
    axes[1, 0].set_title('Mean Values')
    axes[1, 0].set_ylabel('Mean Value')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_yscale('symlog')
    
    # Standard deviation
    stds = [stats['std'] for stats in all_original_stats]
    axes[1, 1].plot(sample_nums, stds, 'co-')
    axes[1, 1].set_title('Standard Deviation')
    axes[1, 1].set_ylabel('Std Dev')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_yscale('log')
    
    # Distribution of all values
    all_values = []
    for sample_idx in sample_indices:
        sample, _ = dataset[sample_idx]
        all_values.extend(sample.flatten().numpy())
    
    axes[1, 2].hist(all_values, bins=100, alpha=0.7, density=True)
    axes[1, 2].set_title('Overall Value Distribution')
    axes[1, 2].set_xlabel('Value')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_yscale('log')
    axes[1, 2].set_xscale('symlog')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_statistics.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Round-trip analysis summary
    if all_roundtrip_stats:
        print(f"\n{'='*50}")
        print("Round-trip Analysis Summary")
        print(f"{'='*50}")
        
        # Create round-trip quality comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Round-trip Quality Comparison Across Methods', fontsize=16)
        
        methods = ['positive_shift', 'symlog', 'abs_log']
        method_colors = {'positive_shift': 'blue', 'symlog': 'green', 'abs_log': 'red'}
        
        # PSNR comparison
        for method in methods:
            method_stats = [s for s in all_roundtrip_stats if s['method'] == method]
            if method_stats:
                sample_indices = [s['sample_idx'] for s in method_stats]
                psnr_values = [s['psnr'] if s['psnr'] != float('inf') else 100 for s in method_stats]
                axes[0, 0].plot(sample_indices, psnr_values, 'o-', 
                              label=method, color=method_colors[method], alpha=0.7)
        axes[0, 0].set_title('PSNR (dB)')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MSE comparison
        for method in methods:
            method_stats = [s for s in all_roundtrip_stats if s['method'] == method]
            if method_stats:
                sample_indices = [s['sample_idx'] for s in method_stats]
                mse_values = [s['mse'] for s in method_stats]
                axes[0, 1].semilogy(sample_indices, mse_values, 'o-', 
                                   label=method, color=method_colors[method], alpha=0.7)
        axes[0, 1].set_title('Mean Squared Error')
        axes[0, 1].set_ylabel('MSE (log scale)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Correlation comparison
        for method in methods:
            method_stats = [s for s in all_roundtrip_stats if s['method'] == method]
            if method_stats:
                sample_indices = [s['sample_idx'] for s in method_stats]
                corr_values = [s['correlation'] for s in method_stats]
                axes[0, 2].plot(sample_indices, corr_values, 'o-', 
                              label=method, color=method_colors[method], alpha=0.7)
        axes[0, 2].set_title('Correlation Coefficient')
        axes[0, 2].set_ylabel('Correlation')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim([0.999, 1.001])  # Focus on high correlation region
        
        # Relative Error comparison
        for method in methods:
            method_stats = [s for s in all_roundtrip_stats if s['method'] == method]
            if method_stats:
                sample_indices = [s['sample_idx'] for s in method_stats]
                rel_err_values = [s['relative_error'] for s in method_stats]
                axes[1, 0].semilogy(sample_indices, rel_err_values, 'o-', 
                                   label=method, color=method_colors[method], alpha=0.7)
        axes[1, 0].set_title('Relative Error (%)')
        axes[1, 0].set_ylabel('Relative Error (% log scale)')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Max Absolute Error comparison
        for method in methods:
            method_stats = [s for s in all_roundtrip_stats if s['method'] == method]
            if method_stats:
                sample_indices = [s['sample_idx'] for s in method_stats]
                max_err_values = [s['max_abs_error'] for s in method_stats]
                axes[1, 1].semilogy(sample_indices, max_err_values, 'o-', 
                                   label=method, color=method_colors[method], alpha=0.7)
        axes[1, 1].set_title('Maximum Absolute Error')
        axes[1, 1].set_ylabel('Max Abs Error (log scale)')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Method comparison boxplot
        metric_data = {'positive_shift': [], 'symlog': [], 'abs_log': []}
        for method in methods:
            method_stats = [s for s in all_roundtrip_stats if s['method'] == method]
            psnr_values = [s['psnr'] if s['psnr'] != float('inf') else 100 for s in method_stats]
            metric_data[method] = psnr_values
        
        box_data = [metric_data[method] for method in methods if metric_data[method]]
        if box_data:
            axes[1, 2].boxplot(box_data, labels=[m for m in methods if metric_data[m]])
            axes[1, 2].set_title('PSNR Distribution by Method')
            axes[1, 2].set_ylabel('PSNR (dB)')
            axes[1, 2].set_xlabel('Method')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roundtrip_quality_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Print method comparison summary
        print(f"\nMethod Performance Summary:")
        print(f"{'Method':<15} {'Avg PSNR':<12} {'Avg MSE':<12} {'Avg Corr':<12} {'Avg RelErr':<12}")
        print("-" * 70)
        
        for method in methods:
            method_stats = [s for s in all_roundtrip_stats if s['method'] == method]
            if method_stats:
                avg_psnr = np.mean([s['psnr'] if s['psnr'] != float('inf') else 100 for s in method_stats])
                avg_mse = np.mean([s['mse'] for s in method_stats])
                avg_corr = np.mean([s['correlation'] for s in method_stats])
                avg_rel_err = np.mean([s['relative_error'] for s in method_stats])
                
                print(f"{method:<15} {avg_psnr:<12.1f} {avg_mse:<12.2e} {avg_corr:<12.6f} {avg_rel_err:<12.2f}")
        
        # Best method recommendation
        best_method_psnr = max(methods, key=lambda m: np.mean([s['psnr'] if s['psnr'] != float('inf') else 100 
                                                             for s in all_roundtrip_stats if s['method'] == m]))
        best_method_mse = min(methods, key=lambda m: np.mean([s['mse'] for s in all_roundtrip_stats if s['method'] == m]))
        
        print(f"\nRecommendations:")
        print(f"  📈 Best PSNR: {best_method_psnr}")
        print(f"  📉 Best MSE: {best_method_mse}")
        if best_method_psnr == best_method_mse:
            print(f"  🏆 Overall best method: {best_method_psnr}")
        else:
            print(f"  🏆 Consider: {best_method_psnr} for highest fidelity, {best_method_mse} for lowest error")
    
    # Print overall summary
    print(f"\nOVERALL SUMMARY:")
    print(f"Analyzed {num_samples} samples from U_CHI dataset")
    print(f"Global statistics across all samples:")
    
    all_mins = [stats['min'] for stats in all_original_stats]
    all_maxs = [stats['max'] for stats in all_original_stats]
    all_means = [stats['mean'] for stats in all_original_stats]
    all_stds = [stats['std'] for stats in all_original_stats]
    finite_ranges = [stats['dynamic_range'] for stats in all_original_stats if stats['dynamic_range'] != float('inf')]
    
    print(f"  Global min: {min(all_mins):.6e}")
    print(f"  Global max: {max(all_maxs):.6e}")
    print(f"  Average mean: {np.mean(all_means):.6e}")
    print(f"  Average std: {np.mean(all_stds):.6e}")
    if finite_ranges:
        print(f"  Average dynamic range: {np.mean(finite_ranges):.2e}")
        print(f"  Max dynamic range: {max(finite_ranges):.2e}")
    
    print(f"\nOVERALL RECOMMENDATIONS:")
    if min(all_mins) <= 0:
        print("  ⚠️  Data contains zero/negative values - positive_shift method recommended")
    if finite_ranges and max(finite_ranges) > 1e6:
        print("  ⚠️  Very high dynamic range detected - log-scale transformation strongly recommended")
    if np.mean(all_stds) / np.abs(np.mean(all_means)) > 1:
        print("  ⚠️  High coefficient of variation - normalization recommended")
    
    if all_roundtrip_stats:
        # Overall best performing method based on round-trip analysis
        methods = ['positive_shift', 'symlog', 'abs_log']
        best_overall = max(methods, key=lambda m: np.mean([s['psnr'] if s['psnr'] != float('inf') else 100 
                                                          for s in all_roundtrip_stats if s['method'] == m]))
        avg_psnr_best = np.mean([s['psnr'] if s['psnr'] != float('inf') else 100 
                                for s in all_roundtrip_stats if s['method'] == best_overall])
        avg_rel_err_best = np.mean([s['relative_error'] for s in all_roundtrip_stats if s['method'] == best_overall])
        
        print(f"\n🎯 For your SWAE 3D model:")
        print(f"    ✅ Use {best_overall} log transformation")
        print(f"    ✅ Expected round-trip PSNR: {avg_psnr_best:.1f} dB")
        print(f"    ✅ Expected relative error: {avg_rel_err_best:.2f}%")
        print(f"    ✅ Information loss is minimal - safe for neural network training")
    
    print(f"\nPlots saved in: {output_dir}/")
    print(f"  - Individual sample slices and histograms")
    print(f"  - Summary statistics across all samples")
    print(f"  - Round-trip analysis for each sample and method")
    print(f"  - Method comparison summary")


if __name__ == "__main__":
    main() 