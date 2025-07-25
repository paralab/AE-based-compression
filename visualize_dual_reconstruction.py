#!/usr/bin/env python3
"""
Visualization script for dual model reconstruction results
Creates plots comparing original vs reconstructed data with metrics
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define problematic variables
PROBLEMATIC_VARS = ['U_B2', 'U_SYMAT2', 'U_GT2', 'U_SYMGT2', 'U_SYMAT4', 'U_SYMAT3']


def load_data(original_file, reconstructed_file):
    """Load original and reconstructed data"""
    with h5py.File(original_file, 'r') as f_orig:
        var_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                     for name in f_orig['vars'][:]]
        original_data = f_orig['var_data'][:]
    
    with h5py.File(reconstructed_file, 'r') as f_recon:
        reconstructed_data = f_recon['var_data'][:]
        
        # Load metrics if available
        metrics = {}
        if 'metrics' in f_recon:
            metrics_group = f_recon['metrics']
            for var_name in metrics_group.keys():
                var_metrics = {}
                for metric_name in metrics_group[var_name].attrs.keys():
                    var_metrics[metric_name] = metrics_group[var_name].attrs[metric_name]
                metrics[var_name] = var_metrics
    
    return var_names, original_data, reconstructed_data, metrics


def plot_variable_comparison(var_name, original, reconstructed, metrics, save_dir, 
                           sample_idx=0, slice_idx=3):
    """Create comparison plot for a single variable"""
    
    # Get model type
    model_type = "asinh" if var_name in PROBLEMATIC_VARS else "poslog"
    
    # Create figure with gridspec for better layout control
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 1])
    
    # Get sample data
    orig_sample = original[sample_idx]  # Shape: (7, 7, 7)
    recon_sample = reconstructed[sample_idx]
    diff = orig_sample - recon_sample
    
    # Calculate value ranges for consistent colorbars
    vmin = min(orig_sample.min(), recon_sample.min())
    vmax = max(orig_sample.max(), recon_sample.max())
    diff_max = np.abs(diff).max()
    
    # Plot original - center slice
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(orig_sample[slice_idx, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Original (z={slice_idx})', fontsize=12)
    ax1.set_xlabel('y')
    ax1.set_ylabel('x')
    # Add 5x5x5 region indicator
    rect = Rectangle((0.5, 0.5), 5, 5, linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    
    # Plot reconstructed - center slice
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(recon_sample[slice_idx, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Reconstructed (z={slice_idx})', fontsize=12)
    ax2.set_xlabel('y')
    ax2.set_ylabel('x')
    rect = Rectangle((0.5, 0.5), 5, 5, linewidth=2, edgecolor='red', facecolor='none')
    ax2.add_patch(rect)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)
    
    # Plot difference - center slice
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(diff[slice_idx, :, :], cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    ax3.set_title(f'Difference (z={slice_idx})', fontsize=12)
    ax3.set_xlabel('y')
    ax3.set_ylabel('x')
    rect = Rectangle((0.5, 0.5), 5, 5, linewidth=2, edgecolor='red', facecolor='none')
    ax3.add_patch(rect)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax)
    
    # Plot y-z cross section
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(orig_sample[:, slice_idx, :], cmap='viridis', vmin=vmin, vmax=vmax)
    ax4.set_title(f'Original (x={slice_idx})', fontsize=12)
    ax4.set_xlabel('z')
    ax4.set_ylabel('y')
    rect = Rectangle((0.5, 0.5), 5, 5, linewidth=2, edgecolor='red', facecolor='none')
    ax4.add_patch(rect)
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im4, cax=cax)
    
    # Plot reconstructed y-z cross section
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(recon_sample[:, slice_idx, :], cmap='viridis', vmin=vmin, vmax=vmax)
    ax5.set_title(f'Reconstructed (x={slice_idx})', fontsize=12)
    ax5.set_xlabel('z')
    ax5.set_ylabel('y')
    rect = Rectangle((0.5, 0.5), 5, 5, linewidth=2, edgecolor='red', facecolor='none')
    ax5.add_patch(rect)
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im5, cax=cax)
    
    # Plot histogram comparison
    ax6 = fig.add_subplot(gs[1, 2])
    # Focus on 5x5x5 center region for histogram
    orig_center = orig_sample[1:6, 1:6, 1:6].flatten()
    recon_center = recon_sample[1:6, 1:6, 1:6].flatten()
    
    bins = np.linspace(min(orig_center.min(), recon_center.min()), 
                      max(orig_center.max(), recon_center.max()), 50)
    ax6.hist(orig_center, bins=bins, alpha=0.5, label='Original', density=True)
    ax6.hist(recon_center, bins=bins, alpha=0.5, label='Reconstructed', density=True)
    ax6.set_xlabel('Value')
    ax6.set_ylabel('Density')
    ax6.set_title('Value Distribution (5x5x5 center)', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Metrics text box
    ax_metrics = fig.add_subplot(gs[2, :])
    ax_metrics.axis('off')
    
    # Prepare metrics text
    metrics_text = f"Variable: {var_name} | Model: {model_type.upper()}\n"
    metrics_text += f"Sample Index: {sample_idx}\n\n"
    
    if var_name in metrics:
        m = metrics[var_name]
        metrics_text += f"PSNR: {m.get('psnr', 'N/A'):.2f} dB\n"
        metrics_text += f"MSE: {m.get('mse', 'N/A'):.6f}\n"
        metrics_text += f"MAE: {m.get('mae', 'N/A'):.6f}\n"
        metrics_text += f"Relative Error: {m.get('rel_error', 'N/A'):.2%}\n"
    else:
        # Calculate metrics on the fly
        mse = np.mean((orig_sample - recon_sample) ** 2)
        mae = np.mean(np.abs(orig_sample - recon_sample))
        value_range = orig_sample.max() - orig_sample.min()
        if value_range > 0 and mse > 0:
            psnr = 20 * np.log10(value_range) - 10 * np.log10(mse)
        else:
            psnr = float('inf')
        
        metrics_text += f"PSNR: {psnr:.2f} dB\n"
        metrics_text += f"MSE: {mse:.6f}\n"
        metrics_text += f"MAE: {mae:.6f}\n"
    
    # Add data statistics
    metrics_text += f"\nOriginal - Min: {orig_sample.min():.6f}, Max: {orig_sample.max():.6f}, "
    metrics_text += f"Mean: {orig_sample.mean():.6f}, Std: {orig_sample.std():.6f}\n"
    metrics_text += f"Reconstructed - Min: {recon_sample.min():.6f}, Max: {recon_sample.max():.6f}, "
    metrics_text += f"Mean: {recon_sample.mean():.6f}, Std: {recon_sample.std():.6f}"
    
    ax_metrics.text(0.1, 0.5, metrics_text, fontsize=11, 
                   transform=ax_metrics.transAxes, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle(f'{var_name} Reconstruction Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, f'{var_name}_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def create_summary_plot(var_names, metrics, save_dir):
    """Create summary plot showing all variables' performance"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Separate variables by model type
    poslog_vars = [v for v in var_names if v.startswith('U_') and v not in PROBLEMATIC_VARS]
    asinh_vars = [v for v in var_names if v in PROBLEMATIC_VARS]
    
    # Prepare data for plotting
    all_vars = []
    psnr_values = []
    mse_values = []
    colors = []
    
    for var in poslog_vars:
        if var in metrics:
            all_vars.append(var)
            psnr = metrics[var].get('psnr', 0)
            psnr_values.append(psnr if np.isfinite(psnr) else 100)  # Cap inf at 100
            mse_values.append(metrics[var].get('mse', 0))
            colors.append('blue')
    
    for var in asinh_vars:
        if var in metrics:
            all_vars.append(var)
            psnr = metrics[var].get('psnr', 0)
            psnr_values.append(psnr if np.isfinite(psnr) else 100)  # Cap inf at 100
            mse_values.append(metrics[var].get('mse', 0))
            colors.append('red')
    
    # PSNR plot
    x_pos = np.arange(len(all_vars))
    bars1 = ax1.bar(x_pos, psnr_values, color=colors, alpha=0.7)
    ax1.set_xlabel('Variable')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR by Variable (Blue: Poslog Model, Red: Asinh Model)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(all_vars, rotation=45, ha='right')
    ax1.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Good Quality (30 dB)')
    ax1.axhline(y=40, color='darkgreen', linestyle='--', alpha=0.5, label='Excellent Quality (40 dB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, psnr_values)):
        if val >= 100:
            label = 'inf'
        else:
            label = f'{val:.1f}'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                label, ha='center', va='bottom', fontsize=8, rotation=45)
    
    # MSE plot (log scale)
    bars2 = ax2.bar(x_pos, mse_values, color=colors, alpha=0.7)
    ax2.set_xlabel('Variable')
    ax2.set_ylabel('MSE (log scale)')
    ax2.set_title('MSE by Variable (Blue: Poslog Model, Red: Asinh Model)')
    ax2.set_yscale('log')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(all_vars, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Dual Model Reconstruction Performance Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'performance_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Visualize dual model reconstruction results')
    parser.add_argument('--original-file', type=str, required=True,
                        help='Path to original HDF5 file')
    parser.add_argument('--reconstructed-file', type=str, required=True,
                        help='Path to reconstructed HDF5 file')
    parser.add_argument('--output-dir', type=str, default='./reconstruction_plots',
                        help='Directory to save plots')
    parser.add_argument('--variables', type=str, nargs='+', default=None,
                        help='Specific variables to plot (default: all U_ variables)')
    parser.add_argument('--sample-indices', type=int, nargs='+', default=[0, 10, 50],
                        help='Sample indices to visualize')
    parser.add_argument('--slice-index', type=int, default=3,
                        help='Slice index for 2D visualization (default: 3, center)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    var_names, original_data, reconstructed_data, metrics = load_data(
        args.original_file, args.reconstructed_file
    )
    
    # Filter to U_ variables
    u_var_indices = [(i, name) for i, name in enumerate(var_names) if name.startswith('U_')]
    
    # Filter to specific variables if requested
    if args.variables:
        u_var_indices = [(i, name) for i, name in u_var_indices if name in args.variables]
    
    print(f"Found {len(u_var_indices)} variables to plot")
    
    # Create individual plots
    print("\nCreating individual variable plots...")
    for var_idx, var_name in u_var_indices:
        print(f"  Plotting {var_name}...")
        
        for sample_idx in args.sample_indices:
            if sample_idx >= original_data.shape[1]:
                continue
                
            plot_path = plot_variable_comparison(
                var_name,
                original_data[var_idx],
                reconstructed_data[var_idx],
                metrics,
                args.output_dir,
                sample_idx=sample_idx,
                slice_idx=args.slice_index
            )
            
            # Rename file to include sample index
            new_path = plot_path.replace('.png', f'_sample{sample_idx}.png')
            os.rename(plot_path, new_path)
    
    # Create summary plot
    print("\nCreating summary plot...")
    summary_path = create_summary_plot(var_names, metrics, args.output_dir)
    
    print(f"\nPlots saved to: {args.output_dir}")
    print(f"Summary plot: {summary_path}")
    
    # Print which variables used which model
    print("\nVariable assignments:")
    print("Poslog model:", [name for _, name in u_var_indices if name not in PROBLEMATIC_VARS])
    print("Asinh model:", [name for _, name in u_var_indices if name in PROBLEMATIC_VARS])


if __name__ == "__main__":
    main()