#!/usr/bin/env python
"""
Analyze distribution of all variables (except U_CHI) from BSSN dataset
and apply poslog transformation to show their distributions.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
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

def load_all_variables(data_folder, exclude_vars=['U_CHI']):
    """Load all variables from HDF5 files (except excluded ones)."""
    file_pattern = os.path.join(data_folder, "bssn_gr_*_extracted.hdf5")
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        raise ValueError(f"No HDF5 files found in {data_folder}")
    
    print(f"Found {len(files)} HDF5 files")
    
    # First, get the list of variables from the first file
    with h5py.File(files[0], 'r') as f:
        all_vars = list(f['vars'][:].astype(str))
        vars_to_load = [var for var in all_vars if var not in exclude_vars]
        print(f"\nAvailable variables: {all_vars}")
        print(f"Variables to analyze: {vars_to_load}")
    
    # Initialize dictionary to store data for each variable
    variable_data = {var: [] for var in vars_to_load}
    
    # Load data from all files
    for file_path in tqdm(files, desc="Loading files"):
        with h5py.File(file_path, 'r') as f:
            vars_list = f['vars'][:].astype(str)
            var_data = f['var_data'][:]  # Shape: (n_vars, n_blocks, 7, 7, 7)
            
            for var in vars_to_load:
                if var in vars_list:
                    var_idx = list(vars_list).index(var)
                    # Extract 5x5x5 center crop from 7x7x7 data
                    data_7x7x7 = var_data[var_idx]  # Shape: (n_blocks, 7, 7, 7)
                    data_5x5x5 = data_7x7x7[:, 1:6, 1:6, 1:6]  # Center crop
                    variable_data[var].append(data_5x5x5)
    
    # Concatenate all data for each variable
    for var in vars_to_load:
        if variable_data[var]:
            variable_data[var] = np.concatenate(variable_data[var], axis=0)
            print(f"{var}: loaded {variable_data[var].shape[0]} blocks")
        else:
            print(f"Warning: No data found for {var}")
    
    return variable_data

def plot_distributions(variable_data, output_dir="variable_distributions"):
    """Plot distributions for all variables (original and poslog transformed)."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary plot with all variables
    n_vars = len(variable_data)
    fig, axes = plt.subplots(n_vars, 2, figsize=(15, 5*n_vars))
    if n_vars == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (var_name, data) in enumerate(variable_data.items()):
        if data.size == 0:
            continue
            
        # Flatten the data for histogram
        data_flat = data.flatten()
        
        # Original distribution
        ax1 = axes[idx, 0]
        hist1, bins1, _ = ax1.hist(data_flat, bins=100, alpha=0.7, density=True, color='blue')
        ax1.set_title(f'{var_name} - Original Distribution')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'Mean: {np.mean(data_flat):.4f}\nStd: {np.std(data_flat):.4f}\nMin: {np.min(data_flat):.4f}\nMax: {np.max(data_flat):.4f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Poslog transformed distribution
        data_poslog = poslog_transform(data_flat)
        ax2 = axes[idx, 1]
        hist2, bins2, _ = ax2.hist(data_poslog, bins=100, alpha=0.7, density=True, color='green')
        ax2.set_title(f'{var_name} - Poslog Transformed Distribution')
        ax2.set_xlabel('Log Value')
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics for poslog
        stats_text_poslog = f'Mean: {np.mean(data_poslog):.4f}\nStd: {np.std(data_poslog):.4f}\nMin: {np.min(data_poslog):.4f}\nMax: {np.max(data_poslog):.4f}'
        ax2.text(0.02, 0.98, stats_text_poslog, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Save individual plots
        fig_individual = plt.figure(figsize=(12, 5))
        
        # Original
        plt.subplot(1, 2, 1)
        plt.hist(data_flat, bins=100, alpha=0.7, density=True, color='blue')
        plt.title(f'{var_name} - Original Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Poslog
        plt.subplot(1, 2, 2)
        plt.hist(data_poslog, bins=100, alpha=0.7, density=True, color='green')
        plt.title(f'{var_name} - Poslog Transformed Distribution')
        plt.xlabel('Log Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.text(0.02, 0.98, stats_text_poslog, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{var_name}_distribution.png'), dpi=150)
        plt.close()
        
        print(f"\n{var_name}:")
        print(f"  Original - Mean: {np.mean(data_flat):.4f}, Std: {np.std(data_flat):.4f}, Range: [{np.min(data_flat):.4f}, {np.max(data_flat):.4f}]")
        print(f"  Poslog   - Mean: {np.mean(data_poslog):.4f}, Std: {np.std(data_poslog):.4f}, Range: [{np.min(data_poslog):.4f}, {np.max(data_poslog):.4f}]")
    
    # Save combined plot
    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_variables_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")

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
    
    # Load all variables except U_CHI
    variable_data = load_all_variables(data_folder, exclude_vars=['U_CHI'])
    
    # Plot distributions
    plot_distributions(variable_data)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()