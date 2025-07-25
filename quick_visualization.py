#!/usr/bin/env python3
"""
Quick visualization script for key variables
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main visualization
from visualize_dual_reconstruction import load_data, plot_variable_comparison, create_summary_plot

def main():
    # Paths
    original_file = "/u/tawal/BSSN-Extracted-Data/tt_q08/bssn_gr_11200_extracted.hdf5"
    reconstructed_file = "./reconstructed/bssn_gr_11200_dual_reconstructed.hdf5"
    output_dir = "./quick_plots"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    var_names, original_data, reconstructed_data, metrics = load_data(
        original_file, reconstructed_file
    )
    
    # Key variables to visualize
    key_vars = ['U_CHI', 'U_ALPHA', 'U_B2', 'U_SYMAT2', 'U_GT2', 'U_SYMAT4']
    
    # Create plots
    print("\nCreating quick plots for key variables...")
    for var_name in key_vars:
        if var_name in var_names:
            var_idx = var_names.index(var_name)
            print(f"  Plotting {var_name}...")
            
            plot_variable_comparison(
                var_name,
                original_data[var_idx],
                reconstructed_data[var_idx],
                metrics,
                output_dir,
                sample_idx=10,  # Just one sample
                slice_idx=3
            )
    
    # Create summary plot
    print("\nCreating summary plot...")
    create_summary_plot(var_names, metrics, output_dir)
    
    print(f"\nQuick plots saved to: {output_dir}")


if __name__ == "__main__":
    main()