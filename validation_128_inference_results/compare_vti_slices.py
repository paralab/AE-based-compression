#!/usr/bin/env python3
"""
VTI File Comparison Script
Compares original and reconstructed VTI files by taking slices from different axes
at 1/2, 1/3, and 3/4 positions and plotting them side by side.
"""

import vtk
from vtk.util import numpy_support
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os

def read_vti_file(filename):
    """Read a VTI file and return the data array."""
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    
    # Get the data
    image_data = reader.GetOutput()
    dims = image_data.GetDimensions()
    
    # Convert to numpy array
    vtk_array = image_data.GetPointData().GetScalars()
    numpy_array = numpy_support.vtk_to_numpy(vtk_array)
    
    # Reshape to 3D array (VTK uses Fortran ordering)
    data = numpy_array.reshape(dims, order='F')
    
    return data, dims

def get_slice_positions(dim_size):
    """Get slice positions at 1/3, 1/2, and 3/4 of the dimension."""
    return [
        int(dim_size / 3),      # 1/3
        int(dim_size / 2),      # 1/2
        int(dim_size * 3 / 4)   # 3/4
    ]

def create_comparison_plots():
    """Create comprehensive comparison plots between original and reconstructed VTI files."""
    
    # File paths
    original_file = "sample_009_128x128x128_original.vti"
    reconstructed_file = "sample_009_128x128x128_reconstructed.vti"
    
    # Read the VTI files
    print("Reading VTI files...")
    original_data, original_dims = read_vti_file(original_file)
    reconstructed_data, reconstructed_dims = read_vti_file(reconstructed_file)
    
    print(f"Original dimensions: {original_dims}")
    print(f"Reconstructed dimensions: {reconstructed_dims}")
    print(f"Original data range: [{original_data.min():.6f}, {original_data.max():.6f}]")
    print(f"Reconstructed data range: [{reconstructed_data.min():.6f}, {reconstructed_data.max():.6f}]")
    
    # Calculate error
    error_data = np.abs(original_data - reconstructed_data)
    mse = np.mean((original_data - reconstructed_data)**2)
    mae = np.mean(error_data)
    max_error = np.max(error_data)
    
    print(f"\nError Statistics:")
    print(f"MSE: {mse:.8f}")
    print(f"MAE: {mae:.8f}")
    print(f"Max Error: {max_error:.8f}")
    print(f"PSNR: {20 * np.log10(np.max(np.abs(original_data)) / np.sqrt(mse)):.2f} dB")
    
    # Get slice positions for each axis
    x_positions = get_slice_positions(original_dims[0])
    y_positions = get_slice_positions(original_dims[1])
    z_positions = get_slice_positions(original_dims[2])
    
    # Create the main comparison figure
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 9, figure=fig, hspace=0.3, wspace=0.3)
    
    # Define colormap and normalization
    vmin = min(original_data.min(), reconstructed_data.min())
    vmax = max(original_data.max(), reconstructed_data.max())
    
    axes_names = ['X-axis', 'Y-axis', 'Z-axis']
    position_labels = ['1/3', '1/2', '3/4']
    
    # Plot slices for each axis
    for axis_idx, (axis_name, positions) in enumerate(zip(axes_names, [x_positions, y_positions, z_positions])):
        for pos_idx, pos in enumerate(positions):
            # Extract slices
            if axis_idx == 0:  # X-axis slices (YZ planes)
                original_slice = original_data[pos, :, :]
                reconstructed_slice = reconstructed_data[pos, :, :]
                error_slice = error_data[pos, :, :]
            elif axis_idx == 1:  # Y-axis slices (XZ planes)
                original_slice = original_data[:, pos, :]
                reconstructed_slice = reconstructed_data[:, pos, :]
                error_slice = error_data[:, pos, :]
            else:  # Z-axis slices (XY planes)
                original_slice = original_data[:, :, pos]
                reconstructed_slice = reconstructed_data[:, :, pos]
                error_slice = error_data[:, :, pos]
            
            # Original slice
            ax_orig = fig.add_subplot(gs[axis_idx, pos_idx*3])
            im_orig = ax_orig.imshow(original_slice.T, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
            ax_orig.set_title(f'{axis_name} - {position_labels[pos_idx]}\nOriginal', fontsize=10)
            ax_orig.set_xlabel('X' if axis_idx != 0 else 'Y')
            ax_orig.set_ylabel('Z' if axis_idx != 2 else 'Y')
            
            # Reconstructed slice
            ax_recon = fig.add_subplot(gs[axis_idx, pos_idx*3 + 1])
            im_recon = ax_recon.imshow(reconstructed_slice.T, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
            ax_recon.set_title(f'Reconstructed', fontsize=10)
            ax_recon.set_xlabel('X' if axis_idx != 0 else 'Y')
            ax_recon.set_ylabel('Z' if axis_idx != 2 else 'Y')
            
            # Error slice
            ax_error = fig.add_subplot(gs[axis_idx, pos_idx*3 + 2])
            im_error = ax_error.imshow(error_slice.T, cmap='hot', vmin=0, vmax=error_slice.max(), origin='lower')
            ax_error.set_title(f'Error\n(Max: {error_slice.max():.6f})', fontsize=10)
            ax_error.set_xlabel('X' if axis_idx != 0 else 'Y')
            ax_error.set_ylabel('Z' if axis_idx != 2 else 'Y')
            
            # Add colorbars for the first row
            if axis_idx == 0:
                if pos_idx == 0:
                    cbar_orig = plt.colorbar(im_orig, ax=ax_orig, shrink=0.8)
                    cbar_orig.set_label('Value', fontsize=8)
                elif pos_idx == 1:
                    cbar_recon = plt.colorbar(im_recon, ax=ax_recon, shrink=0.8)
                    cbar_recon.set_label('Value', fontsize=8)
                elif pos_idx == 2:
                    cbar_error = plt.colorbar(im_error, ax=ax_error, shrink=0.8)
                    cbar_error.set_label('Error', fontsize=8)
    
    # Add main title
    fig.suptitle(f'VTI Comparison: Original vs Reconstructed\n'
                 f'MSE: {mse:.8f}, MAE: {mae:.8f}, Max Error: {max_error:.8f}', 
                 fontsize=16, y=0.95)
    
    # Save the main comparison plot
    plt.savefig('sample_009_128x128x128_comparison_slices.png', dpi=300, bbox_inches='tight')
    plt.savefig('sample_009_128x128x128_comparison_slices.pdf', bbox_inches='tight')
    print(f"\nSaved main comparison plot: sample_009_128x128x128_comparison_slices.png")
    
    # Create individual detailed plots for each axis
    for axis_idx, (axis_name, positions) in enumerate(zip(axes_names, [x_positions, y_positions, z_positions])):
        fig_detail, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig_detail.suptitle(f'Detailed {axis_name} Slices Comparison', fontsize=16)
        
        for pos_idx, pos in enumerate(positions):
            # Extract slices
            if axis_idx == 0:  # X-axis slices (YZ planes)
                original_slice = original_data[pos, :, :]
                reconstructed_slice = reconstructed_data[pos, :, :]
                error_slice = error_data[pos, :, :]
                plane_name = f'YZ plane at X={pos}'
            elif axis_idx == 1:  # Y-axis slices (XZ planes)
                original_slice = original_data[:, pos, :]
                reconstructed_slice = reconstructed_data[:, pos, :]
                error_slice = error_data[:, pos, :]
                plane_name = f'XZ plane at Y={pos}'
            else:  # Z-axis slices (XY planes)
                original_slice = original_data[:, :, pos]
                reconstructed_slice = reconstructed_data[:, :, pos]
                error_slice = error_data[:, :, pos]
                plane_name = f'XY plane at Z={pos}'
            
            # Original
            im1 = axes[pos_idx, 0].imshow(original_slice.T, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
            axes[pos_idx, 0].set_title(f'{plane_name}\nOriginal')
            axes[pos_idx, 0].set_xlabel('X' if axis_idx != 0 else 'Y')
            axes[pos_idx, 0].set_ylabel('Z' if axis_idx != 2 else 'Y')
            
            # Reconstructed
            im2 = axes[pos_idx, 1].imshow(reconstructed_slice.T, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
            axes[pos_idx, 1].set_title(f'Reconstructed')
            axes[pos_idx, 1].set_xlabel('X' if axis_idx != 0 else 'Y')
            axes[pos_idx, 1].set_ylabel('Z' if axis_idx != 2 else 'Y')
            
            # Error
            im3 = axes[pos_idx, 2].imshow(error_slice.T, cmap='hot', vmin=0, vmax=error_slice.max(), origin='lower')
            axes[pos_idx, 2].set_title(f'Absolute Error\nMax: {error_slice.max():.6f}')
            axes[pos_idx, 2].set_xlabel('X' if axis_idx != 0 else 'Y')
            axes[pos_idx, 2].set_ylabel('Z' if axis_idx != 2 else 'Y')
            
            # Add colorbars
            plt.colorbar(im1, ax=axes[pos_idx, 0], shrink=0.8)
            plt.colorbar(im2, ax=axes[pos_idx, 1], shrink=0.8)
            plt.colorbar(im3, ax=axes[pos_idx, 2], shrink=0.8)
        
        plt.tight_layout()
        filename = f'sample_009_128x128x128_comparison_{axis_name.lower().replace("-", "_")}_detailed.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved detailed {axis_name} plot: {filename}")
        plt.close()
    
    # Create a 1D profile comparison plot
    fig_profile, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig_profile.suptitle('1D Profile Comparisons', fontsize=16)
    
    # Central line profiles
    center_x, center_y, center_z = original_dims[0]//2, original_dims[1]//2, original_dims[2]//2
    
    # X-direction profile (through center)
    x_profile_orig = original_data[:, center_y, center_z]
    x_profile_recon = reconstructed_data[:, center_y, center_z]
    axes[0, 0].plot(x_profile_orig, 'b-', label='Original', linewidth=2)
    axes[0, 0].plot(x_profile_recon, 'r--', label='Reconstructed', linewidth=2)
    axes[0, 0].set_title('X-direction Profile (through center)')
    axes[0, 0].set_xlabel('X index')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Y-direction profile (through center)
    y_profile_orig = original_data[center_x, :, center_z]
    y_profile_recon = reconstructed_data[center_x, :, center_z]
    axes[0, 1].plot(y_profile_orig, 'b-', label='Original', linewidth=2)
    axes[0, 1].plot(y_profile_recon, 'r--', label='Reconstructed', linewidth=2)
    axes[0, 1].set_title('Y-direction Profile (through center)')
    axes[0, 1].set_xlabel('Y index')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Z-direction profile (through center)
    z_profile_orig = original_data[center_x, center_y, :]
    z_profile_recon = reconstructed_data[center_x, center_y, :]
    axes[1, 0].plot(z_profile_orig, 'b-', label='Original', linewidth=2)
    axes[1, 0].plot(z_profile_recon, 'r--', label='Reconstructed', linewidth=2)
    axes[1, 0].set_title('Z-direction Profile (through center)')
    axes[1, 0].set_xlabel('Z index')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error histogram
    axes[1, 1].hist(error_data.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(mae, color='blue', linestyle='--', linewidth=2, label=f'MAE: {mae:.6f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('sample_009_128x128x128_comparison_profiles.png', dpi=300, bbox_inches='tight')
    print("Saved profile comparison plot: sample_009_128x128x128_comparison_profiles.png")
    plt.close()
    
    # Create summary statistics
    print(f"\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Original data shape: {original_data.shape}")
    print(f"Reconstructed data shape: {reconstructed_data.shape}")
    print(f"Original data range: [{original_data.min():.6f}, {original_data.max():.6f}]")
    print(f"Reconstructed data range: [{reconstructed_data.min():.6f}, {reconstructed_data.max():.6f}]")
    print(f"Mean Squared Error (MSE): {mse:.8f}")
    print(f"Mean Absolute Error (MAE): {mae:.8f}")
    print(f"Maximum Absolute Error: {max_error:.8f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.8f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {20 * np.log10(np.max(np.abs(original_data)) / np.sqrt(mse)):.2f} dB")
    print(f"Structural Similarity (correlation): {np.corrcoef(original_data.flatten(), reconstructed_data.flatten())[0,1]:.6f}")
    
    # Save statistics to file
    with open('sample_009_128x128x128_comparison_statistics.txt', 'w') as f:
        f.write("VTI Comparison Statistics\n")
        f.write("="*50 + "\n")
        f.write(f"Original data shape: {original_data.shape}\n")
        f.write(f"Reconstructed data shape: {reconstructed_data.shape}\n")
        f.write(f"Original data range: [{original_data.min():.6f}, {original_data.max():.6f}]\n")
        f.write(f"Reconstructed data range: [{reconstructed_data.min():.6f}, {reconstructed_data.max():.6f}]\n")
        f.write(f"Mean Squared Error (MSE): {mse:.8f}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.8f}\n")
        f.write(f"Maximum Absolute Error: {max_error:.8f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.8f}\n")
        f.write(f"Peak Signal-to-Noise Ratio (PSNR): {20 * np.log10(np.max(np.abs(original_data)) / np.sqrt(mse)):.2f} dB\n")
        f.write(f"Structural Similarity (correlation): {np.corrcoef(original_data.flatten(), reconstructed_data.flatten())[0,1]:.6f}\n")
    
    print("Saved statistics to: sample_009_128x128x128_comparison_statistics.txt")
    print("\nAll plots and statistics have been generated successfully!")

if __name__ == "__main__":
    create_comparison_plots() 