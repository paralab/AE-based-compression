#!/usr/bin/env python3
"""
SWAE U_CHI Validation Inference Script
Evaluates trained SWAE model on validation set of U_CHI data
"""

import os
import sys
import argparse
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# VTK imports (optional)
try:
    import vtk
    from vtk.util import numpy_support
    VTK_AVAILABLE = True
except ImportError:
    print("Warning: VTK not available, VTI files will not be saved")
    VTK_AVAILABLE = False

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.swae_pure_3d_7x7x7 import create_swae_3d_7x7x7_model
from datasets.u_chi_dataset import create_u_chi_datasets


def calculate_metrics(original, reconstructed):
    """Calculate reconstruction quality metrics"""
    # MSE
    mse = np.mean((original - reconstructed) ** 2)
    
    # PSNR
    value_range = np.max(original) - np.min(original)
    if value_range == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(value_range) - 10 * np.log10(mse + 1e-8)
    
    # MAE
    mae = np.mean(np.abs(original - reconstructed))
    
    # Correlation coefficient
    flat_orig = original.flatten()
    flat_recon = reconstructed.flatten()
    correlation = np.corrcoef(flat_orig, flat_recon)[0, 1]
    
    return {
        'mse': mse,
        'psnr': psnr,
        'mae': mae,
        'correlation': correlation,
        'min_error': np.min(original - reconstructed),
        'max_error': np.max(original - reconstructed),
        'mean_error': np.mean(original - reconstructed),
        'std_error': np.std(original - reconstructed)
    }


def save_vti_file(data, filename, spacing=(1.0, 1.0, 1.0)):
    """Save 3D numpy array as VTI file"""
    if not VTK_AVAILABLE:
        print(f"VTK not available, skipping VTI file: {filename}")
        return
    
    # Create VTK image data
    image_data = vtk.vtkImageData()
    dims = data.shape
    image_data.SetDimensions(dims[0], dims[1], dims[2])
    image_data.SetSpacing(spacing)
    image_data.SetOrigin(0.0, 0.0, 0.0)
    
    # Convert numpy array to VTK array
    vtk_array = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_array.SetName("U_CHI_Normalized_LogPos")
    
    # Add array to image data
    image_data.GetPointData().SetScalars(vtk_array)
    
    # Write VTI file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()
    print(f"Saved VTI file: {filename}")


def plot_comparison_slices(original, reconstructed, error, output_dir, sample_idx, scale_type="normalized"):
    """Plot comparison slices from different axes at 1/3 and 1/2 positions"""
    # Create figure with subplots
    fig, axes = plt.subplots(6, 3, figsize=(15, 24))
    
    # Set title based on scale type
    if scale_type == "normalized":
        scale_label = "Normalized Log Pos Scale"
        filename_suffix = "normalized"
    else:
        scale_label = "Denormalized Original Scale"
        filename_suffix = "denormalized"
    
    fig.suptitle(f'Sample {sample_idx:03d} - Reconstruction Comparison ({scale_label})\n'
                 f'Original: [{original.min():.3e}, {original.max():.3e}], '
                 f'Reconstructed: [{reconstructed.min():.3e}, {reconstructed.max():.3e}]', 
                 fontsize=14)
    
    # Get dimensions
    depth, height, width = original.shape
    
    # Define slice positions (1/3 and 1/2)
    slice_positions = {
        'z': [depth // 3, depth // 2],      # Z-axis (depth)
        'y': [height // 3, height // 2],    # Y-axis (height)  
        'x': [width // 3, width // 2]       # X-axis (width)
    }
    
    plot_row = 0
    
    # Calculate global min/max for consistent colormap scaling
    global_min = min(original.min(), reconstructed.min())
    global_max = max(original.max(), reconstructed.max())
    
    # Calculate error range for symmetric colormap
    error_max = max(abs(error.min()), abs(error.max()))
    
    for axis_name, positions in slice_positions.items():
        for pos in positions:
            # Extract slices based on axis
            if axis_name == 'z':
                orig_slice = original[pos, :, :]
                recon_slice = reconstructed[pos, :, :]
                error_slice = error[pos, :, :]
                title_suffix = f"Z={pos}/{depth}"
            elif axis_name == 'y':
                orig_slice = original[:, pos, :]
                recon_slice = reconstructed[:, pos, :]
                error_slice = error[:, pos, :]
                title_suffix = f"Y={pos}/{height}"
            else:  # x
                orig_slice = original[:, :, pos]
                recon_slice = reconstructed[:, :, pos]
                error_slice = error[:, :, pos]
                title_suffix = f"X={pos}/{width}"
            
            # Plot original (consistent color scaling)
            im1 = axes[plot_row, 0].imshow(orig_slice, cmap='viridis', aspect='equal',
                                         vmin=global_min, vmax=global_max)
            axes[plot_row, 0].set_title(f'Original - {title_suffix}')
            axes[plot_row, 0].axis('off')
            plt.colorbar(im1, ax=axes[plot_row, 0], shrink=0.6)
            
            # Plot reconstructed (consistent color scaling)
            im2 = axes[plot_row, 1].imshow(recon_slice, cmap='viridis', aspect='equal',
                                         vmin=global_min, vmax=global_max)
            axes[plot_row, 1].set_title(f'Reconstructed - {title_suffix}')
            axes[plot_row, 1].axis('off')
            plt.colorbar(im2, ax=axes[plot_row, 1], shrink=0.6)
            
            # Plot error (symmetric colormap)
            im3 = axes[plot_row, 2].imshow(error_slice, cmap='RdBu_r', aspect='equal',
                                         vmin=-error_max, vmax=error_max)
            axes[plot_row, 2].set_title(f'Error - {title_suffix}')
            axes[plot_row, 2].axis('off')
            plt.colorbar(im3, ax=axes[plot_row, 2], shrink=0.6)
            
            plot_row += 1
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = os.path.join(output_dir, f'sample_{sample_idx:03d}_comparison_slices_{filename_suffix}.png')
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {plot_filename}")


def save_validation_results(original_norm, reconstructed_norm, original_denorm, reconstructed_denorm, 
                           output_dir, sample_idx, save_vti=False):
    """Save validation results as VTI files and comparison plots (no HDF5)"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate errors for both scales
    error_norm = original_norm - reconstructed_norm
    error_denorm = original_denorm - reconstructed_denorm
    
    # Always generate both comparison plots
    plot_comparison_slices(original_norm, reconstructed_norm, error_norm, 
                          output_dir, sample_idx, "normalized")
    plot_comparison_slices(original_denorm, reconstructed_denorm, error_denorm, 
                          output_dir, sample_idx, "denormalized")
    
    # Save VTI files if requested (in normalized scale)
    if save_vti:
        vti_dir = os.path.join(output_dir, 'vti_files')
        os.makedirs(vti_dir, exist_ok=True)
        
        # Save VTI files in normalized (log pos) scale
        save_vti_file(original_norm, os.path.join(vti_dir, f'sample_{sample_idx:03d}_original_normalized.vti'))
        save_vti_file(reconstructed_norm, os.path.join(vti_dir, f'sample_{sample_idx:03d}_reconstructed_normalized.vti'))
        save_vti_file(error_norm, os.path.join(vti_dir, f'sample_{sample_idx:03d}_error_normalized.vti'))


def main():
    parser = argparse.ArgumentParser(description='SWAE U_CHI Validation Inference')
    
    # Data parameters
    parser.add_argument('--data-folder', type=str, 
                        default='/u/tawal/0620-NN-based-compression-thera/tt_q01/',
                        help='Path to folder containing HDF5 files')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output-dir', type=str, default='validation_u_chi_results',
                        help='Directory to save validation results')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of validation samples to process')
    parser.add_argument('--num-vti-samples', type=int, default=5,
                        help='Number of random samples to save as VTI files')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--normalize-method', type=str, default='pos_log',
                        help='Normalization method to use (minmax, zscore, pos_log, none)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading U_CHI datasets...")
    _, val_dataset = create_u_chi_datasets(
        data_folder=args.data_folder,
        train_ratio=0.8,
        normalize=True,
        normalize_method=args.normalize_method
    )
    
    # Create data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )
    
    # Load model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    model = create_swae_3d_7x7x7_model(
        latent_dim=checkpoint['args'].latent_dim,
        lambda_reg=checkpoint['args'].lambda_reg
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully (epoch {checkpoint['epoch']})")
    print(f"Model parameters: latent_dim={checkpoint['args'].latent_dim}, lambda_reg={checkpoint['args'].lambda_reg}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select random samples for VTI output
    vti_sample_indices = random.sample(range(min(args.num_samples, len(val_dataset))), 
                                       min(args.num_vti_samples, args.num_samples, len(val_dataset)))
    print(f"Selected samples for VTI output: {vti_sample_indices}")
    
    # Run validation
    print("\nRunning validation inference...")
    print("Note: Two comparison plots will be generated for each sample:")
    print("      1. Normalized (log pos) scale - as model sees the data")
    print("      2. Denormalized (original) scale - in physical units")
    print("      VTI files will be saved in normalized (log pos) scale")
    print("      Metrics are calculated on denormalized (original physical units) data")
    all_metrics = []
    
    with torch.no_grad():
        for sample_idx in range(min(args.num_samples, len(val_dataset))):
            # Get sample
            sample, metadata = val_dataset[sample_idx]
            sample = sample.unsqueeze(0).to(device)  # Add batch dimension
            
            # Reconstruct
            x_recon, z = model(sample)
            
            # Convert to numpy
            original_normalized = sample.cpu().numpy().squeeze()
            reconstructed_normalized = x_recon.cpu().numpy().squeeze()
            
            # Denormalize for metrics calculation (ensures proper PSNR, MSE etc.)
            if hasattr(val_dataset, 'denormalize'):
                original_denorm = val_dataset.denormalize(original_normalized)
                reconstructed_denorm = val_dataset.denormalize(reconstructed_normalized)
            else:
                original_denorm = original_normalized
                reconstructed_denorm = reconstructed_normalized
                
            # Calculate metrics on denormalized data (original physical units)
            metrics = calculate_metrics(original_denorm, reconstructed_denorm)
            
            # For VTI files and plots, use normalized data (log pos scale)
            original = original_normalized
            reconstructed = reconstructed_normalized
            error = original - reconstructed
            
            all_metrics.append(metrics)
            
            # Determine if this sample should get VTI output
            save_vti = sample_idx in vti_sample_indices
            
            # Save results
            save_validation_results(
                original_norm=original,
                reconstructed_norm=reconstructed,
                original_denorm=original_denorm,
                reconstructed_denorm=reconstructed_denorm,
                output_dir=args.output_dir,
                sample_idx=sample_idx,
                save_vti=save_vti
            )
            
            # Print progress
            print(f"Sample {sample_idx + 1}/{args.num_samples}:")
            print(f"  Normalized (log pos) range: [{original.min():.6f}, {original.max():.6f}]")
            print(f"  Denormalized (original) range: [{original_denorm.min():.6f}, {original_denorm.max():.6f}]")
            print(f"  MSE (on denormalized): {metrics['mse']:.6f}")
            print(f"  PSNR (on denormalized): {metrics['psnr']:.2f} dB")
            print(f"  MAE (on denormalized): {metrics['mae']:.6f}")
            print(f"  Correlation (on denormalized): {metrics['correlation']:.6f}")
            print(f"  Error range (normalized): [{error.min():.6f}, {error.max():.6f}]")
            print(f"  ✓ Generated comparison plots for both normalized and denormalized scales")
            if save_vti:
                print(f"  ✓ Saved VTI files in normalized (log pos) scale")
    
    # Calculate and save average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    print("\n" + "="*50)
    print("FINAL VALIDATION METRICS:")
    print("="*50)
    print(f"Average MSE: {avg_metrics['mse']:.6f}")
    print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"Average MAE: {avg_metrics['mae']:.6f}")
    print(f"Average Correlation: {avg_metrics['correlation']:.6f}")
    print(f"Average Error Range: [{avg_metrics['min_error']:.6f}, {avg_metrics['max_error']:.6f}]")
    
    # Calculate compression ratio
    original_size = 7 * 7 * 7  # 343
    compressed_size = checkpoint['args'].latent_dim  # 16
    compression_ratio = original_size / compressed_size
    print(f"Compression Ratio: {compression_ratio:.1f}:1")
    
    # Save average metrics
    metrics_file = os.path.join(args.output_dir, 'average_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Average Validation Metrics:\n")
        f.write("="*30 + "\n")
        for key, value in avg_metrics.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nCompression Ratio: {compression_ratio:.1f}:1\n")
        f.write(f"Model Parameters: latent_dim={checkpoint['args'].latent_dim}, lambda_reg={checkpoint['args'].lambda_reg}\n")
    
    print(f"\nValidation results saved in: {args.output_dir}")
    print(f"Generated dual comparison plots (normalized + denormalized) for all {args.num_samples} samples")
    print(f"VTI files and additional plots saved for {len(vti_sample_indices)} random samples")


if __name__ == "__main__":
    main() 