#!/usr/bin/env python3
"""
SWAE 3D Validation Inference Script
Loads the best trained model, picks 5 random validation samples,
runs inference, calculates losses, and saves results as VTI files for ParaView
"""

import os
import sys
import numpy as np
import torch
import random
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.swae_pure_3d import create_swae_3d_model
from datasets.swae_3d_dataset import create_swae_3d_datasets


def calculate_psnr(original, reconstructed):
    """Calculate PSNR"""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    
    value_range = np.max(original) - np.min(original)
    psnr = 20 * np.log10(value_range) - 10 * np.log10(mse)
    return psnr


def save_as_vti(data, filename, spacing=(1.0, 1.0, 1.0)):
    """
    Save 3D numpy array as VTI file for ParaView visualization
    
    Args:
        data: 3D numpy array
        filename: Output filename
        spacing: Grid spacing (dx, dy, dz)
    """
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError:
        print("Warning: VTK not available. Installing pyvista as alternative...")
        try:
            import pyvista as pv
            
            # Create structured grid
            grid = pv.ImageData(dimensions=data.shape)
            grid.spacing = spacing
            grid.point_data["values"] = data.flatten(order='F')
            
            # Save as VTI
            grid.save(filename)
            print(f"Saved {filename} using PyVista")
            return
        except ImportError:
            print("Error: Neither VTK nor PyVista available. Saving as numpy array instead.")
            np.save(filename.replace('.vti', '.npy'), data)
            return
    
    # Use VTK directly
    # Create VTK image data
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(data.shape)
    vtk_data.SetSpacing(spacing)
    vtk_data.SetOrigin(0, 0, 0)
    
    # Convert numpy array to VTK array
    vtk_array = numpy_support.numpy_to_vtk(data.flatten(order='F'))
    vtk_array.SetName("values")
    vtk_data.GetPointData().SetScalars(vtk_array)
    
    # Write VTI file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(vtk_data)
    writer.Write()
    
    print(f"Saved {filename} using VTK")


def load_best_model(save_dir, device):
    """Load the best trained model"""
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    
    # Load checkpoint (set weights_only=False for compatibility with older checkpoints)
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    
    # Extract model parameters from checkpoint
    if 'args' in checkpoint:
        args = checkpoint['args']
        model = create_swae_3d_model(
            block_size=getattr(args, 'block_size', 8),
            latent_dim=getattr(args, 'latent_dim', 16),
            lambda_reg=getattr(args, 'lambda_reg', 1.0)
        )
    else:
        # Use default parameters
        model = create_swae_3d_model()
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['val_losses']['loss']:.6f}")
    
    return model, checkpoint


def run_inference_on_validation_samples(model, val_dataset, device, num_samples=5, output_dir="validation_results"):
    """
    Run inference on random validation samples and save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Pick random samples
    total_samples = len(val_dataset.base_dataset)
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    print(f"Selected samples: {sample_indices}")
    
    results = []
    
    with torch.no_grad():
        for i, sample_idx in enumerate(sample_indices):
            print(f"\nProcessing sample {i+1}/{num_samples} (index {sample_idx})...")
            
            # Reconstruct full sample
            reconstructed, original = val_dataset.reconstruct_full_sample(sample_idx, model, device)
            
            # Calculate error
            error = np.abs(original - reconstructed)
            
            # Calculate metrics
            mse = np.mean((original - reconstructed) ** 2)
            mae = np.mean(error)
            max_error = np.max(error)
            psnr = calculate_psnr(original, reconstructed)
            
            # Print metrics
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  Max Error: {max_error:.6f}")
            print(f"  PSNR: {psnr:.2f} dB")
            print(f"  Original range: [{original.min():.3f}, {original.max():.3f}]")
            print(f"  Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
            
            # Save as VTI files
            sample_name = f"sample_{sample_idx:03d}"
            
            # Original data
            original_file = os.path.join(output_dir, f"{sample_name}_original.vti")
            save_as_vti(original, original_file)
            
            # Reconstructed data
            recon_file = os.path.join(output_dir, f"{sample_name}_reconstructed.vti")
            save_as_vti(reconstructed, recon_file)
            
            # Error data
            error_file = os.path.join(output_dir, f"{sample_name}_error.vti")
            save_as_vti(error, error_file)
            
            # Store results
            results.append({
                'sample_idx': sample_idx,
                'mse': mse,
                'mae': mae,
                'max_error': max_error,
                'psnr': psnr,
                'original_range': (original.min(), original.max()),
                'reconstructed_range': (reconstructed.min(), reconstructed.max())
            })
    
    return results


def main():
    # Configuration
    save_dir = "save/swae_3d_pure_20250622_161533"  # Most recent training
    output_dir = "validation_inference_results"
    num_samples = 5
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load best model
    print("Loading best trained model...")
    model, checkpoint = load_best_model(save_dir, device)
    
    # Create validation dataset (use same parameters as training)
    print("Creating validation dataset...")
    if 'args' in checkpoint:
        args = checkpoint['args']
        k_values = getattr(args, 'k_values', [2, 3])
        resolution = getattr(args, 'resolution', 40)
        block_size = getattr(args, 'block_size', 8)
        train_split = getattr(args, 'train_split', 0.8)
    else:
        # Use defaults
        k_values = [2, 3]
        resolution = 40
        block_size = 8
        train_split = 0.8
    
    _, val_dataset = create_swae_3d_datasets(
        k_values=k_values,
        resolution=resolution,
        block_size=block_size,
        train_split=train_split,
        normalize=True
    )
    
    print(f"Validation dataset size: {len(val_dataset.base_dataset)} samples")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run inference
    print(f"\nRunning inference on {num_samples} random validation samples...")
    results = run_inference_on_validation_samples(
        model, val_dataset, device, num_samples, output_dir
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Model: Best epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['val_losses']['loss']:.6f}")
    print(f"Samples processed: {len(results)}")
    print(f"Results saved to: {output_dir}/")
    
    print(f"\n{'Sample':<8} {'MSE':<10} {'MAE':<10} {'Max Err':<10} {'PSNR (dB)':<10}")
    print("-" * 60)
    
    total_mse = 0
    total_mae = 0
    total_max_error = 0
    total_psnr = 0
    
    for result in results:
        print(f"{result['sample_idx']:<8} {result['mse']:<10.6f} {result['mae']:<10.6f} "
              f"{result['max_error']:<10.6f} {result['psnr']:<10.2f}")
        
        total_mse += result['mse']
        total_mae += result['mae']
        total_max_error += result['max_error']
        total_psnr += result['psnr']
    
    # Print averages
    print("-" * 60)
    avg_mse = total_mse / len(results)
    avg_mae = total_mae / len(results)
    avg_max_error = total_max_error / len(results)
    avg_psnr = total_psnr / len(results)
    
    print(f"{'Average':<8} {avg_mse:<10.6f} {avg_mae:<10.6f} "
          f"{avg_max_error:<10.6f} {avg_psnr:<10.2f}")
    
    print(f"\nVTI files saved for ParaView visualization:")
    print(f"  - *_original.vti: Original data")
    print(f"  - *_reconstructed.vti: Reconstructed data")
    print(f"  - *_error.vti: Absolute error")
    
    print(f"\nTo view in ParaView:")
    print(f"  1. Open ParaView")
    print(f"  2. File -> Open -> Select VTI files")
    print(f"  3. Apply -> Choose 'values' for coloring")
    print(f"  4. Use Volume rendering or Contour filter for 3D visualization")


if __name__ == "__main__":
    main() 