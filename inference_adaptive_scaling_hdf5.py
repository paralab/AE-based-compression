#!/usr/bin/env python3
"""
Inference script for adaptive scaling model to reconstruct HDF5 files
"""

import os
import sys
import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm
import time
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.swae_pure_3d_5x5x5_opt import create_swae_3d_5x5x5_model


def load_model(checkpoint_path, device, arch='mlp'):
    """Load trained model with adaptive scaling"""
    print("Loading adaptive scaling model...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    # Create model
    encoder_channels = [int(c) for c in args.encoder_channels.split(',')]
    model = create_swae_3d_5x5x5_model(
        latent_dim=args.latent_dim,
        lambda_reg=args.lambda_reg,
        encoder_channels=encoder_channels,
        arch=arch
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"✓ Validation loss: {checkpoint['val_loss']:.6f}")
    
    return model, args


def load_scaling_params(filepath='./scaling_params.json'):
    """Load scaling parameters"""
    with open(filepath, 'r') as f:
        scaling_params = json.load(f)
    print(f"✓ Loaded scaling parameters for {len(scaling_params)} variables")
    return scaling_params


def apply_adaptive_scaling(data, var_name, scaling_params):
    """Apply adaptive scaling to data"""
    if var_name not in scaling_params:
        print(f"Warning: No scaling params for {var_name}, using identity")
        return data, 1.0
    
    params = scaling_params[var_name]
    scale_factor = params['scale_factor']
    
    if params['scale_type'] == 'constant':
        scaled_data = (data + params['shift']) * scale_factor
    else:
        scaled_data = data * scale_factor
    
    return scaled_data, scale_factor


def inverse_adaptive_scaling(scaled_data, var_name, scaling_params):
    """Apply inverse scaling to get back original scale"""
    if var_name not in scaling_params:
        return scaled_data
    
    params = scaling_params[var_name]
    scale_factor = params['scale_factor']
    
    if params['scale_type'] == 'constant':
        original_data = scaled_data / scale_factor - params['shift']
    else:
        original_data = scaled_data / scale_factor
    
    return original_data


def calculate_metrics(original, reconstructed):
    """Calculate reconstruction metrics"""
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    
    # PSNR calculation
    value_range = np.max(original) - np.min(original)
    if value_range > 0 and mse > 0:
        psnr = 20 * np.log10(value_range) - 10 * np.log10(mse)
    else:
        psnr = float('inf')
    
    # Relative error (with protection for near-zero values)
    abs_original = np.abs(original)
    mask = abs_original > 1e-10
    if mask.any():
        relative_errors = np.abs(original[mask] - reconstructed[mask]) / abs_original[mask]
        mean_rel_error = np.mean(relative_errors)
    else:
        mean_rel_error = 0.0
    
    return {
        'mse': mse,
        'mae': mae,
        'psnr': psnr,
        'rel_error': mean_rel_error
    }


def process_variable(var_data, var_name, model, device, scaling_params, batch_size=256):
    """Process a single variable through the model"""
    
    print(f"  Processing {var_name}...")
    
    # Process each sample
    num_samples = var_data.shape[0]
    reconstructed_data = np.zeros_like(var_data)
    latent_codes = []
    
    for i in tqdm(range(0, num_samples, batch_size), desc=f"  {var_name}"):
        batch_end = min(i + batch_size, num_samples)
        batch_data = var_data[i:batch_end]  # Shape: (batch, 7, 7, 7)
        
        # Extract 5x5x5 center crop
        batch_5x5x5 = batch_data[:, 1:6, 1:6, 1:6]
        
        # Apply adaptive scaling
        scaled_batch, scale_factor = apply_adaptive_scaling(batch_5x5x5, var_name, scaling_params)
        
        # Add channel dimension and convert to tensor
        scaled_batch = scaled_batch[:, np.newaxis, :, :, :]  # (batch, 1, 5, 5, 5)
        batch_tensor = torch.from_numpy(scaled_batch).float().to(device)
        
        # Forward pass through model
        with torch.no_grad():
            recon_5x5x5, z = model(batch_tensor)
            latent_codes.append(z.cpu().numpy())
        
        # Convert back to numpy
        recon_5x5x5 = recon_5x5x5.cpu().numpy().squeeze(1)  # Remove channel dim
        
        # Inverse scaling
        denorm_recon = inverse_adaptive_scaling(recon_5x5x5, var_name, scaling_params)
        
        # Pad back to 7x7x7
        for j in range(len(denorm_recon)):
            padded_sample = np.zeros((7, 7, 7), dtype=var_data.dtype)
            padded_sample[1:6, 1:6, 1:6] = denorm_recon[j]
            
            # Copy boundary values from original
            padded_sample[0, :, :] = batch_data[j, 0, :, :]
            padded_sample[6, :, :] = batch_data[j, 6, :, :]
            padded_sample[:, 0, :] = batch_data[j, :, 0, :]
            padded_sample[:, 6, :] = batch_data[j, :, 6, :]
            padded_sample[:, :, 0] = batch_data[j, :, :, 0]
            padded_sample[:, :, 6] = batch_data[j, :, :, 6]
            
            reconstructed_data[i + j] = padded_sample
    
    latent_codes = np.concatenate(latent_codes, axis=0)
    
    return reconstructed_data, latent_codes


def main():
    parser = argparse.ArgumentParser(description='Reconstruct HDF5 file using adaptive scaling SWAE model')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to input HDF5 file')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to output reconstructed HDF5 file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--scaling-params', type=str, default='./scaling_params.json',
                        help='Path to scaling parameters file')
    parser.add_argument('--latent-file', type=str, default=None,
                        help='Path to save latent codes (optional)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for inference')
    parser.add_argument('--arch', type=str, default='mlp',
                        choices=['conv', 'mlp', 'gmlp'],
                        help='Model architecture')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and scaling parameters
    model, train_args = load_model(args.checkpoint, device, args.arch)
    scaling_params = load_scaling_params(args.scaling_params)
    
    # Process HDF5 file
    print(f"\nProcessing {args.input_file}...")
    
    with h5py.File(args.input_file, 'r') as f_in:
        var_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                     for name in f_in['vars'][:]]
        var_data = f_in['var_data'][:]  # Shape: (num_vars, num_samples, 7, 7, 7)
        
        # Get U_ variables only
        u_var_indices = [i for i, name in enumerate(var_names) if name.startswith('U_')]
        u_var_names = [var_names[i] for i in u_var_indices]
        
        print(f"Found {len(u_var_names)} U_ variables to process")
        
        # Create output arrays
        reconstructed_var_data = np.copy(var_data)  # Copy to preserve non-U variables
        all_latent_codes = {}
        all_metrics = {}
        
        # Process each U_ variable
        start_time = time.time()
        
        for idx in u_var_indices:
            var_name = var_names[idx]
            print(f"\nProcessing {var_name}...")
            
            # Process variable
            original_data = var_data[idx]
            reconstructed_data, latent_codes = process_variable(
                original_data, var_name, model, device, scaling_params, args.batch_size
            )
            
            # Store results
            reconstructed_var_data[idx] = reconstructed_data
            all_latent_codes[var_name] = latent_codes
            
            # Calculate metrics
            metrics = calculate_metrics(original_data, reconstructed_data)
            all_metrics[var_name] = metrics
            
            print(f"  Metrics: PSNR={metrics['psnr']:.2f} dB, MSE={metrics['mse']:.6e}, " +
                  f"MAE={metrics['mae']:.6e}, RelErr={metrics['rel_error']:.2%}")
        
        # Save reconstructed HDF5
        print(f"\nSaving reconstructed data to {args.output_file}...")
        with h5py.File(args.output_file, 'w') as f_out:
            f_out.create_dataset('vars', data=f_in['vars'][:])
            f_out.create_dataset('var_data', data=reconstructed_var_data, compression='gzip')
            
            # Save metadata
            f_out.attrs['original_file'] = args.input_file
            f_out.attrs['checkpoint'] = args.checkpoint
            f_out.attrs['reconstruction_time'] = time.time() - start_time
            f_out.attrs['scaling_method'] = 'adaptive'
            
            # Save metrics as attributes
            metrics_group = f_out.create_group('metrics')
            for var_name, metrics in all_metrics.items():
                var_group = metrics_group.create_group(var_name)
                for metric_name, value in metrics.items():
                    var_group.attrs[metric_name] = value
        
        # Optionally save latent codes
        if args.latent_file:
            print(f"Saving latent codes to {args.latent_file}...")
            np.savez_compressed(args.latent_file, **all_latent_codes)
        
        # Print summary
        print("\n" + "="*80)
        print("RECONSTRUCTION SUMMARY")
        print("="*80)
        print(f"{'Variable':<12} {'PSNR (dB)':<12} {'MSE':<12} {'MAE':<12} {'Rel Error':<12}")
        print("-"*60)
        
        for var_name, metrics in sorted(all_metrics.items()):
            print(f"{var_name:<12} {metrics['psnr']:<12.2f} {metrics['mse']:<12.3e} "
                  f"{metrics['mae']:<12.3e} {metrics['rel_error']:<12.2%}")
        
        # Overall statistics
        all_psnrs = [m['psnr'] for m in all_metrics.values() if np.isfinite(m['psnr'])]
        all_mses = [m['mse'] for m in all_metrics.values()]
        all_maes = [m['mae'] for m in all_metrics.values()]
        all_rel_errors = [m['rel_error'] for m in all_metrics.values()]
        
        print("-"*60)
        print(f"{'AVERAGE':<12} {np.mean(all_psnrs):<12.2f} {np.mean(all_mses):<12.3e} "
              f"{np.mean(all_maes):<12.3e} {np.mean(all_rel_errors):<12.2%}")
        
        print(f"\nTotal time: {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    main()