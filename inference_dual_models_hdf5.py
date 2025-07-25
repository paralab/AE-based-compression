#!/usr/bin/env python3
"""
Inference script using dual models to reconstruct HDF5 files
Uses poslog model for standard variables and asinh model for problematic variables
"""

import os
import sys
import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.swae_pure_3d_5x5x5_opt import create_swae_3d_5x5x5_model
from models.swae_robust_loss_3d_5x5x5 import create_swae_robust_loss_model


# Define problematic variables that use asinh model
PROBLEMATIC_VARS = ['U_B2', 'U_SYMAT2', 'U_GT2', 'U_SYMGT2', 'U_SYMAT4', 'U_SYMAT3']


def load_models(poslog_checkpoint, asinh_checkpoint, device, arch='mlp'):
    """Load both trained models"""
    print("Loading dual models...")
    
    # Load poslog model
    poslog_state = torch.load(poslog_checkpoint, map_location=device)
    poslog_args = poslog_state['args']
    
    poslog_model = create_swae_3d_5x5x5_model(
        latent_dim=poslog_args.latent_dim,
        lambda_reg=poslog_args.lambda_reg,
        encoder_channels=[int(c) for c in poslog_args.encoder_channels.split(',')],
        arch=arch
    ).to(device)
    poslog_model.load_state_dict(poslog_state['model_state_dict'])
    poslog_model.eval()
    
    # Load asinh model
    asinh_state = torch.load(asinh_checkpoint, map_location=device)
    asinh_args = asinh_state['args']
    
    # Check if it's a robust loss model
    if hasattr(asinh_args, 'use_robust_loss') and asinh_args.use_robust_loss:
        asinh_model = create_swae_robust_loss_model(
            latent_dim=asinh_args.latent_dim,
            lambda_reg=asinh_args.lambda_reg,
            encoder_channels=[int(c) for c in asinh_args.encoder_channels.split(',')],
            arch=arch,
            loss_type=asinh_args.robust_loss_type
        ).to(device)
    else:
        asinh_model = create_swae_3d_5x5x5_model(
            latent_dim=asinh_args.latent_dim,
            lambda_reg=asinh_args.lambda_reg,
            encoder_channels=[int(c) for c in asinh_args.encoder_channels.split(',')],
            arch=arch
        ).to(device)
    
    asinh_model.load_state_dict(asinh_state['model_state_dict'])
    asinh_model.eval()
    
    print(f"✓ Loaded poslog model from epoch {poslog_state.get('epoch', 'unknown')}")
    print(f"✓ Loaded asinh model from epoch {asinh_state.get('epoch', 'unknown')}")
    
    return poslog_model, asinh_model, poslog_args, asinh_args


def apply_poslog_transform(data, epsilon=1e-8):
    """Apply positive-shift log transformation"""
    data_min = data.min()
    data_shifted = data - data_min + epsilon
    log_data = np.log(data_shifted + epsilon)
    return log_data, data_min


def inverse_poslog_transform(log_data, data_min, epsilon=1e-8):
    """Inverse positive-shift log transformation"""
    return np.exp(log_data) - epsilon + data_min


def apply_asinh_transform(data, scale=0.01):
    """Apply asinh transformation"""
    return np.arcsinh(data / scale)


def inverse_asinh_transform(transformed_data, scale=0.01):
    """Inverse asinh transformation"""
    return np.sinh(transformed_data) * scale


def get_asinh_scale(var_name):
    """Get optimal asinh scale for each variable"""
    scales = {
        'U_B2': 0.001,
        'U_SYMAT2': 0.01,
        'U_GT2': 0.001,
        'U_SYMGT2': 0.01,
        'U_SYMAT4': 0.01,
        'U_SYMAT3': 0.01
    }
    return scales.get(var_name, 0.01)


def process_variable(var_data, var_name, model, device, batch_size=256):
    """Process a single variable through the appropriate model"""
    
    # Determine which model and transformation to use
    if var_name in PROBLEMATIC_VARS:
        model_type = 'asinh'
        transform_func = apply_asinh_transform
        inverse_func = inverse_asinh_transform
        transform_params = {'scale': get_asinh_scale(var_name)}
    else:
        model_type = 'poslog'
        transform_func = apply_poslog_transform
        inverse_func = inverse_poslog_transform
        transform_params = {}
    
    print(f"  Using {model_type} model for {var_name}")
    
    # Process each sample
    num_samples = var_data.shape[0]
    reconstructed_data = np.zeros_like(var_data)
    latent_codes = []
    
    for i in tqdm(range(0, num_samples, batch_size), desc=f"  Processing {var_name}"):
        batch_end = min(i + batch_size, num_samples)
        batch_data = var_data[i:batch_end]  # Shape: (batch, 7, 7, 7)
        
        # Extract 5x5x5 center crop
        batch_5x5x5 = batch_data[:, 1:6, 1:6, 1:6]
        
        # Apply transformation
        if model_type == 'poslog':
            # Process each sample individually for poslog
            transformed_batch = []
            transform_params_list = []
            
            for sample in batch_5x5x5:
                transformed, data_min = transform_func(sample, **transform_params)
                transformed_batch.append(transformed)
                transform_params_list.append({'data_min': data_min})
            
            transformed_batch = np.array(transformed_batch)
        else:
            # Apply asinh transformation to whole batch
            transformed_batch = transform_func(batch_5x5x5, **transform_params)
            transform_params_list = [transform_params] * len(batch_5x5x5)
        
        # Add channel dimension and convert to tensor
        transformed_batch = transformed_batch[:, np.newaxis, :, :, :]  # (batch, 1, 5, 5, 5)
        batch_tensor = torch.from_numpy(transformed_batch).float().to(device)
        
        # Forward pass through model
        with torch.no_grad():
            recon_5x5x5, z = model(batch_tensor)
            latent_codes.append(z.cpu().numpy())
        
        # Convert back to numpy
        recon_5x5x5 = recon_5x5x5.cpu().numpy().squeeze(1)  # Remove channel dim
        
        # Inverse transformation
        for j, (recon_sample, params) in enumerate(zip(recon_5x5x5, transform_params_list)):
            if model_type == 'poslog':
                denorm_sample = inverse_func(recon_sample, **params)
            else:
                denorm_sample = inverse_func(recon_sample, **transform_params)
            
            # Pad back to 7x7x7
            padded_sample = np.zeros((7, 7, 7), dtype=var_data.dtype)
            padded_sample[1:6, 1:6, 1:6] = denorm_sample
            
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


def main():
    parser = argparse.ArgumentParser(description='Reconstruct HDF5 file using dual SWAE models')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to input HDF5 file')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to output reconstructed HDF5 file')
    parser.add_argument('--poslog-checkpoint', type=str, required=True,
                        help='Path to poslog model checkpoint')
    parser.add_argument('--asinh-checkpoint', type=str, required=True,
                        help='Path to asinh model checkpoint')
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
    
    # Load models
    poslog_model, asinh_model, _, _ = load_models(
        args.poslog_checkpoint, 
        args.asinh_checkpoint, 
        device,
        args.arch
    )
    
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
        print(f"  Standard variables (poslog): {len([v for v in u_var_names if v not in PROBLEMATIC_VARS])}")
        print(f"  Problematic variables (asinh): {len([v for v in u_var_names if v in PROBLEMATIC_VARS])}")
        
        # Create output arrays
        reconstructed_var_data = np.copy(var_data)  # Copy to preserve non-U variables
        all_latent_codes = {}
        all_metrics = {}
        
        # Process each U_ variable
        start_time = time.time()
        
        for idx in u_var_indices:
            var_name = var_names[idx]
            print(f"\nProcessing {var_name}...")
            
            # Select appropriate model
            if var_name in PROBLEMATIC_VARS:
                model = asinh_model
            else:
                model = poslog_model
            
            # Process variable
            original_data = var_data[idx]
            reconstructed_data, latent_codes = process_variable(
                original_data, var_name, model, device, args.batch_size
            )
            
            # Store results
            reconstructed_var_data[idx] = reconstructed_data
            all_latent_codes[var_name] = latent_codes
            
            # Calculate metrics
            metrics = calculate_metrics(original_data, reconstructed_data)
            all_metrics[var_name] = metrics
            
            print(f"  Metrics: PSNR={metrics['psnr']:.2f} dB, MSE={metrics['mse']:.6f}, " +
                  f"MAE={metrics['mae']:.6f}, RelErr={metrics['rel_error']:.2%}")
        
        # Save reconstructed HDF5
        print(f"\nSaving reconstructed data to {args.output_file}...")
        with h5py.File(args.output_file, 'w') as f_out:
            f_out.create_dataset('vars', data=f_in['vars'][:])
            f_out.create_dataset('var_data', data=reconstructed_var_data, compression='gzip')
            
            # Save metadata
            f_out.attrs['original_file'] = args.input_file
            f_out.attrs['poslog_checkpoint'] = args.poslog_checkpoint
            f_out.attrs['asinh_checkpoint'] = args.asinh_checkpoint
            f_out.attrs['reconstruction_time'] = time.time() - start_time
            
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
        
        # Group metrics by model type
        poslog_metrics = {k: v for k, v in all_metrics.items() if k not in PROBLEMATIC_VARS}
        asinh_metrics = {k: v for k, v in all_metrics.items() if k in PROBLEMATIC_VARS}
        
        print("\nPoslog Model Variables:")
        for var_name, metrics in sorted(poslog_metrics.items()):
            print(f"  {var_name}: PSNR={metrics['psnr']:.2f} dB, MSE={metrics['mse']:.6f}")
        
        print("\nAsinh Model Variables:")
        for var_name, metrics in sorted(asinh_metrics.items()):
            print(f"  {var_name}: PSNR={metrics['psnr']:.2f} dB, MSE={metrics['mse']:.6f}")
        
        # Overall statistics
        all_psnrs = [m['psnr'] for m in all_metrics.values() if np.isfinite(m['psnr'])]
        all_mses = [m['mse'] for m in all_metrics.values()]
        
        print(f"\nOverall Performance:")
        print(f"  Average PSNR: {np.mean(all_psnrs):.2f} dB")
        print(f"  Average MSE: {np.mean(all_mses):.6f}")
        print(f"  Total time: {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    main()