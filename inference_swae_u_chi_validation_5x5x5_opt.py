#!/usr/bin/env python3
"""
SWAE U_CHI Validation Inference Script (5x5x5 Version)
Evaluates trained SWAE model on validation set of U_CHI data
"""

import os
import sys
import argparse
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import time
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

from models.swae_pure_3d_5x5x5_opt import create_swae_3d_5x5x5_model
from datasets.u_chi_dataset_5x5x5_opt import create_u_chi_5x5x5_datasets


def calculate_data_size_bytes(data_shape, dtype=np.float32):
    """Calculate data size in bytes"""
    if dtype == np.float32:
        bytes_per_element = 4
    elif dtype == np.float64:
        bytes_per_element = 8
    else:
        bytes_per_element = 4  # Default to float32
    
    total_elements = np.prod(data_shape)
    return total_elements * bytes_per_element


def measure_compression_speed(model, data_batch, device, num_warmup=5, num_trials=10):
    """
    Measure compression (encoding) speed in GBps
    
    Args:
        model: SWAE model
        data_batch: Input data batch (B, C, H, W, D)
        device: torch device
        num_warmup: Number of warmup runs
        num_trials: Number of measurement trials
    
    Returns:
        compression_speed_gbps: Compression speed in GBps
        compression_time_ms: Average compression time in milliseconds
    """
    model.eval()
    
    # Calculate input data size in bytes
    batch_size = data_batch.shape[0]
    input_size_bytes = calculate_data_size_bytes(data_batch.shape)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.encoder(data_batch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    # Measurement runs
    compression_times = []
    with torch.no_grad():
        for _ in range(num_trials):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            latent = model.encoder(data_batch)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            compression_times.append(end_time - start_time)
    
    # Calculate average time and speed
    avg_compression_time = np.mean(compression_times)
    compression_time_ms = avg_compression_time * 1000
    compression_speed_gbps = (input_size_bytes / avg_compression_time) / 1e9
    
    return compression_speed_gbps, compression_time_ms


def measure_decompression_speed(model, latent_batch, device, num_warmup=5, num_trials=10):
    """
    Measure decompression (decoding) speed in GBps
    
    Args:
        model: SWAE model
        latent_batch: Latent representation batch (B, latent_dim)
        device: torch device
        num_warmup: Number of warmup runs
        num_trials: Number of measurement trials
    
    Returns:
        decompression_speed_gbps: Decompression speed in GBps
        decompression_time_ms: Average decompression time in milliseconds
    """
    model.eval()
    
    # Calculate output data size in bytes (reconstructed data)
    # For SWAE 5x5x5: output shape will be (batch_size, 1, 5, 5, 5)
    batch_size = latent_batch.shape[0]
    output_shape = (batch_size, 1, 5, 5, 5)  # Known output shape for SWAE 5x5x5
    output_size_bytes = calculate_data_size_bytes(output_shape)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.decoder(latent_batch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    # Measurement runs
    decompression_times = []
    with torch.no_grad():
        for _ in range(num_trials):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            reconstructed = model.decoder(latent_batch)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            decompression_times.append(end_time - start_time)
    
    # Calculate average time and speed
    avg_decompression_time = np.mean(decompression_times)
    decompression_time_ms = avg_decompression_time * 1000
    decompression_speed_gbps = (output_size_bytes / avg_decompression_time) / 1e9
    
    return decompression_speed_gbps, decompression_time_ms


def benchmark_model_speed(model, eval_dataset, device, num_samples=50, batch_size=32):
    """
    Comprehensive speed benchmarking of the model
    
    Args:
        model: SWAE model
        eval_dataset: Evaluation dataset
        device: torch device
        num_samples: Number of samples to benchmark
        batch_size: Batch size for benchmarking
    
    Returns:
        speed_results: Dictionary with speed measurements
    """
    print(f"\n🚀 SPEED BENCHMARKING")
    print(f"Benchmarking compression/decompression speeds...")
    print(f"Samples: {num_samples}, Batch size: {batch_size}")
    
    # Prepare test batches
    test_batches = []
    num_batches = min(num_samples // batch_size, len(eval_dataset) // batch_size)
    
    for i in range(num_batches):
        batch_data = []
        for j in range(batch_size):
            if i * batch_size + j < len(eval_dataset):
                sample, _ = eval_dataset[i * batch_size + j]
                batch_data.append(sample)
        
        if len(batch_data) == batch_size:
            batch_tensor = torch.stack(batch_data).to(device)
            test_batches.append(batch_tensor)
    
    if not test_batches:
        print("⚠️ Not enough data for speed benchmarking")
        return None
    
    print(f"Created {len(test_batches)} test batches for benchmarking")
    
    # Measure speeds for each batch
    compression_speeds = []
    decompression_speeds = []
    compression_times = []
    decompression_times = []
    
    for i, batch in enumerate(test_batches):
        print(f"  Benchmarking batch {i+1}/{len(test_batches)}...")
        
        # Measure compression speed
        comp_speed, comp_time = measure_compression_speed(model, batch, device)
        compression_speeds.append(comp_speed)
        compression_times.append(comp_time)
        
        # Get latent representation for decompression benchmark
        with torch.no_grad():
            latent = model.encoder(batch)
        
        # Measure decompression speed
        decomp_speed, decomp_time = measure_decompression_speed(model, latent, device)
        decompression_speeds.append(decomp_speed)
        decompression_times.append(decomp_time)
    
    # Calculate statistics
    speed_results = {
        'compression_speed_gbps_mean': np.mean(compression_speeds),
        'compression_speed_gbps_std': np.std(compression_speeds),
        'compression_speed_gbps_min': np.min(compression_speeds),
        'compression_speed_gbps_max': np.max(compression_speeds),
        'compression_time_ms_mean': np.mean(compression_times),
        'compression_time_ms_std': np.std(compression_times),
        
        'decompression_speed_gbps_mean': np.mean(decompression_speeds),
        'decompression_speed_gbps_std': np.std(decompression_speeds),
        'decompression_speed_gbps_min': np.min(decompression_speeds),
        'decompression_speed_gbps_max': np.max(decompression_speeds),
        'decompression_time_ms_mean': np.mean(decompression_times),
        'decompression_time_ms_std': np.std(decompression_times),
        
        'total_batches_tested': len(test_batches),
        'batch_size': batch_size,
        'device_type': device.type,
    }
    
    # Calculate data processing rates
    single_sample_bytes = calculate_data_size_bytes((1, 5, 5, 5))
    latent_bytes = 16 * 4  # 16 latent dimensions × 4 bytes per float32
    compression_ratio = single_sample_bytes / latent_bytes
    
    speed_results['single_sample_bytes'] = single_sample_bytes
    speed_results['latent_bytes'] = latent_bytes
    speed_results['compression_ratio'] = compression_ratio
    
    return speed_results


def calculate_metrics(original, reconstructed):
    """Calculate reconstruction quality metrics including relative errors"""
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
    
    # Relative errors
    # Avoid division by zero with small epsilon
    epsilon = 1e-12
    abs_original = np.abs(original) + epsilon
    
    # Element-wise relative error
    relative_error_elementwise = np.abs(original - reconstructed) / abs_original
    
    # Statistics of relative errors
    mean_relative_error = np.mean(relative_error_elementwise)
    max_relative_error = np.max(relative_error_elementwise)
    std_relative_error = np.std(relative_error_elementwise)
    
    # Relative error percentiles for detailed analysis
    relative_error_p50 = np.percentile(relative_error_elementwise, 50)  # Median
    relative_error_p95 = np.percentile(relative_error_elementwise, 95)  # 95th percentile
    relative_error_p99 = np.percentile(relative_error_elementwise, 99)  # 99th percentile
    
    # Alternative relative error calculations
    # Global relative error (norm-based)
    norm_original = np.linalg.norm(original)
    norm_error = np.linalg.norm(original - reconstructed)
    global_relative_error = norm_error / (norm_original + epsilon)
    
    # RMS relative error
    rms_relative_error = np.sqrt(np.mean(relative_error_elementwise ** 2))
    
    return {
        'mse': mse,
        'psnr': psnr,
        'mae': mae,
        'correlation': correlation,
        'min_error': np.min(original - reconstructed),
        'max_error': np.max(original - reconstructed),
        'mean_error': np.mean(original - reconstructed),
        'std_error': np.std(original - reconstructed),
        
        # Relative error metrics
        'mean_relative_error': mean_relative_error,
        'mean_relative_error_percent': mean_relative_error * 100,
        'max_relative_error': max_relative_error,
        'max_relative_error_percent': max_relative_error * 100,
        'std_relative_error': std_relative_error,
        'median_relative_error': relative_error_p50,
        'p95_relative_error': relative_error_p95,
        'p99_relative_error': relative_error_p99,
        'global_relative_error': global_relative_error,
        'global_relative_error_percent': global_relative_error * 100,
        'rms_relative_error': rms_relative_error,
        'rms_relative_error_percent': rms_relative_error * 100
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
    parser = argparse.ArgumentParser(description='SWAE U_CHI Validation Inference (5x5x5 Version)')
    
    # Data parameters
    parser.add_argument('--data-folder', type=str, 
                        default='/u/tawal/BSSN-Extracted-Data/tt_q01/',
                        help='Path to folder containing HDF5 files')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output-dir', type=str, default='test_u_chi_5x5x5_results',
                        help='Directory to save test results (5%% held-out set)')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of test samples to process (from 5%% held-out set)')
    parser.add_argument('--num-vti-samples', type=int, default=5,
                        help='Number of random samples to save as VTI files')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--normalize-method', type=str, default='pos_log',
                        help='Normalization method to use (minmax, zscore, pos_log, none)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--arch', type=str, default='conv',
                        choices=['conv', 'mlp', 'gmlp'],
                        help='Network backbone: conv (default), mlp, or gmlp')
    parser.add_argument('--use-int8', action='store_true',
                        help='Use INT8 quantized model for inference')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create datasets - USE FIXED 5% TEST SET for unbiased evaluation!
    print("Loading U_CHI 5x5x5 datasets for UNBIASED testing...")
    print("🔒 Using FIXED test set (5% of data) - NEVER seen during training")
    train_dataset, val_dataset, test_dataset = create_u_chi_5x5x5_datasets(
        data_folder=args.data_folder,
        train_ratio=0.8,  # 80% train
        val_ratio=0.15,   # 15% val  
        normalize=True,   # 5% test (automatically calculated)
        normalize_method=args.normalize_method
    )
    
    # Use FIXED 5% TEST set for final evaluation (not validation set!)
    eval_dataset = test_dataset
    print(f"🎯 Using FIXED TEST dataset (5% of data): {len(eval_dataset)} samples")
    print(f"📊 Data splits: Train={len(train_dataset)} (80%), Val={len(val_dataset)} (15%), Test={len(test_dataset)} (5%)")
    print(f"⚠️  CRITICAL: Testing on held-out test set - NEVER seen during training or validation")
    print(f"🔒 Test set is deterministic (seed=42) - same samples across all runs")
    
    # Create data loader for TEST dataset
    test_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )
    
    # Load model
    print(f"Loading model from {args.model_path}")
    
    # Use weights_only parameter only if available (PyTorch 1.13+)
    import inspect
    sig = inspect.signature(torch.load)
    if 'weights_only' in sig.parameters:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
    
    # Determine architecture from checkpoint or command line
    model_arch = getattr(checkpoint['args'], 'arch', args.arch)
    
    # Create model with appropriate architecture
    model = create_swae_3d_5x5x5_model(
        latent_dim=checkpoint['args'].latent_dim,
        lambda_reg=checkpoint['args'].lambda_reg,
        arch=model_arch
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Apply INT8 quantization if requested
    if args.use_int8 and model_arch in ['mlp', 'gmlp']:
        try:
            # Try different import paths for different PyTorch versions
            try:
                from torch.ao.quantization import quantize_dynamic
            except ImportError:
                from torch.quantization import quantize_dynamic
            
            print("Applying INT8 quantization...")
            model = quantize_dynamic(
                model.cpu(),
                {nn.Linear},
                dtype=torch.qint8
            )
            device = torch.device('cpu')  # INT8 models must run on CPU
            print("✓ INT8 quantization applied, using CPU for inference")
        except ImportError as e:
            print(f"⚠️  INT8 quantization not available in PyTorch {torch.__version__}: {e}")
        except Exception as e:
            print(f"⚠️  Could not apply INT8 quantization: {e}")
    
    model.eval()
    
    print(f"Model loaded successfully (epoch {checkpoint['epoch']})")
    print(f"Model parameters: latent_dim={checkpoint['args'].latent_dim}, lambda_reg={checkpoint['args'].lambda_reg}, arch={model_arch}")
    if args.use_int8:
        print(f"Using INT8 quantization: {args.use_int8}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run speed benchmarking first
    speed_results = benchmark_model_speed(
        model=model, 
        eval_dataset=eval_dataset, 
        device=device,
        num_samples=100,  # Use more samples for stable speed measurements
        batch_size=args.batch_size
    )
    
    # Select random samples for VTI output
    vti_sample_indices = random.sample(range(min(args.num_samples, len(eval_dataset))), 
                                       min(args.num_vti_samples, args.num_samples, len(eval_dataset)))
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
        for sample_idx in range(min(args.num_samples, len(eval_dataset))):
            # Get sample
            sample, metadata = eval_dataset[sample_idx]
            sample = sample.unsqueeze(0).to(device)  # Add batch dimension
            
            # Reconstruct
            x_recon, z = model(sample)
            
            # Convert to numpy
            original_normalized = sample.cpu().numpy().squeeze()
            reconstructed_normalized = x_recon.cpu().numpy().squeeze()
            
            # Denormalize for metrics calculation (ensures proper PSNR, MSE etc.)
            # FIXED: Use correct sample index for per-sample denormalization
            if hasattr(eval_dataset, 'denormalize'):
                original_denorm = eval_dataset.denormalize(original_normalized, sample_idx=sample_idx)
                reconstructed_denorm = eval_dataset.denormalize(reconstructed_normalized, sample_idx=sample_idx)
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
            print(f"  Mean Relative Error: {metrics['mean_relative_error_percent']:.2f}%")
            print(f"  Max Relative Error: {metrics['max_relative_error_percent']:.2f}%")
            print(f"  Global Relative Error: {metrics['global_relative_error_percent']:.2f}%")
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
    print("🎯 FINAL VALIDATION METRICS (5x5x5 MODEL):")
    print("="*50)
    print(f"Average MSE: {avg_metrics['mse']:.6f}")
    print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"Average MAE: {avg_metrics['mae']:.6f}")
    print(f"Average Correlation: {avg_metrics['correlation']:.6f}")
    print(f"Average Error Range: [{avg_metrics['min_error']:.6f}, {avg_metrics['max_error']:.6f}]")
    
    print("\n📊 RELATIVE ERROR ANALYSIS:")
    print(f"Mean Relative Error: {avg_metrics['mean_relative_error_percent']:.2f}%")
    print(f"Max Relative Error: {avg_metrics['max_relative_error_percent']:.2f}%")
    print(f"Median Relative Error: {avg_metrics['median_relative_error']*100:.2f}%")
    print(f"95th Percentile Relative Error: {avg_metrics['p95_relative_error']*100:.2f}%")
    print(f"99th Percentile Relative Error: {avg_metrics['p99_relative_error']*100:.2f}%")
    print(f"Global Relative Error (norm-based): {avg_metrics['global_relative_error_percent']:.2f}%")
    print(f"RMS Relative Error: {avg_metrics['rms_relative_error_percent']:.2f}%")
    
    # Display speed results
    if speed_results:
        print("\n🚀 COMPRESSION/DECOMPRESSION SPEED ANALYSIS:")
        print(f"Device: {speed_results['device_type'].upper()}")
        print(f"Batch Size: {speed_results['batch_size']}")
        print(f"Total Batches Tested: {speed_results['total_batches_tested']}")
        print(f"")
        print(f"💾 Data Sizes:")
        print(f"  Original sample: {speed_results['single_sample_bytes']} bytes ({speed_results['single_sample_bytes']/1024:.1f} KB)")
        print(f"  Compressed (latent): {speed_results['latent_bytes']} bytes")
        print(f"  Compression ratio: {speed_results['compression_ratio']:.1f}:1")
        print(f"")
        print(f"⚡ COMPRESSION (Encoding) Speed:")
        print(f"  Average: {speed_results['compression_speed_gbps_mean']:.3f} ± {speed_results['compression_speed_gbps_std']:.3f} GBps")
        print(f"  Range: {speed_results['compression_speed_gbps_min']:.3f} - {speed_results['compression_speed_gbps_max']:.3f} GBps")
        print(f"  Average time: {speed_results['compression_time_ms_mean']:.2f} ± {speed_results['compression_time_ms_std']:.2f} ms")
        print(f"")
        print(f"🔓 DECOMPRESSION (Decoding) Speed:")
        print(f"  Average: {speed_results['decompression_speed_gbps_mean']:.3f} ± {speed_results['decompression_speed_gbps_std']:.3f} GBps")
        print(f"  Range: {speed_results['decompression_speed_gbps_min']:.3f} - {speed_results['decompression_speed_gbps_max']:.3f} GBps")
        print(f"  Average time: {speed_results['decompression_time_ms_mean']:.2f} ± {speed_results['decompression_time_ms_std']:.2f} ms")
    
    print("\n🎯 COMPRESSION COMPARISON:")
    print(f"7x7x7 Model: 21.4:1 compression ratio")
    print(f"5x5x5 Model: 7.8:1 compression ratio")
    print(f"Trade-off: Lower compression but potentially better reconstruction quality")
    
    # Calculate compression ratio
    original_size = 5 * 5 * 5  # 125
    compressed_size = checkpoint['args'].latent_dim  # 16
    compression_ratio = original_size / compressed_size
    print(f"Compression Ratio: {compression_ratio:.1f}:1")
    
    # Save average metrics
    metrics_file = os.path.join(args.output_dir, 'average_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("🎯 5x5x5 MODEL Validation Metrics:\n")
        f.write("="*40 + "\n")
        f.write("Model: SWAE 3D 5x5x5 with FIXED per-sample pos_log normalization\n")
        f.write(f"Timestamp: {args.output_dir.split('_')[-2:]}\n\n")
        
        f.write("Core Quality Metrics:\n")
        f.write("-"*20 + "\n")
        core_metrics = ['mse', 'psnr', 'mae', 'correlation']
        for key in core_metrics:
            if key in avg_metrics:
                f.write(f"{key}: {avg_metrics[key]}\n")
        
        f.write("\nRelative Error Analysis:\n")
        f.write("-"*25 + "\n")
        relative_metrics = [
            'mean_relative_error_percent', 'max_relative_error_percent', 
            'median_relative_error', 'p95_relative_error', 'p99_relative_error',
            'global_relative_error_percent', 'rms_relative_error_percent'
        ]
        for key in relative_metrics:
            if key in avg_metrics:
                if 'percent' not in key and key != 'mean_relative_error_percent' and key != 'max_relative_error_percent' and key != 'global_relative_error_percent' and key != 'rms_relative_error_percent':
                    # Convert to percentage for non-percent keys
                    f.write(f"{key}: {avg_metrics[key]*100:.2f}%\n")
                else:
                    f.write(f"{key}: {avg_metrics[key]:.2f}%\n")
        
        f.write("\nAll Metrics (Complete):\n")
        f.write("-"*25 + "\n")
        for key, value in avg_metrics.items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\nModel Configuration:\n")
        f.write("-"*20 + "\n")
        f.write(f"Compression Ratio: {compression_ratio:.1f}:1\n")
        f.write(f"Latent Dimension: {checkpoint['args'].latent_dim}\n")
        f.write(f"Lambda Regularization: {checkpoint['args'].lambda_reg}\n")
        
        # Add speed results to file
        if speed_results:
            f.write(f"\nSpeed Performance Analysis:\n")
            f.write("-"*30 + "\n")
            f.write(f"Device: {speed_results['device_type'].upper()}\n")
            f.write(f"Batch Size: {speed_results['batch_size']}\n")
            f.write(f"Total Batches Tested: {speed_results['total_batches_tested']}\n")
            f.write(f"\nData Sizes:\n")
            f.write(f"Original sample: {speed_results['single_sample_bytes']} bytes ({speed_results['single_sample_bytes']/1024:.1f} KB)\n")
            f.write(f"Compressed (latent): {speed_results['latent_bytes']} bytes\n")
            f.write(f"Compression ratio: {speed_results['compression_ratio']:.1f}:1\n")
            f.write(f"\nCompression (Encoding) Speed:\n")
            f.write(f"Average: {speed_results['compression_speed_gbps_mean']:.3f} ± {speed_results['compression_speed_gbps_std']:.3f} GBps\n")
            f.write(f"Range: {speed_results['compression_speed_gbps_min']:.3f} - {speed_results['compression_speed_gbps_max']:.3f} GBps\n")
            f.write(f"Average time: {speed_results['compression_time_ms_mean']:.2f} ± {speed_results['compression_time_ms_std']:.2f} ms\n")
            f.write(f"\nDecompression (Decoding) Speed:\n")
            f.write(f"Average: {speed_results['decompression_speed_gbps_mean']:.3f} ± {speed_results['decompression_speed_gbps_std']:.3f} GBps\n")
            f.write(f"Range: {speed_results['decompression_speed_gbps_min']:.3f} - {speed_results['decompression_speed_gbps_max']:.3f} GBps\n")
            f.write(f"Average time: {speed_results['decompression_time_ms_mean']:.2f} ± {speed_results['decompression_time_ms_std']:.2f} ms\n")
        
        f.write(f"\nComparison Analysis:\n")
        f.write("-"*20 + "\n")
        f.write(f"7x7x7 Model: 21.4:1 compression ratio\n")
        f.write(f"5x5x5 Model: 7.8:1 compression ratio\n")
        f.write(f"Trade-off: Lower compression but potentially better reconstruction quality\n")
        
        # Add speed performance summary
        if speed_results:
            f.write(f"\nSpeed Performance Summary:\n")
            f.write("-"*25 + "\n")
            f.write(f"Compression Speed: {speed_results['compression_speed_gbps_mean']:.3f} GBps\n")
            f.write(f"Decompression Speed: {speed_results['decompression_speed_gbps_mean']:.3f} GBps\n")
            f.write(f"Total throughput for round-trip: {min(speed_results['compression_speed_gbps_mean'], speed_results['decompression_speed_gbps_mean']):.3f} GBps\n")
            
            # Calculate real-world performance estimates
            samples_per_second_comp = (speed_results['compression_speed_gbps_mean'] * 1e9) / speed_results['single_sample_bytes']
            samples_per_second_decomp = (speed_results['decompression_speed_gbps_mean'] * 1e9) / speed_results['single_sample_bytes']
            f.write(f"Samples/second (compression): {samples_per_second_comp:,.0f}\n")
            f.write(f"Samples/second (decompression): {samples_per_second_decomp:,.0f}\n")
    
    print(f"\n🎯 TEST RESULTS (5% held-out set) saved in: {args.output_dir}")
    print(f"✅ Unbiased evaluation completed on {args.num_samples} test samples NEVER seen during training")
    print(f"📊 Generated dual comparison plots (normalized + denormalized) for all {args.num_samples} test samples")
    print(f"🗂️ VTI files and additional plots saved for {len(vti_sample_indices)} random test samples")
    
    if speed_results:
        print(f"🚀 Speed benchmarking completed on {speed_results['total_batches_tested']} batches:")
        print(f"   Compression: {speed_results['compression_speed_gbps_mean']:.3f} GBps")
        print(f"   Decompression: {speed_results['decompression_speed_gbps_mean']:.3f} GBps")
    
    print(f"🔒 Test set is deterministic (seed=42) - same samples across all evaluation runs")


if __name__ == "__main__":
    main() 