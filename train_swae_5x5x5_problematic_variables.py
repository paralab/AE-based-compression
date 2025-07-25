#!/usr/bin/env python3
"""
Dual-Model SWAE Training Script
Trains two separate models simultaneously:
1. Model 1: For standard variables using poslog transformation
2. Model 2: For problematic variables using asinh transformation
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

# Try to import tensorboard, make it optional
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: tensorboard not available, logging will be disabled")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.swae_pure_3d_5x5x5_opt import create_swae_3d_5x5x5_model
from models.swae_relative_error_3d_5x5x5 import create_swae_relative_error_model
from models.swae_robust_loss_3d_5x5x5 import create_swae_robust_loss_model
from models.swae_log_loss_3d_5x5x5 import create_swae_log_loss_model
from datasets.all_variables_dataset_5x5x5_opt import create_all_variables_5x5x5_datasets
from datasets.problematic_variables_dataset_5x5x5 import create_problematic_variables_datasets


def calculate_psnr(original, reconstructed):
    """
    Calculate PSNR as defined in the paper:
    PSNR = 20 * log10(vrange(D)) - 10 * log10(mse(D, D'))
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0 or mse < 1e-12:
        return float('inf')
    
    value_range = np.max(original) - np.min(original)
    if value_range == 0 or value_range < 1e-12:
        return float('inf')
    
    # Add small epsilon to prevent log(0)
    psnr = 20 * np.log10(value_range) - 10 * np.log10(mse + 1e-12)
    return psnr


def calculate_compression_ratio():
    """
    Calculate theoretical compression ratio
    Original: 5x5x5 = 125 values
    Compressed: 16 latent dimensions
    Compression ratio: 125/16 ≈ 7.8:1
    """
    original_size = 5 * 5 * 5  # 125
    compressed_size = 16       # latent dimension
    return original_size / compressed_size


def train_epoch_dual(models, train_loaders, optimizers, device, epoch, writer=None):
    """Train both models for one epoch"""
    epoch_losses = {}
    
    for model_name, model in models.items():
        train_loader = train_loaders[model_name]
        optimizer = optimizers[model_name]
        
        model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_sw_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, metadata) in enumerate(train_loader):
            data = data.to(device)  # Shape: (batch_size, 1, 5, 5, 5)
            
            optimizer.zero_grad()
            
            # Forward pass
            x_recon, z = model(data)
            
            # Calculate loss
            loss_dict = model.loss_function(data, x_recon, z)
            loss = loss_dict['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_sw_loss += loss_dict['sw_distance'].item()
            num_batches += 1
            
            # Track actual MSE if using log loss
            if 'actual_mse' in loss_dict:
                if not hasattr(train_epoch_dual, 'total_actual_mse'):
                    train_epoch_dual.total_actual_mse = {}
                if model_name not in train_epoch_dual.total_actual_mse:
                    train_epoch_dual.total_actual_mse[model_name] = 0.0
                train_epoch_dual.total_actual_mse[model_name] += loss_dict['actual_mse'].item()
            
            # Log every 100 batches
            if batch_idx % 100 == 0:
                log_msg = f'[{model_name}] Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                log_msg += f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                log_msg += f'Loss: {loss.item():.9f} '
                log_msg += f'(Recon: {loss_dict["recon_loss"].item():.9f}, '
                log_msg += f'SW: {loss_dict["sw_distance"].item():.9f}'
                if 'actual_mse' in loss_dict:
                    log_msg += f', MSE: {loss_dict["actual_mse"].item():.9e}'
                log_msg += ')'
                print(log_msg)
                
                if writer:
                    global_step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar(f'{model_name}/Train/BatchLoss', loss.item(), global_step)
                    writer.add_scalar(f'{model_name}/Train/BatchReconLoss', loss_dict['recon_loss'].item(), global_step)
                    writer.add_scalar(f'{model_name}/Train/BatchSWLoss', loss_dict['sw_distance'].item(), global_step)
        
        # Store average losses
        epoch_losses[model_name] = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'sw_loss': total_sw_loss / num_batches
        }
    
    return epoch_losses


def validate_epoch_dual(models, val_loaders, device, epoch, writer=None):
    """Validate both models for one epoch"""
    val_losses = {}
    
    for model_name, model in models.items():
        val_loader = val_loaders[model_name]
        
        model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_sw_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, metadata in val_loader:
                data = data.to(device)
                
                # Forward pass
                x_recon, z = model(data)
                
                # Calculate loss
                loss_dict = model.loss_function(data, x_recon, z)
                
                # Accumulate losses
                total_loss += loss_dict['loss'].item()
                total_recon_loss += loss_dict['recon_loss'].item()
                total_sw_loss += loss_dict['sw_distance'].item()
                num_batches += 1
        
        # Store average losses
        avg_losses = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'sw_loss': total_sw_loss / num_batches
        }
        
        val_losses[model_name] = avg_losses
        
        print(f'[{model_name}] Validation Epoch: {epoch}\t'
              f'Loss: {avg_losses["loss"]:.9f} '
              f'(Recon: {avg_losses["recon_loss"]:.9f}, '
              f'SW: {avg_losses["sw_loss"]:.9f})')
        
        if writer:
            writer.add_scalar(f'{model_name}/Val/Loss', avg_losses['loss'], epoch)
            writer.add_scalar(f'{model_name}/Val/ReconLoss', avg_losses['recon_loss'], epoch)
            writer.add_scalar(f'{model_name}/Val/SWLoss', avg_losses['sw_loss'], epoch)
    
    return val_losses


def evaluate_reconstruction_quality_dual(models, val_datasets, device, num_samples=10):
    """Evaluate reconstruction quality for both models"""
    eval_results = {}
    
    for model_name, model in models.items():
        val_dataset = val_datasets[model_name]
        
        model.eval()
        
        psnr_values = []
        mse_values = []
        relative_errors = []
        var_metrics = {}
        
        with torch.no_grad():
            for i in range(min(num_samples, len(val_dataset))):
                # Get sample
                sample, metadata = val_dataset[i]
                sample = sample.unsqueeze(0).to(device)  # Add batch dimension
                
                # Reconstruct
                x_recon, z = model(sample)
                
                # Convert to numpy
                original = sample.cpu().numpy().squeeze()
                reconstructed = x_recon.cpu().numpy().squeeze()
                
                # Denormalize if needed
                if hasattr(val_dataset, 'denormalize'):
                    # Handle different parameter names for different datasets
                    if 'ProblematicVariables' in val_dataset.__class__.__name__:
                        # ProblematicVariablesDataset5x5x5 uses sample_indices (plural)
                        # Need to add batch dimension for the denormalize method
                        original_batch = original[np.newaxis, ...]  # Add batch dimension
                        reconstructed_batch = reconstructed[np.newaxis, ...]
                        original = val_dataset.denormalize(original_batch, sample_indices=[i]).squeeze()
                        reconstructed = val_dataset.denormalize(reconstructed_batch, sample_indices=[i]).squeeze()
                    else:
                        # AllVariablesDataset5x5x5 uses sample_idx (singular)
                        original = val_dataset.denormalize(original, sample_idx=i)
                        reconstructed = val_dataset.denormalize(reconstructed, sample_idx=i)
                
                # Calculate metrics
                mse = np.mean((original - reconstructed) ** 2)
                psnr = calculate_psnr(original, reconstructed)
                
                # Calculate relative error (avoiding division by zero)
                abs_original = np.abs(original)
                mask = abs_original > 1e-10  # Avoid division by very small numbers
                relative_error = np.zeros_like(original)
                relative_error[mask] = np.abs(original[mask] - reconstructed[mask]) / abs_original[mask]
                mean_relative_error = np.mean(relative_error[mask]) if mask.any() else 0.0
                
                psnr_values.append(psnr)
                mse_values.append(mse)
                relative_errors.append(mean_relative_error)
                
                # Track per-variable metrics
                # Handle different metadata keys from different datasets
                var_name = metadata.get('variable', metadata.get('variable_name', 'unknown'))
                if var_name not in var_metrics:
                    var_metrics[var_name] = {'psnrs': [], 'mses': [], 'rel_errors': []}
                var_metrics[var_name]['psnrs'].append(psnr)
                var_metrics[var_name]['mses'].append(mse)
                var_metrics[var_name]['rel_errors'].append(mean_relative_error)
        
        avg_psnr = np.mean(psnr_values)
        avg_mse = np.mean(mse_values)
        avg_rel_error = np.mean(relative_errors)
        
        print(f"\n[{model_name}] Reconstruction Quality ({num_samples} samples):")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average Relative Error: {avg_rel_error:.2%}")
        
        # Print per-variable metrics
        print(f"\nPer-Variable Metrics:")
        for var_name, metrics in sorted(var_metrics.items()):
            var_psnr = np.mean(metrics['psnrs'])
            var_mse = np.mean(metrics['mses'])
            var_rel_error = np.mean(metrics['rel_errors'])
            print(f"  {var_name}: PSNR={var_psnr:.2f} dB, MSE={var_mse:.6f}, RelErr={var_rel_error:.2%}")
        
        eval_results[model_name] = {
            'avg_psnr': avg_psnr,
            'avg_mse': avg_mse,
            'avg_rel_error': avg_rel_error,
            'var_metrics': var_metrics
        }
    
    return eval_results


def main():
    parser = argparse.ArgumentParser(description='Train Dual SWAE Models with Different Transformations')
    
    # Data parameters
    parser.add_argument('--data-folder', type=str, required=True,
                        help='Path to folder containing HDF5 files')
    
    # Define problematic variables
    PROBLEMATIC_VARS = ['U_B2', 'U_SYMAT2', 'U_GT2', 'U_SYMGT2', 'U_SYMAT4', 'U_SYMAT3']
    
    # Model parameters (following Table VI for 3D data)
    parser.add_argument('--latent-dim', type=int, default=16,
                        help='Latent dimension (16 as per Table VI)')
    parser.add_argument('--lambda-reg', type=float, default=0.9,
                        help='Regularization weight for SW distance')
    parser.add_argument('--encoder-channels', type=str, default='32,64,128,256',
                        help='Encoder channel configuration (comma-separated, e.g., "32,64,128,256")')
    parser.add_argument('--arch', type=str, default='conv',
                        choices=['conv', 'mlp', 'gmlp'],
                        help='Network backbone: conv (default), mlp, or gmlp')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Training data ratio (val=0.15, test=0.05 - FIXED 5%% test set)')
    parser.add_argument('--early-stopping-patience', type=int, default=40,
                        help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--early-stopping-min-delta', type=float, default=1e-4,
                        help='Minimum change in validation loss to qualify as an improvement')
    
    # System parameters
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--save-dir', type=str, default='./save/swae_all_vars_5x5x5',
                        help='Directory to save models and logs')
    
    # Evaluation parameters
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='Evaluate reconstruction quality every N epochs')
    parser.add_argument('--save-interval', type=int, default=20,
                        help='Save model every N epochs')
    
    # Loss function parameters
    parser.add_argument('--use-relative-error', action='store_true',
                        help='Use relative error loss for problematic variables model')
    parser.add_argument('--rel-error-epsilon', type=float, default=1e-6,
                        help='Epsilon for relative error calculation')
    parser.add_argument('--use-robust-loss', action='store_true',
                        help='Use robust loss (log-cosh) for problematic variables model')
    parser.add_argument('--robust-loss-type', type=str, default='log_cosh',
                        choices=['log_cosh', 'huber', 'mae'],
                        help='Type of robust loss to use')
    parser.add_argument('--use-log-loss', action='store_true',
                        help='Use log-space loss to prevent vanishing gradients')
    parser.add_argument('--loss-scale', type=float, default=1000.0,
                        help='Scale factor for loss when not using log loss')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Data folder: {args.data_folder}")
    print(f"Theoretical compression ratio: {calculate_compression_ratio():.1f}:1")
    
    # Mixed precision & compile optimizations
    torch.backends.cudnn.benchmark = True
    
    # TF32 optimization (PyTorch 1.7+)
    if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True
        print("✓ TF32 optimization enabled")
    else:
        print(f"⚠️  TF32 not available in PyTorch {torch.__version__}")
    
    # Create save directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'poslog'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'asinh'), exist_ok=True)
    
    # Setup logging
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))
    else:
        writer = None
    
    # Create datasets for both groups
    print("\n" + "="*80)
    print("Creating Dual Datasets:")
    print("="*80)
    
    # 1. Create datasets for standard variables (poslog)
    print("\n1. Creating dataset for standard variables (poslog transformation)...")
    train_dataset_poslog, val_dataset_poslog, test_dataset_poslog = create_all_variables_5x5x5_datasets(
        data_folder=args.data_folder,
        train_ratio=args.train_split,
        val_ratio=0.15,
        normalize=True,
        normalize_method='pos_log',
        exclude_vars=PROBLEMATIC_VARS  # Exclude problematic variables
    )
    
    # 2. Create datasets for problematic variables (asinh)
    print("\n2. Creating dataset for problematic variables (asinh transformation)...")
    # Get all outputs from the function (3 datasets + 3 dataloaders)
    problematic_outputs = create_problematic_variables_datasets(
        data_folder=args.data_folder,
        train_ratio=args.train_split,
        val_ratio=0.15,
        normalize=True,
        normalize_method='auto',  # Uses asinh with optimal scales
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_vars=PROBLEMATIC_VARS
    )
    # Extract only the datasets (first 3 items)
    train_dataset_asinh, val_dataset_asinh, test_dataset_asinh = problematic_outputs[:3]
    
    print(f"\nDataset sizes:")
    print(f"  Standard variables (poslog): Train={len(train_dataset_poslog)}, Val={len(val_dataset_poslog)}, Test={len(test_dataset_poslog)}")
    print(f"  Problematic variables (asinh): Train={len(train_dataset_asinh)}, Val={len(val_dataset_asinh)}, Test={len(test_dataset_asinh)}")
    
    # Create data loaders for both groups
    train_loaders = {
        'poslog': DataLoader(train_dataset_poslog, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=(device.type == 'cuda')),
        'asinh': DataLoader(train_dataset_asinh, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    }
    
    val_loaders = {
        'poslog': DataLoader(val_dataset_poslog, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=(device.type == 'cuda')),
        'asinh': DataLoader(val_dataset_asinh, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    }
    
    val_datasets = {'poslog': val_dataset_poslog, 'asinh': val_dataset_asinh}
    
    # Create models for both groups
    print("\nCreating dual SWAE models...")
    encoder_channels = [int(c) for c in args.encoder_channels.split(',')]
    
    models = {}
    optimizers = {}
    schedulers = {}
    
    for model_name in ['poslog', 'asinh']:
        print(f"\nCreating {model_name} model...")
        
        # Use special loss for asinh (problematic variables) if requested
        if model_name == 'asinh' and args.use_relative_error:
            print(f"  Using relative error loss (epsilon={args.rel_error_epsilon})")
            model = create_swae_relative_error_model(
                latent_dim=args.latent_dim,
                lambda_reg=args.lambda_reg,
                encoder_channels=encoder_channels,
                arch=args.arch,
                use_relative_error=True
            ).to(device)
        elif model_name == 'asinh' and args.use_robust_loss:
            print(f"  Using robust loss: {args.robust_loss_type}")
            model = create_swae_robust_loss_model(
                latent_dim=args.latent_dim,
                lambda_reg=args.lambda_reg,
                encoder_channels=encoder_channels,
                arch=args.arch,
                loss_type=args.robust_loss_type
            ).to(device)
        elif args.use_log_loss:  # Apply log loss to both models if specified
            print(f"  Using log-space loss (scale={args.loss_scale})")
            model = create_swae_log_loss_model(
                latent_dim=args.latent_dim,
                lambda_reg=args.lambda_reg,
                encoder_channels=encoder_channels,
                arch=args.arch,
                loss_scale=args.loss_scale,
                use_log_loss=True
            ).to(device)
        else:
            model = create_swae_3d_5x5x5_model(
                latent_dim=args.latent_dim,
                lambda_reg=args.lambda_reg,
                encoder_channels=encoder_channels,
                arch=args.arch
            ).to(device)
        
        models[model_name] = model
        optimizers[model_name] = optim.Adam(model.parameters(), lr=args.lr)
        schedulers[model_name] = optim.lr_scheduler.StepLR(optimizers[model_name], step_size=30, gamma=0.5)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  {model_name} model: {total_params:,} parameters")
    
    # Apply compile optimization for faster training (PyTorch 2.0+)
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("✓ Model compilation enabled (PyTorch 2.0+)")
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")
    else:
        if not hasattr(torch, 'compile'):
            print(f"⚠️  torch.compile() not available in PyTorch {torch.__version__} (requires 2.0+)")
        else:
            print("⚠️  CUDA not available, skipping model compilation")
    
    print(f"\nTraining settings:")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Lambda regularization: {args.lambda_reg}")
    print(f"  Architecture: {args.arch}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    if args.early_stopping_patience:
        print(f"  Early stopping: enabled (patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta})")
    else:
        print(f"  Early stopping: disabled")
    
    # Training loop
    print(f"\nStarting dual-model training for {args.epochs} epochs...")
    best_val_losses = {'poslog': float('inf'), 'asinh': float('inf')}
    early_stopping_counters = {'poslog': 0, 'asinh': 0}
    best_model_states = {'poslog': None, 'asinh': None}
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train both models
        train_losses = train_epoch_dual(models, train_loaders, optimizers, device, epoch, writer)
        
        # Validate both models
        val_losses = validate_epoch_dual(models, val_loaders, device, epoch, writer)
        
        # Update learning rates
        for model_name in ['poslog', 'asinh']:
            schedulers[model_name].step()
        
        # Early stopping check for both models
        for model_name in ['poslog', 'asinh']:
            current_val_loss = val_losses[model_name]["loss"]
            if current_val_loss < best_val_losses[model_name] - args.early_stopping_min_delta:
                best_val_losses[model_name] = current_val_loss
                early_stopping_counters[model_name] = 0
                # Save best model state
                best_model_states[model_name] = copy.deepcopy(models[model_name].state_dict())
            else:
                early_stopping_counters[model_name] += 1
        
        # Log epoch results
        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch {epoch} completed in {epoch_time:.1f}s')
        print(f'\nModel Performance Summary:')
        for model_name in ['poslog', 'asinh']:
            print(f'  [{model_name}] Train Loss: {train_losses[model_name]["loss"]:.6f}, Val Loss: {val_losses[model_name]["loss"]:.6f}')
            print(f'  [{model_name}] Early stopping counter: {early_stopping_counters[model_name]}/{args.early_stopping_patience}')
        
        # Check for early stopping
        max_patience = max(early_stopping_counters.values())
        if max_patience >= args.early_stopping_patience:
            print(f'\nEarly stopping triggered after {epoch} epochs')
            for model_name in ['poslog', 'asinh']:
                if best_model_states[model_name] is not None:
                    models[model_name].load_state_dict(best_model_states[model_name])
            break
        
        # Save best models
        for model_name in ['poslog', 'asinh']:
            if val_losses[model_name]['loss'] == best_val_losses[model_name]:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': models[model_name].state_dict(),
                    'optimizer_state_dict': optimizers[model_name].state_dict(),
                    'scheduler_state_dict': schedulers[model_name].state_dict(),
                    'train_loss': train_losses[model_name]['loss'],
                    'val_loss': val_losses[model_name]['loss'],
                    'args': args,
                    'model_type': model_name
                }, os.path.join(args.save_dir, model_name, 'best_model.pth'))
                print(f'[{model_name}] New best model saved (val_loss: {best_val_losses[model_name]:.6f})')
        
        # Evaluate reconstruction quality
        if epoch % args.eval_interval == 0:
            eval_results = evaluate_reconstruction_quality_dual(models, val_datasets, device, num_samples=100)
            if writer:
                for model_name in ['poslog', 'asinh']:
                    writer.add_scalar(f'{model_name}/Eval/PSNR', eval_results[model_name]['avg_psnr'], epoch)
                    writer.add_scalar(f'{model_name}/Eval/MSE', eval_results[model_name]['avg_mse'], epoch)
        
        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            for model_name in ['poslog', 'asinh']:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': models[model_name].state_dict(),
                    'optimizer_state_dict': optimizers[model_name].state_dict(),
                    'scheduler_state_dict': schedulers[model_name].state_dict(),
                    'train_loss': train_losses[model_name]['loss'],
                    'val_loss': val_losses[model_name]['loss'],
                    'args': args,
                    'model_type': model_name
                }, os.path.join(args.save_dir, model_name, f'checkpoint_epoch_{epoch}.pth'))
            print(f'Checkpoints saved at epoch {epoch}')
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    final_eval_results = evaluate_reconstruction_quality_dual(models, val_datasets, device, num_samples=50)
    
    # Save final models
    for model_name in ['poslog', 'asinh']:
        torch.save({
            'epoch': epoch,
            'model_state_dict': models[model_name].state_dict(),
            'optimizer_state_dict': optimizers[model_name].state_dict(),
            'scheduler_state_dict': schedulers[model_name].state_dict(),
            'final_train_loss': train_losses[model_name]['loss'],
            'final_val_loss': val_losses[model_name]['loss'],
            'final_psnr': final_eval_results[model_name]['avg_psnr'],
            'final_mse': final_eval_results[model_name]['avg_mse'],
            'args': args,
            'model_type': model_name
        }, os.path.join(args.save_dir, model_name, 'final_model.pth'))
    
    print(f"\nDual-model training completed!")
    print(f"\nFinal Results:")
    print(f"{'='*60}")
    
    overall_psnr = []
    for model_name in ['poslog', 'asinh']:
        print(f"\n[{model_name}] Model:")
        print(f"  Best validation loss: {best_val_losses[model_name]:.6f}")
        print(f"  Final PSNR: {final_eval_results[model_name]['avg_psnr']:.2f} dB")
        print(f"  Final MSE: {final_eval_results[model_name]['avg_mse']:.6f}")
        
        # Collect all PSNRs for overall average
        for var_metrics in final_eval_results[model_name]['var_metrics'].values():
            overall_psnr.extend(var_metrics['psnrs'])
    
    print(f"\nOverall PSNR (both models): {np.mean(overall_psnr):.2f} dB")
    print(f"Models saved in: {args.save_dir}")
    
    # Optional: Save INT8 quantized model for inference
    if args.arch in ['mlp', 'gmlp']:
        try:
            # Try different import paths for different PyTorch versions
            try:
                from torch.ao.quantization import quantize_dynamic
            except ImportError:
                from torch.quantization import quantize_dynamic
            
            print("\nCreating INT8 quantized model for inference...")
            int8_model = quantize_dynamic(
                model.cpu(),                          # must be on CPU
                {nn.Linear},                          # layers to quantise
                dtype=torch.qint8
            )
            int8_path = os.path.join(args.save_dir, 'int8_model.pth')
            torch.save({
                'model_state_dict': int8_model.state_dict(),
                'args': args
            }, int8_path)
            print(f"✓ INT8 quantized model saved to: {int8_path}")
        except ImportError as e:
            print(f"⚠️  INT8 quantization not available in PyTorch {torch.__version__}: {e}")
        except Exception as e:
            print(f"⚠️  Could not create INT8 model: {e}")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main() 