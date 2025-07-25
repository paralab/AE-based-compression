#!/usr/bin/env python3
"""
Train SWAE with adaptive scaling for each variable
This ensures all variables are in a learnable range regardless of their original scale
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: TensorBoard not available. Logging disabled.")
import numpy as np
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.swae_pure_3d_5x5x5_opt import create_swae_3d_5x5x5_model
from datasets.adaptive_scaling_dataset_5x5x5 import create_adaptive_scaling_datasets

# Define problematic variables that need special attention
PROBLEMATIC_VARS = ['U_B2', 'U_SYMAT2', 'U_GT2', 'U_SYMGT2', 'U_SYMAT4', 'U_SYMAT3']


def train_epoch(model, train_loader, optimizer, device, epoch, writer=None):
    """Train for one epoch with adaptive scaling"""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_sw_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, metadata) in enumerate(progress_bar):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        x_recon, z = model(data)
        
        # Calculate loss
        loss_dict = model.loss_function(data, x_recon, z)
        loss = loss_dict['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += loss_dict['recon_loss'].item()
        total_sw_loss += loss_dict['sw_distance'].item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'Recon': f'{loss_dict["recon_loss"].item():.6f}',
            'SW': f'{loss_dict["sw_distance"].item():.6f}'
        })
        
        # Log to tensorboard
        if writer and batch_idx % 100 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            writer.add_scalar('Train/BatchReconLoss', loss_dict['recon_loss'].item(), global_step)
            writer.add_scalar('Train/BatchSWLoss', loss_dict['sw_distance'].item(), global_step)
    
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_sw_loss = total_sw_loss / num_batches
    
    return avg_loss, avg_recon_loss, avg_sw_loss


def validate(model, val_loader, device, epoch, writer=None):
    """Validate the model"""
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
            
            total_loss += loss_dict['loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_sw_loss += loss_dict['sw_distance'].item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_sw_loss = total_sw_loss / num_batches
    
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/ReconLoss', avg_recon_loss, epoch)
        writer.add_scalar('Val/SWLoss', avg_sw_loss, epoch)
    
    return avg_loss, avg_recon_loss, avg_sw_loss


def calculate_psnr(original, reconstructed, data_range=None):
    """Calculate PSNR"""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    if data_range is None:
        data_range = np.max(original) - np.min(original)
    if data_range == 0:
        return float('inf')
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


def evaluate_reconstruction(model, val_dataset, device, num_samples=10):
    """Evaluate reconstruction quality with proper denormalization"""
    model.eval()
    
    var_metrics = {}
    
    with torch.no_grad():
        for i in range(min(num_samples, len(val_dataset))):
            sample, metadata = val_dataset[i]
            sample_input = sample.unsqueeze(0).to(device)
            
            # Reconstruct
            x_recon, _ = model(sample_input)
            
            # Convert to numpy
            scaled_orig = sample.numpy()
            scaled_recon = x_recon.cpu().numpy().squeeze()
            
            # Denormalize to original scale
            orig_denorm = val_dataset.denormalize(scaled_orig[np.newaxis, ...], [i])
            recon_denorm = val_dataset.denormalize(scaled_recon[np.newaxis, ...], [i])
            
            # Ensure same shape
            orig_denorm = orig_denorm.squeeze()
            recon_denorm = recon_denorm.squeeze()
            
            # Calculate metrics in original scale
            mse = np.mean((orig_denorm - recon_denorm) ** 2)
            mae = np.mean(np.abs(orig_denorm - recon_denorm))
            psnr = calculate_psnr(orig_denorm, recon_denorm)
            
            # Calculate relative error
            abs_orig = np.abs(orig_denorm)
            mask = abs_orig > 1e-10
            if mask.any():
                rel_errors = np.abs(orig_denorm[mask] - recon_denorm[mask]) / abs_orig[mask]
                mean_rel_error = np.mean(rel_errors)
            else:
                mean_rel_error = 0.0
            
            # Track per-variable metrics
            var_name = metadata['variable_name']
            if var_name not in var_metrics:
                var_metrics[var_name] = {
                    'mse': [], 'mae': [], 'psnr': [], 'rel_error': []
                }
            
            var_metrics[var_name]['mse'].append(mse)
            var_metrics[var_name]['mae'].append(mae)
            var_metrics[var_name]['psnr'].append(psnr)
            var_metrics[var_name]['rel_error'].append(mean_rel_error)
    
    # Calculate and print averages
    print("\nPer-variable metrics (original scale):")
    print(f"{'Variable':<12} {'MSE':<12} {'MAE':<12} {'PSNR (dB)':<12} {'Rel Error':<12}")
    print("-" * 60)
    
    overall_metrics = {'mse': [], 'mae': [], 'psnr': [], 'rel_error': []}
    
    for var_name in sorted(var_metrics.keys()):
        metrics = var_metrics[var_name]
        avg_mse = np.mean(metrics['mse'])
        avg_mae = np.mean(metrics['mae'])
        avg_psnr = np.mean([p for p in metrics['psnr'] if np.isfinite(p)])
        avg_rel_error = np.mean(metrics['rel_error'])
        
        overall_metrics['mse'].extend(metrics['mse'])
        overall_metrics['mae'].extend(metrics['mae'])
        overall_metrics['psnr'].extend([p for p in metrics['psnr'] if np.isfinite(p)])
        overall_metrics['rel_error'].extend(metrics['rel_error'])
        
        print(f"{var_name:<12} {avg_mse:<12.3e} {avg_mae:<12.3e} {avg_psnr:<12.2f} {avg_rel_error:<12.2%}")
    
    # Overall averages
    print("-" * 60)
    print(f"{'OVERALL':<12} {np.mean(overall_metrics['mse']):<12.3e} "
          f"{np.mean(overall_metrics['mae']):<12.3e} "
          f"{np.mean(overall_metrics['psnr']):<12.2f} "
          f"{np.mean(overall_metrics['rel_error']):<12.2%}")
    
    return np.mean(overall_metrics['mse']), var_metrics


def main():
    parser = argparse.ArgumentParser(description='Train SWAE with adaptive scaling')
    
    # Data parameters
    parser.add_argument('--data-folder', type=str, required=True,
                        help='Path to folder containing HDF5 files')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Ratio of data to use for training')
    
    # Model parameters
    parser.add_argument('--latent-dim', type=int, default=16,
                        help='Latent dimension size')
    parser.add_argument('--lambda-reg', type=float, default=1.0,
                        help='Regularization weight for SW distance')
    parser.add_argument('--encoder-channels', type=str, default='32,64,128,256',
                        help='Encoder channel sizes (comma-separated)')
    parser.add_argument('--arch', type=str, default='mlp',
                        choices=['conv', 'mlp', 'gmlp'],
                        help='Model architecture')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loading workers')
    
    # Scaling parameters
    parser.add_argument('--target-range', type=float, default=10.0,
                        help='Target range for scaled data')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--save-dir', type=str, default='./save/swae_adaptive_scaling',
                        help='Directory to save models and logs')
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='Evaluate every N epochs')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(os.path.join(args.save_dir, 'logs')) if HAS_TENSORBOARD else None
    
    # Create datasets with adaptive scaling
    print("Creating datasets with adaptive scaling...")
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = \
        create_adaptive_scaling_datasets(
            data_folder=args.data_folder,
            train_ratio=args.train_split,
            val_ratio=0.15,
            target_range=args.target_range,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Create model
    encoder_channels = [int(c) for c in args.encoder_channels.split(',')]
    model = create_swae_3d_5x5x5_model(
        latent_dim=args.latent_dim,
        lambda_reg=args.lambda_reg,
        encoder_channels=encoder_channels,
        arch=args.arch
    ).to(device)
    
    print(f"\nModel created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_recon, train_sw = train_epoch(
            model, train_loader, optimizer, device, epoch, writer
        )
        
        # Validate
        val_loss, val_recon, val_sw = validate(
            model, val_loader, device, epoch, writer
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train - Loss: {train_loss:.6f}, Recon: {train_recon:.6f}, SW: {train_sw:.6f}")
        print(f"  Val   - Loss: {val_loss:.6f}, Recon: {val_recon:.6f}, SW: {val_sw:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': args
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print("  ✓ New best model saved!")
        
        # Periodic evaluation
        if epoch % args.eval_interval == 0:
            print("\nEvaluating reconstruction quality...")
            avg_mse, var_metrics = evaluate_reconstruction(
                model, val_dataset, device, num_samples=100
            )
            print(f"Average MSE (original scale): {avg_mse:.6e}")
            
            if writer:
                writer.add_scalar('Eval/MSE_original_scale', avg_mse, epoch)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'args': args
    }, os.path.join(args.save_dir, 'final_model.pth'))
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()