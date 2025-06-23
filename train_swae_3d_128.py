#!/usr/bin/env python3
"""
SWAE 3D Training Script for 128x128x128 Resolution
Extended from the 40x40x40 implementation to demonstrate scalability

This implements the same SWAE approach but for larger volumes:
- Block-wise processing of 3D data (8x8x8 blocks)
- SWAE with sliced Wasserstein distance
- 128x128x128 resolution = 32,768 blocks per sample (vs 125 for 40x40x40)
- Same compression ratio: 32:1 per block
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

from models.swae_pure_3d import create_swae_3d_model
from datasets.swae_3d_dataset import create_swae_3d_datasets


def calculate_psnr(original, reconstructed):
    """
    Calculate PSNR as defined in the paper:
    PSNR = 20 * log10(vrange(D)) - 10 * log10(mse(D, D'))
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    
    value_range = np.max(original) - np.min(original)
    psnr = 20 * np.log10(value_range) - 10 * np.log10(mse)
    return psnr


def calculate_compression_ratio(original_size, latent_dim, batch_size):
    """
    Calculate theoretical compression ratio
    Original: 8x8x8 = 512 float32 values = 2048 bytes
    Compressed: latent_dim float32 values = latent_dim * 4 bytes
    """
    original_bytes = 512 * 4  # 8x8x8 * 4 bytes per float32
    compressed_bytes = latent_dim * 4
    return original_bytes / compressed_bytes


def train_epoch(model, train_loader, optimizer, device, epoch, writer=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_sw_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, metadata) in enumerate(train_loader):
        data = data.to(device)  # Shape: (batch_size, 1, 8, 8, 8)
        
        optimizer.zero_grad()
        
        # Forward pass
        x_recon, z = model(data)
        
        # Calculate loss
        loss_dict = model.loss_function(data, x_recon, z)
        loss = loss_dict['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += loss_dict['recon_loss'].item()
        total_sw_loss += loss_dict['sw_distance'].item()
        num_batches += 1
        
        # Log every 200 batches (less frequent due to more blocks)
        if batch_idx % 200 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f} '
                  f'(Recon: {loss_dict["recon_loss"].item():.6f}, '
                  f'SW: {loss_dict["sw_distance"].item():.6f})')
            
            if writer:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                writer.add_scalar('Train/BatchReconLoss', loss_dict['recon_loss'].item(), global_step)
                writer.add_scalar('Train/BatchSWLoss', loss_dict['sw_distance'].item(), global_step)
    
    # Return average losses
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'sw_loss': total_sw_loss / num_batches
    }


def validate_epoch(model, val_loader, device, epoch, writer=None):
    """Validate for one epoch"""
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
    
    # Return average losses
    avg_losses = {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'sw_loss': total_sw_loss / num_batches
    }
    
    print(f'Validation Epoch: {epoch}\t'
          f'Loss: {avg_losses["loss"]:.6f} '
          f'(Recon: {avg_losses["recon_loss"]:.6f}, '
          f'SW: {avg_losses["sw_loss"]:.6f})')
    
    if writer:
        writer.add_scalar('Val/Loss', avg_losses['loss'], epoch)
        writer.add_scalar('Val/ReconLoss', avg_losses['recon_loss'], epoch)
        writer.add_scalar('Val/SWLoss', avg_losses['sw_loss'], epoch)
    
    return avg_losses


def evaluate_reconstruction_quality(model, val_dataset, device, num_samples=3):
    """Evaluate reconstruction quality with PSNR (fewer samples due to size)"""
    model.eval()
    psnr_values = []
    
    print(f"Evaluating reconstruction quality on {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(val_dataset.base_dataset))):
            print(f"  Reconstructing sample {i+1}/{num_samples}...")
            # Reconstruct full sample
            reconstructed, original = val_dataset.reconstruct_full_sample(i, model, device)
            
            # Calculate PSNR
            psnr = calculate_psnr(original, reconstructed)
            psnr_values.append(psnr)
            
            print(f"  Sample {i}: PSNR = {psnr:.2f} dB")
    
    avg_psnr = np.mean(psnr_values)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    return avg_psnr, psnr_values


def main():
    parser = argparse.ArgumentParser(description='Train SWAE 3D Autoencoder for 128x128x128 Resolution')
    
    # Data parameters (128x128x128 specific)
    parser.add_argument('--k-values', nargs='+', type=int, default=[2, 3, 4, 5, 6],
                        help='K values for mathematical functions')
    parser.add_argument('--resolution', type=int, default=128,
                        help='Resolution of 3D functions (128x128x128)')
    parser.add_argument('--block-size', type=int, default=8,
                        help='Block size (8x8x8 as per paper)')
    
    # Model parameters (same as 40x40x40)
    parser.add_argument('--latent-dim', type=int, default=16,
                        help='Latent dimension (16 as per Table VI)')
    parser.add_argument('--lambda-reg', type=float, default=1.0,
                        help='Regularization weight for SW distance')
    
    # Training parameters (adjusted for larger dataset)
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (smaller due to more blocks per sample)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (fewer due to more data)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Train/validation split ratio')
    
    # System parameters
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--save-dir', type=str, default='./save/swae_3d_128',
                        help='Directory to save models and logs')
    
    # Evaluation parameters (adjusted for computational cost)
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='Evaluate reconstruction quality every N epochs')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save model every N epochs')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Training SWAE 3D for {args.resolution}x{args.resolution}x{args.resolution} resolution")
    
    # Calculate blocks per sample
    blocks_per_dim = args.resolution // args.block_size
    blocks_per_sample = blocks_per_dim ** 3
    print(f"Blocks per sample: {blocks_per_dim}^3 = {blocks_per_sample:,}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup logging
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))
    else:
        writer = None
    
    # Create datasets (fewer total functions due to computational cost)
    print("Creating datasets...")
    
    # For 128x128x128, use fewer functions to manage computational load
    # 128^3 has 32,768 blocks per sample vs 125 for 40^3
    num_functions = 50 if args.resolution >= 128 else 500
    print(f"Using {num_functions} functions for resolution {args.resolution}^3")
    
    # Create base dataset with adjusted parameters
    from datasets.math_function_3d import MathFunction3DDataset
    from torch.utils.data import Subset
    
    base_dataset = MathFunction3DDataset(
        k_values=args.k_values,
        resolution=args.resolution,
        num_functions=num_functions,
        repeat=1,
        cache='in_memory',
        diagonal_only=True
    )
    
    # Split into train and validation indices
    total_samples = len(base_dataset)
    train_size = int(args.train_split * total_samples)
    
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subset datasets
    train_base = Subset(base_dataset, train_indices)
    val_base = Subset(base_dataset, val_indices)
    
    # Create block datasets
    from datasets.swae_3d_dataset import SWAE3DBlockDataset
    train_dataset = SWAE3DBlockDataset(train_base, block_size=args.block_size, normalize=True)
    val_dataset = SWAE3DBlockDataset(val_base, block_size=args.block_size, normalize=True)
    
    # Use same normalization parameters for validation
    val_dataset.global_min = train_dataset.global_min
    val_dataset.global_max = train_dataset.global_max
    
    print(f"\nDataset split:")
    print(f"  Train samples: {len(train_base)} ({len(train_dataset):,} blocks)")
    print(f"  Val samples: {len(val_base)} ({len(val_dataset):,} blocks)")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    
    # Create model (same architecture, different input resolution)
    print("Creating SWAE 3D model...")
    model = create_swae_3d_model(
        block_size=args.block_size,
        latent_dim=args.latent_dim,
        lambda_reg=args.lambda_reg
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Calculate theoretical compression ratio
    compression_ratio = calculate_compression_ratio(512, args.latent_dim, args.batch_size)
    print(f"Theoretical compression ratio: {compression_ratio:.1f}:1 (per 8x8x8 block)")
    
    # Overall compression for full volume
    original_volume_size = args.resolution ** 3 * 4  # float32 bytes
    compressed_volume_size = blocks_per_sample * args.latent_dim * 4
    overall_compression = original_volume_size / compressed_volume_size
    print(f"Overall volume compression: {overall_compression:.1f}:1 ({args.resolution}^3 → {blocks_per_sample}×{args.latent_dim})")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device, epoch, writer)
        
        # Validate
        val_losses = validate_epoch(model, val_loader, device, epoch, writer)
        
        # Log epoch results
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch} completed in {epoch_time:.1f}s')
        
        if writer:
            writer.add_scalar('Train/EpochLoss', train_losses['loss'], epoch)
            writer.add_scalar('Train/EpochReconLoss', train_losses['recon_loss'], epoch)
            writer.add_scalar('Train/EpochSWLoss', train_losses['sw_loss'], epoch)
        
        # Save best model
        if val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'args': args,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'New best model saved (val_loss: {best_val_loss:.6f})')
        
        # Evaluate reconstruction quality (less frequent due to computational cost)
        if epoch % args.eval_interval == 0:
            avg_psnr, _ = evaluate_reconstruction_quality(model, val_dataset, device, num_samples=2)
            if writer:
                writer.add_scalar('Val/PSNR', avg_psnr, epoch)
        
        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'args': args,
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f'Checkpoint saved at epoch {epoch}')
    
    # Final evaluation
    print("\nFinal evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), weights_only=False)['model_state_dict'])
    final_psnr, psnr_values = evaluate_reconstruction_quality(model, val_dataset, device, num_samples=3)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final PSNR: {final_psnr:.2f} dB")
    print(f"Per-block compression ratio: {compression_ratio:.1f}:1")
    print(f"Overall volume compression: {overall_compression:.1f}:1")
    
    # Save final results
    results = {
        'best_val_loss': best_val_loss,
        'final_psnr': final_psnr,
        'psnr_values': psnr_values,
        'compression_ratio': compression_ratio,
        'overall_compression': overall_compression,
        'resolution': args.resolution,
        'blocks_per_sample': blocks_per_sample,
        'args': vars(args)
    }
    
    torch.save(results, os.path.join(args.save_dir, 'final_results.pth'))
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main() 