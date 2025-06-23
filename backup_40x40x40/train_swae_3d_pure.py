#!/usr/bin/env python3
"""
Pure SWAE 3D Training Script
Based on "Exploring Autoencoder-based Error-bounded Compression for Scientific Data"

This implements exactly what the paper describes:
- Block-wise processing of 3D data (8x8x8 blocks)
- SWAE with sliced Wasserstein distance
- Convolutional architecture with GDN/iGDN
- Configuration matching Table VI for 3D data
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
        
        # Log every 100 batches
        if batch_idx % 100 == 0:
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


def evaluate_reconstruction_quality(model, val_dataset, device, num_samples=5):
    """Evaluate reconstruction quality with PSNR"""
    model.eval()
    psnr_values = []
    
    print(f"Evaluating reconstruction quality on {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(val_dataset.base_dataset))):
            # Reconstruct full sample
            reconstructed, original = val_dataset.reconstruct_full_sample(i, model, device)
            
            # Calculate PSNR
            psnr = calculate_psnr(original, reconstructed)
            psnr_values.append(psnr)
            
            print(f"Sample {i}: PSNR = {psnr:.2f} dB")
    
    avg_psnr = np.mean(psnr_values)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    return avg_psnr, psnr_values


def main():
    parser = argparse.ArgumentParser(description='Train Pure SWAE 3D Autoencoder')
    
    # Data parameters (following paper specifications)
    parser.add_argument('--k-values', nargs='+', type=int, default=[2, 3, 4, 5, 6],
                        help='K values for mathematical functions')
    parser.add_argument('--resolution', type=int, default=40,
                        help='Resolution of 3D functions')
    parser.add_argument('--block-size', type=int, default=8,
                        help='Block size (8x8x8 as per paper)')
    
    # Model parameters (following Table VI for 3D data)
    parser.add_argument('--latent-dim', type=int, default=16,
                        help='Latent dimension (16 as per Table VI)')
    parser.add_argument('--lambda-reg', type=float, default=1.0,
                        help='Regularization weight for SW distance')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Train/validation split ratio')
    
    # System parameters
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--save-dir', type=str, default='./save/swae_3d_pure',
                        help='Directory to save models and logs')
    
    # Evaluation parameters
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='Evaluate reconstruction quality every N epochs')
    parser.add_argument('--save-interval', type=int, default=20,
                        help='Save model every N epochs')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup logging
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))
    else:
        writer = None
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset = create_swae_3d_datasets(
        k_values=args.k_values,
        resolution=args.resolution,
        block_size=args.block_size,
        train_split=args.train_split,
        normalize=True
    )
    
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
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
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
    print(f"Theoretical compression ratio: {compression_ratio:.1f}:1")
    
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
        
        # Evaluate reconstruction quality
        if epoch % args.eval_interval == 0:
            avg_psnr, _ = evaluate_reconstruction_quality(model, val_dataset, device)
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
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'))['model_state_dict'])
    final_psnr, psnr_values = evaluate_reconstruction_quality(model, val_dataset, device, num_samples=10)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final PSNR: {final_psnr:.2f} dB")
    print(f"Theoretical compression ratio: {compression_ratio:.1f}:1")
    
    # Save final results
    results = {
        'best_val_loss': best_val_loss,
        'final_psnr': final_psnr,
        'psnr_values': psnr_values,
        'compression_ratio': compression_ratio,
        'args': vars(args)
    }
    
    torch.save(results, os.path.join(args.save_dir, 'final_results.pth'))
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main() 