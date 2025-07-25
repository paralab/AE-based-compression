#!/usr/bin/env python3

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


def train_epoch(model, train_loader, optimizer, device, epoch, writer=None):
    """Train for one epoch"""
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
        
        # Gradient clipping for stability (increased for new data distribution)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
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
    
    # Return average losses
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_sw_loss = total_sw_loss / num_batches
    
    # Log to tensorboard
    if writer is not None:
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/train_recon', avg_recon_loss, epoch)
        writer.add_scalar('Loss/train_sw', avg_sw_loss, epoch)
    
    return avg_loss, avg_recon_loss, avg_sw_loss


def evaluate(model, data_loader, device, dataset, epoch, writer=None, split='val'):
    """Evaluate the model on validation or test set"""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_sw_loss = 0.0
    all_psnrs = []
    num_batches = 0
    
    # Store per-variable metrics
    var_metrics = {}
    
    with torch.no_grad():
        for batch_idx, (data, metadata) in enumerate(data_loader):
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
            
            # Calculate PSNR for each sample
            data_np = data.cpu().numpy()
            recon_np = x_recon.cpu().numpy()
            
            # Get sample indices for denormalization
            sample_indices = [metadata['sample_idx'][i].item() for i in range(len(data))]
            
            # Denormalize for PSNR calculation
            data_denorm = dataset.denormalize(data_np, sample_indices)
            recon_denorm = dataset.denormalize(recon_np, sample_indices)
            
            for i in range(len(data)):
                psnr = calculate_psnr(data_denorm[i], recon_denorm[i])
                all_psnrs.append(psnr)
                
                # Track per-variable metrics
                var_name = metadata['variable'][i]
                if var_name not in var_metrics:
                    var_metrics[var_name] = {'psnrs': [], 'mses': []}
                
                mse = np.mean((data_denorm[i] - recon_denorm[i]) ** 2)
                var_metrics[var_name]['psnrs'].append(psnr)
                var_metrics[var_name]['mses'].append(mse)
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_sw_loss = total_sw_loss / num_batches
    avg_psnr = np.mean(all_psnrs)
    
    # Calculate per-variable averages
    print(f"\n{split.upper()} Per-Variable Metrics:")
    print("-" * 60)
    for var_name, metrics in var_metrics.items():
        avg_var_psnr = np.mean(metrics['psnrs'])
        avg_var_mse = np.mean(metrics['mses'])
        print(f"{var_name:12s}: PSNR={avg_var_psnr:8.2f} dB, MSE={avg_var_mse:.6f}")
    
    # Log to tensorboard
    if writer is not None:
        writer.add_scalar(f'Loss/{split}', avg_loss, epoch)
        writer.add_scalar(f'Loss/{split}_recon', avg_recon_loss, epoch)
        writer.add_scalar(f'Loss/{split}_sw', avg_sw_loss, epoch)
        writer.add_scalar(f'PSNR/{split}', avg_psnr, epoch)
        
        # Log per-variable metrics
        for var_name, metrics in var_metrics.items():
            writer.add_scalar(f'PSNR_pervar/{split}_{var_name}', np.mean(metrics['psnrs']), epoch)
    
    return avg_loss, avg_recon_loss, avg_sw_loss, avg_psnr


def main():
    parser = argparse.ArgumentParser(description='Train SWAE for problematic variables with improved transformations')
    
    # Data arguments
    parser.add_argument('--data-folder', type=str, required=True,
                        help='Path to folder containing HDF5 files')
    parser.add_argument('--target-vars', nargs='+', default=None,
                        help='Specific variables to train on (default: problematic variables)')
    
    # Model arguments
    parser.add_argument('--arch', type=str, default='mlp', choices=['conv', 'mlp', 'gmlp'],
                        help='Architecture type: conv (CNN), mlp, or gmlp')
    parser.add_argument('--latent-dim', type=int, default=8,
                        help='Dimension of latent space')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--lambda-reg', type=float, default=1.0,
                        help='Regularization weight for SW distance')
    
    # Normalization arguments
    parser.add_argument('--normalize-method', type=str, default='auto',
                        choices=['auto', 'symlog', 'asinh', 'robust', 'zscore', 'none'],
                        help='Normalization method (auto uses optimal per-variable)')
    
    # Other arguments
    parser.add_argument('--save-dir', type=str, default='./save/swae_problematic_vars',
                        help='Directory to save models and logs')
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save model every N epochs')
    parser.add_argument('--early-stopping-patience', type=int, default=40,
                        help='Early stopping patience (0 to disable)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create datasets and dataloaders
    print("\nCreating datasets for problematic variables...")
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = \
        create_problematic_variables_datasets(
            args.data_folder,
            train_ratio=0.8,
            val_ratio=0.15,
            normalize=True,
            normalize_method=args.normalize_method,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_vars=args.target_vars
        )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Create model
    print(f"\nCreating SWAE model (architecture: {args.arch})...")
    model = create_swae_3d_5x5x5_model(
        arch=args.arch,
        latent_dim=args.latent_dim,
        lambda_reg=args.lambda_reg
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    # Tensorboard writer
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))
    else:
        writer = None
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    best_val_psnr = float('-inf')
    patience_counter = 0
    
    compression_ratio = calculate_compression_ratio()
    print(f"Theoretical compression ratio: {compression_ratio:.1f}:1")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        start_time = time.time()
        train_loss, train_recon_loss, train_sw_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, writer
        )
        
        # Evaluate
        if epoch % args.eval_interval == 0:
            val_loss, val_recon_loss, val_sw_loss, val_psnr = evaluate(
                model, val_loader, device, val_dataset, epoch, writer, 'val'
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print epoch summary
            print(f'\nEpoch {epoch}/{args.epochs} - Time: {time.time()-start_time:.1f}s')
            print(f'  Train Loss: {train_loss:.6f} (Recon: {train_recon_loss:.6f}, SW: {train_sw_loss:.6f})')
            print(f'  Val Loss: {val_loss:.6f} (Recon: {val_recon_loss:.6f}, SW: {val_sw_loss:.6f})')
            print(f'  Val PSNR: {val_psnr:.2f} dB')
            print(f'  Learning rate: {optimizer.param_groups[0]["lr"]:.2e}')
            
            # Early stopping
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                best_model_path = os.path.join(args.save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_psnr': val_psnr,
                    'args': args,
                }, best_model_path)
                print(f'  💾 Saved best model (PSNR: {val_psnr:.2f} dB)')
            else:
                patience_counter += 1
                if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                    print(f'\n⚠️  Early stopping triggered after {epoch} epochs')
                    break
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'args': args,
            }, checkpoint_path)
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_recon_loss, test_sw_loss, test_psnr = evaluate(
        model, test_loader, device, test_dataset, epoch, writer, 'test'
    )
    
    print(f'\nTest Results:')
    print(f'  Loss: {test_loss:.6f} (Recon: {test_recon_loss:.6f}, SW: {test_sw_loss:.6f})')
    print(f'  PSNR: {test_psnr:.2f} dB')
    print(f'  Compression ratio: {compression_ratio:.1f}:1')
    
    # Save final results
    results_path = os.path.join(args.save_dir, 'final_results.txt')
    with open(results_path, 'w') as f:
        f.write(f'Final Test Results\n')
        f.write(f'==================\n')
        f.write(f'Model: SWAE 3D ({args.arch} architecture)\n')
        f.write(f'Variables: {train_dataset.var_names}\n')
        f.write(f'Normalization: {args.normalize_method}\n')
        f.write(f'Best epoch: {checkpoint["epoch"]}\n')
        f.write(f'Test Loss: {test_loss:.6f}\n')
        f.write(f'Test PSNR: {test_psnr:.2f} dB\n')
        f.write(f'Compression ratio: {compression_ratio:.1f}:1\n')
    
    if writer is not None:
        writer.close()
    
    print(f'\n✅ Training completed! Results saved to {args.save_dir}')


if __name__ == '__main__':
    main()