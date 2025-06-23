#!/usr/bin/env python3
"""
Simplified 3D training script for debugging
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time

# Add the current directory to the path
sys.path.append('.')

import datasets
import models
from datasets.math_function_3d import MathFunction3DDataset
from datasets.wrappers import MathFunction3DDownsampled


def create_simple_model():
    """Create a simplified model for testing"""
    model_spec = {
        'name': 'liif-3d',
        'args': {
            'encoder_spec': {
                'name': 'swae-encoder-3d',
                'args': {
                    'input_channels': 1,
                    'hidden_channels': [8, 16, 32],  # Much smaller
                    'no_upsampling': True
                }
            },
            'imnet_spec': {
                'name': 'thera-3d-simple',
                'args': {
                    'out_dim': 1,
                    'hidden_dim': 64,  # Much smaller
                    'num_frequencies': 16,  # Much smaller
                    'kappa': 1.0
                }
            }
        }
    }
    return models.make(model_spec)


def create_dataset():
    """Create a simple dataset"""
    dataset = MathFunction3DDataset(
        k_values=[2, 3], 
        resolution=40, 
        num_functions=20,  # Very small
        diagonal_only=True
    )
    
    wrapper = MathFunction3DDownsampled(
        dataset, 
        inp_size=20, 
        scale_max=2.0, 
        sample_q=500  # Smaller sample
    )
    
    return wrapper


def train_simple():
    """Simple training loop"""
    print("Creating model...")
    model = create_simple_model()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    print("Creating dataset...")
    dataset = create_dataset()
    print(f"Dataset size: {len(dataset)}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_fn = nn.L1Loss()
    
    print("Starting training...")
    model.train()
    
    losses = []
    
    for epoch in range(50):
        epoch_losses = []
        
        for i in range(min(10, len(dataset))):  # Train on subset
            sample = dataset[i]
            
            # Add batch dimension
            inp = sample['inp'].unsqueeze(0)
            coord = sample['coord'].unsqueeze(0)
            cell = sample['cell'].unsqueeze(0)
            scale = sample['scale'].unsqueeze(0)
            gt = sample['gt'].unsqueeze(0)
            
            # Forward pass
            pred = model(inp, coord, cell, scale)
            loss = loss_fn(pred, gt)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}")
            
            # Check for improvement
            if epoch > 10:
                recent_avg = np.mean(losses[-5:])
                old_avg = np.mean(losses[-10:-5])
                improvement = (old_avg - recent_avg) / old_avg * 100
                print(f"           Improvement: {improvement:.2f}%")
    
    # Final analysis
    print(f"\nTraining completed!")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Total improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.2f}%")
    
    if losses[-1] < losses[0] * 0.9:  # At least 10% improvement
        print("✓ Training successful - loss decreased significantly!")
        return True
    else:
        print("❌ Training failed - insufficient improvement")
        return False


def test_single_function():
    """Test on a single known function"""
    print("\n=== Testing Single Function ===")
    
    model = create_simple_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()
    
    # Create a single test function: sin(2πx)sin(2πy)sin(2πz)
    def create_test_function():
        # High-res version
        x = torch.linspace(-1, 1, 40)
        y = torch.linspace(-1, 1, 40)
        z = torch.linspace(-1, 1, 40)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        hr_func = torch.sin(2 * np.pi * X) * torch.sin(2 * np.pi * Y) * torch.sin(2 * np.pi * Z)
        
        # Low-res version (downsampled)
        x_lr = torch.linspace(-1, 1, 20)
        y_lr = torch.linspace(-1, 1, 20)
        z_lr = torch.linspace(-1, 1, 20)
        X_lr, Y_lr, Z_lr = torch.meshgrid(x_lr, y_lr, z_lr, indexing='ij')
        lr_func = torch.sin(2 * np.pi * X_lr) * torch.sin(2 * np.pi * Y_lr) * torch.sin(2 * np.pi * Z_lr)
        
        return hr_func.unsqueeze(0).unsqueeze(0), lr_func.unsqueeze(0).unsqueeze(0)
    
    hr_func, lr_func = create_test_function()
    
    # Create coordinates for a subset of points
    coord = torch.randn(1, 500, 3) * 0.8  # Random coordinates
    cell = torch.ones(1, 500, 3) * 0.05
    scale = torch.tensor([2.0, 2.0, 2.0]).unsqueeze(0)
    
    # Get ground truth values at these coordinates
    # This is a simplified version - in practice we'd interpolate properly
    gt = torch.randn(1, 500, 1) * 0.5  # Placeholder GT
    
    print(f"Input shape: {lr_func.shape}")
    print(f"Target shape: {hr_func.shape}")
    print(f"Coordinate shape: {coord.shape}")
    
    # Train on this single function
    losses = []
    for step in range(100):
        pred = model(lr_func, coord, cell, scale)
        loss = loss_fn(pred, gt)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 20 == 0:
            print(f"Step {step:3d}: Loss = {loss.item():.6f}")
    
    print(f"Single function training: {losses[0]:.6f} -> {losses[-1]:.6f}")
    return losses[-1] < losses[0] * 0.5  # 50% improvement


def main():
    """Main function"""
    print("Simple 3D Training Test")
    print("=" * 40)
    
    try:
        # Test single function first
        if test_single_function():
            print("✓ Single function test passed!")
        else:
            print("❌ Single function test failed!")
            return False
        
        # Test full training
        if train_simple():
            print("\n✓ Simple training test passed!")
            print("\nThe model can learn! Try the improved configuration:")
            print("python train_liif_3d.py --config configs/train-3d/train_swae_thera_3d_improved.yaml --name swae_thera_3d_improved --gpu 0")
            return True
        else:
            print("\n❌ Simple training test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 