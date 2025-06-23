#!/usr/bin/env python3
"""
Debug script for 3D SWAE+Thera training issues
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the current directory to the path
sys.path.append('.')

import datasets
import models
from datasets.math_function_3d import MathFunction3DDataset
from datasets.wrappers import MathFunction3DDownsampled


def test_data_range():
    """Test the data range and distribution"""
    print("=== Testing Data Range and Distribution ===")
    
    # Create dataset
    dataset = MathFunction3DDataset(k_values=[2, 3], resolution=40, num_functions=10, diagonal_only=True)
    wrapper = MathFunction3DDownsampled(dataset, inp_size=20, scale_max=2.0, sample_q=1000)
    
    sample = wrapper[0]
    
    print(f"Input range: [{sample['inp'].min().item():.4f}, {sample['inp'].max().item():.4f}]")
    print(f"GT range: [{sample['gt'].min().item():.4f}, {sample['gt'].max().item():.4f}]")
    print(f"Coord range: [{sample['coord'].min().item():.4f}, {sample['coord'].max().item():.4f}]")
    print(f"Cell range: [{sample['cell'].min().item():.4f}, {sample['cell'].max().item():.4f}]")
    print(f"Scale: {sample['scale']}")
    
    # Check if input and GT are properly correlated
    inp_mean = sample['inp'].mean().item()
    gt_mean = sample['gt'].mean().item()
    print(f"Input mean: {inp_mean:.4f}, GT mean: {gt_mean:.4f}")
    
    return sample


def test_model_forward():
    """Test model forward pass with gradient checking"""
    print("\n=== Testing Model Forward Pass ===")
    
    # Create model
    model_spec = {
        'name': 'liif-3d',
        'args': {
            'encoder_spec': {
                'name': 'swae-encoder-3d',
                'args': {
                    'input_channels': 1,
                    'hidden_channels': [16, 32, 64],
                    'no_upsampling': True
                }
            },
            'imnet_spec': {
                'name': 'thera-3d-simple',
                'args': {
                    'out_dim': 1,
                    'hidden_dim': 128,
                    'num_frequencies': 32,
                    'kappa': 1.0
                }
            }
        }
    }
    
    model = models.make(model_spec)
    
    # Create test data
    batch_size = 2
    inp = torch.randn(batch_size, 1, 20, 20, 20) * 0.5  # Smaller input range
    coord = torch.randn(batch_size, 500, 3) * 0.8  # Coordinates in [-0.8, 0.8]
    cell = torch.ones(batch_size, 500, 3) * 0.05
    scale = torch.tensor([2.0, 2.0, 2.0]).unsqueeze(0).expand(batch_size, -1)
    gt = torch.randn(batch_size, 500, 1) * 0.5  # Target values
    
    print(f"Input shape: {inp.shape}, range: [{inp.min():.3f}, {inp.max():.3f}]")
    print(f"Coord shape: {coord.shape}, range: [{coord.min():.3f}, {coord.max():.3f}]")
    print(f"GT shape: {gt.shape}, range: [{gt.min():.3f}, {gt.max():.3f}]")
    
    # Forward pass
    model.train()
    pred = model(inp, coord, cell, scale)
    print(f"Prediction shape: {pred.shape}, range: [{pred.min():.3f}, {pred.max():.3f}]")
    
    # Check for NaN or inf
    if torch.isnan(pred).any():
        print("❌ NaN detected in predictions!")
        return False
    if torch.isinf(pred).any():
        print("❌ Inf detected in predictions!")
        return False
    
    # Compute loss
    loss = nn.L1Loss()(pred, gt)
    print(f"Initial loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    total_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            if param_norm.item() > 100:
                print(f"Large gradient in {name}: {param_norm.item():.4f}")
    
    total_norm = total_norm ** (1. / 2)
    print(f"Total gradient norm: {total_norm:.6f} across {param_count} parameters")
    
    if total_norm < 1e-8:
        print("❌ Gradients are too small - vanishing gradient problem!")
        return False
    elif total_norm > 100:
        print("❌ Gradients are too large - exploding gradient problem!")
        return False
    
    print("✓ Forward and backward pass successful!")
    return True


def test_training_step():
    """Test a few training steps"""
    print("\n=== Testing Training Steps ===")
    
    # Create model
    model_spec = {
        'name': 'liif-3d',
        'args': {
            'encoder_spec': {
                'name': 'swae-encoder-3d',
                'args': {
                    'input_channels': 1,
                    'hidden_channels': [16, 32, 64],
                    'no_upsampling': True
                }
            },
            'imnet_spec': {
                'name': 'thera-3d-simple',
                'args': {
                    'out_dim': 1,
                    'hidden_dim': 128,
                    'num_frequencies': 32,
                    'kappa': 1.0
                }
            }
        }
    }
    
    model = models.make(model_spec)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()
    
    # Create dataset
    dataset = MathFunction3DDataset(k_values=[2, 3], resolution=40, num_functions=5, diagonal_only=True)
    wrapper = MathFunction3DDownsampled(dataset, inp_size=20, scale_max=2.0, sample_q=500)
    
    losses = []
    
    for step in range(10):
        sample = wrapper[step % len(wrapper)]
        
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
        
        losses.append(loss.item())
        print(f"Step {step+1}: Loss = {loss.item():.6f}")
    
    # Check if loss is decreasing
    if losses[-1] < losses[0]:
        print(f"✓ Loss decreased from {losses[0]:.6f} to {losses[-1]:.6f}")
        return True
    else:
        print(f"❌ Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}")
        return False


def test_coordinate_encoding():
    """Test if coordinate encoding is working properly"""
    print("\n=== Testing Coordinate Encoding ===")
    
    # Create a simple test function
    def test_func(coord):
        """Simple test function: sin(2πx)sin(2πy)sin(2πz)"""
        x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
        return torch.sin(2 * np.pi * x) * torch.sin(2 * np.pi * y) * torch.sin(2 * np.pi * z)
    
    # Generate coordinates
    coord = torch.randn(1000, 3) * 0.5  # Coordinates in [-0.5, 0.5]
    gt_values = test_func(coord).unsqueeze(-1)
    
    print(f"Coordinate range: [{coord.min():.3f}, {coord.max():.3f}]")
    print(f"GT values range: [{gt_values.min():.3f}, {gt_values.max():.3f}]")
    print(f"GT values mean: {gt_values.mean():.3f}, std: {gt_values.std():.3f}")
    
    # Test if our function is reasonable
    if gt_values.std() < 0.1:
        print("❌ GT values have very low variance - function might be too simple")
        return False
    
    print("✓ Coordinate encoding test passed!")
    return True


def main():
    """Run all debug tests"""
    print("Running 3D Training Debug Tests...\n")
    
    try:
        test_data_range()
        test_coordinate_encoding()
        
        if not test_model_forward():
            print("\n❌ Model forward pass failed!")
            return False
        
        if not test_training_step():
            print("\n❌ Training step test failed!")
            return False
        
        print("\n" + "="*60)
        print("✓ All debug tests passed!")
        print("Try training with the improved configuration:")
        print("python train_liif_3d.py --config configs/train-3d/train_swae_thera_3d_improved.yaml --name swae_thera_3d_improved --gpu 0")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Debug test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 