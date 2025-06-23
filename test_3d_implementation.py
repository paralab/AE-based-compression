#!/usr/bin/env python3
"""
Test script for 3D SWAE+Thera implementation
Tests all components: 3D dataset, SWAE encoder, Thera decoder, LIIF-3D
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append('.')

import datasets
import models
from datasets.math_function_3d import MathFunction3DDataset
from datasets.wrappers import MathFunction3DDownsampled
from models.swae import SWAE3D, SWAE3DEncoder
from models.thera_3d import TheraNet3D, TheraNet3DSimple
from models.liif_3d import LIIF3D


def test_3d_math_dataset():
    """Test 3D mathematical function dataset"""
    print("\n=== Testing 3D Math Function Dataset ===")
    
    dataset = MathFunction3DDataset(
        k_values=[2, 3, 4, 5], 
        resolution=40, 
        num_functions=10,
        diagonal_only=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test a sample
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample range: [{sample.min().item():.4f}, {sample.max().item():.4f}]")
    
    # Test superposition
    superpos = MathFunction3DDataset.generate_superposition([(2,2,2), (3,3,3)], resolution=80)
    print(f"Superposition shape: {superpos.shape}")
    
    print("✓ 3D Math Function Dataset test passed!")
    return True


def test_3d_wrapper():
    """Test 3D downsampling wrapper"""
    print("\n=== Testing 3D Downsampling Wrapper ===")
    
    base_dataset = MathFunction3DDataset(
        k_values=[2, 3, 4], 
        resolution=40, 
        num_functions=5,
        diagonal_only=True
    )
    
    wrapper = MathFunction3DDownsampled(
        dataset=base_dataset,
        inp_size=20,
        scale_max=2.0,
        sample_q=1000
    )
    
    sample = wrapper[0]
    print(f"Wrapped sample keys: {list(sample.keys())}")
    print(f"Input shape: {sample['inp'].shape}")
    print(f"Coord shape: {sample['coord'].shape}")
    print(f"Cell shape: {sample['cell'].shape}")
    print(f"GT shape: {sample['gt'].shape}")
    print(f"Scale: {sample['scale']}")
    
    print("✓ 3D Wrapper test passed!")
    return True


def test_swae_3d():
    """Test 3D SWAE model"""
    print("\n=== Testing 3D SWAE Model ===")
    
    # Test full SWAE
    swae = SWAE3D(
        input_channels=1,
        latent_dim=64,
        hidden_channels=[32, 64, 128]
    )
    
    # Test input
    batch_size = 2
    input_data = torch.randn(batch_size, 1, 40, 40, 40)
    
    # Forward pass
    recon, latent = swae(input_data)
    print(f"Input shape: {input_data.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    
    # Test loss computation
    loss_dict = swae.compute_loss(input_data, recon, latent)
    print(f"Loss components: {list(loss_dict.keys())}")
    print(f"Total loss: {loss_dict['total_loss'].item():.6f}")
    
    # Test SWAE encoder for LIIF
    encoder = SWAE3DEncoder(input_channels=1, hidden_channels=[32, 64, 128])
    features = encoder(input_data)
    print(f"Encoder output shape: {features.shape}")
    print(f"Encoder out_dim: {encoder.out_dim}")
    
    print("✓ 3D SWAE test passed!")
    return True


def test_thera_3d():
    """Test 3D Thera model"""
    print("\n=== Testing 3D Thera Model ===")
    
    # Test Thera model
    in_dim = 128 + 3 + 3 + 1  # features + coord + cell + scale
    thera = TheraNet3DSimple(
        in_dim=in_dim,
        out_dim=1,
        hidden_dim=256,
        num_frequencies=64
    )
    
    batch_size = 100
    input_data = torch.randn(batch_size, in_dim)
    
    output = thera(input_data)
    print(f"Thera input shape: {input_data.shape}")
    print(f"Thera output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    print("✓ 3D Thera test passed!")
    return True


def test_liif_3d():
    """Test 3D LIIF model"""
    print("\n=== Testing 3D LIIF Model ===")
    
    # Create LIIF-3D model
    liif_3d = LIIF3D(
        encoder_spec={
            'name': 'swae-encoder-3d',
            'args': {
                'input_channels': 1,
                'hidden_channels': [32, 64, 128],
                'no_upsampling': True
            }
        },
        imnet_spec={
            'name': 'thera-3d-simple',
            'args': {
                'out_dim': 1,
                'hidden_dim': 256,
                'num_frequencies': 64,
                'kappa': 1.0
            }
        }
    )
    
    # Test data
    batch_size = 2
    inp = torch.randn(batch_size, 1, 20, 20, 20)  # Low-res input
    
    # Generate coordinates for 40x40x40 output
    num_queries = 1000
    coord = torch.randn(batch_size, num_queries, 3) * 0.8  # Coordinates in [-1, 1]
    cell = torch.ones(batch_size, num_queries, 3) * 0.05   # Cell size
    scale = torch.tensor([2.0, 2.0, 2.0]).unsqueeze(0).expand(batch_size, -1)  # 2x scale
    
    # Forward pass
    output = liif_3d(inp, coord, cell, scale)
    
    print(f"LIIF-3D input shape: {inp.shape}")
    print(f"Coordinate shape: {coord.shape}")
    print(f"Cell shape: {cell.shape}")
    print(f"Scale shape: {scale.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    print("✓ 3D LIIF test passed!")
    return True


def test_full_pipeline():
    """Test the complete 3D super-resolution pipeline"""
    print("\n=== Testing Full 3D Pipeline ===")
    
    # Create dataset
    base_dataset = MathFunction3DDataset(
        k_values=[2, 3], 
        resolution=40, 
        num_functions=5,
        diagonal_only=True
    )
    
    wrapper = MathFunction3DDownsampled(
        dataset=base_dataset,
        inp_size=20,
        scale_max=2.0,
        sample_q=500
    )
    
    # Create model using the registry
    model_spec = {
        'name': 'liif-3d',
        'args': {
            'encoder_spec': {
                'name': 'swae-encoder-3d',
                'args': {
                    'input_channels': 1,
                    'hidden_channels': [16, 32, 64],  # Smaller for testing
                    'no_upsampling': True
                }
            },
            'imnet_spec': {
                'name': 'thera-3d-simple',
                'args': {
                    'out_dim': 1,
                    'hidden_dim': 128,  # Smaller for testing
                    'num_frequencies': 32,
                    'kappa': 1.0
                }
            }
        }
    }
    
    model = models.make(model_spec)
    
    # Test with a batch
    sample = wrapper[0]
    
    # Add batch dimension
    inp = sample['inp'].unsqueeze(0)
    coord = sample['coord'].unsqueeze(0)
    cell = sample['cell'].unsqueeze(0)
    scale = sample['scale'].unsqueeze(0)
    gt = sample['gt'].unsqueeze(0)
    
    print(f"Pipeline input shape: {inp.shape}")
    print(f"Pipeline coord shape: {coord.shape}")
    print(f"Pipeline GT shape: {gt.shape}")
    
    # Forward pass
    pred = model(inp, coord, cell, scale)
    
    print(f"Pipeline prediction shape: {pred.shape}")
    
    # Compute loss
    loss = torch.nn.functional.l1_loss(pred, gt)
    print(f"L1 Loss: {loss.item():.6f}")
    
    # Test backward pass
    loss.backward()
    print("Backward pass successful!")
    
    print("✓ Full 3D Pipeline test passed!")
    return True


def main():
    """Run all tests"""
    print("Running 3D SWAE+Thera implementation tests...\n")
    
    try:
        # Run individual component tests
        test_3d_math_dataset()
        test_3d_wrapper()
        test_swae_3d()
        test_thera_3d()
        test_liif_3d()
        test_full_pipeline()
        
        print("\n" + "="*60)
        print("🎉 All 3D implementation tests passed!")
        print("You can now start training with:")
        print("sbatch train_swae_thera_3d.sbatch")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 