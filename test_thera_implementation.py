#!/usr/bin/env python3

import torch
import numpy as np
import yaml

import models
import datasets
from utils import make_coord


def test_thera_model():
    """Test Thera model implementation"""
    print("Testing Thera model implementation...")
    
    # Test parameters
    batch_size = 2
    num_queries = 1000
    in_dim = 256 + 9 + 2 + 2 + 1  # encoder_dim + unfold + coord + cell + scale
    out_dim = 1
    
    # Create Thera model
    thera_model = models.make({
        'name': 'thera-simple',
        'args': {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'hidden_dim': 256,
            'num_frequencies': 64,
            'kappa': 1.0
        }
    })
    
    print(f"Thera model created with {sum(p.numel() for p in thera_model.parameters())} parameters")
    
    # Test forward pass
    x = torch.randn(batch_size * num_queries, in_dim)
    
    with torch.no_grad():
        output = thera_model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    assert output.shape == (batch_size * num_queries, out_dim), f"Expected shape {(batch_size * num_queries, out_dim)}, got {output.shape}"
    print("✓ Thera model test passed!")
    
    return thera_model


def test_liif_with_thera():
    """Test LIIF model with Thera decoder"""
    print("\nTesting LIIF with Thera decoder...")
    
    # Create LIIF model with Thera decoder
    liif_model = models.make({
        'name': 'liif',
        'args': {
            'encoder_spec': {
                'name': 'edsr-baseline',
                'args': {
                    'no_upsampling': True,
                    'n_colors': 1
                }
            },
            'imnet_spec': {
                'name': 'thera-simple',
                'args': {
                    'out_dim': 1,
                    'hidden_dim': 256,
                    'num_frequencies': 64,
                    'kappa': 1.0
                }
            }
        }
    })
    
    print(f"LIIF+Thera model created with {sum(p.numel() for p in liif_model.parameters())} parameters")
    
    # Test with sample data
    batch_size = 2
    lr_size = 48
    hr_size = 192  # 4x scale
    num_queries = 2304
    
    # Create sample input
    inp = torch.randn(batch_size, 1, lr_size, lr_size)
    coord = make_coord((hr_size, hr_size)).unsqueeze(0).expand(batch_size, -1, -1)
    coord = coord[:, :num_queries, :]  # Sample subset
    
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / hr_size
    cell[:, :, 1] *= 2 / hr_size
    
    scale = torch.tensor([4.0, 4.0])  # 4x scale factor
    
    # Test forward pass
    with torch.no_grad():
        output = liif_model(inp, coord, cell, scale)
    
    print(f"Input shape: {inp.shape}")
    print(f"Coord shape: {coord.shape}")
    print(f"Cell shape: {cell.shape}")
    print(f"Scale shape: {scale.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    expected_shape = (batch_size, num_queries, 1)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    print("✓ LIIF+Thera model test passed!")
    
    return liif_model


def test_dataset_with_scale():
    """Test dataset wrapper with scale information"""
    print("\nTesting dataset with scale information...")
    
    # Create dataset
    dataset = datasets.make({
        'name': 'math-function',
        'args': {
            'k_values': [2, 3, 4, 5],
            'resolution': 256,
            'num_functions': 10,
            'repeat': 1,
            'cache': 'in_memory'
        }
    })
    
    # Create wrapper
    wrapper = datasets.make({
        'name': 'math-function-downsampled',
        'args': {
            'dataset': dataset,
            'inp_size': 48,
            'scale_max': 4,
            'augment': False,
            'sample_q': 100
        }
    })
    
    print(f"Dataset created with {len(wrapper)} samples")
    
    # Test sample
    sample = wrapper[0]
    
    print("Sample keys:", list(sample.keys()))
    print(f"Input shape: {sample['inp'].shape}")
    print(f"Coord shape: {sample['coord'].shape}")
    print(f"Cell shape: {sample['cell'].shape}")
    print(f"GT shape: {sample['gt'].shape}")
    print(f"Scale: {sample['scale'].item():.4f}")
    
    assert 'scale' in sample, "Scale information missing from dataset sample"
    assert sample['scale'].dim() == 0, f"Scale should be scalar, got shape {sample['scale'].shape}"
    print("✓ Dataset with scale test passed!")
    
    return wrapper


def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    
    # Create model
    model = models.make({
        'name': 'liif',
        'args': {
            'encoder_spec': {
                'name': 'edsr-baseline',
                'args': {
                    'no_upsampling': True,
                    'n_colors': 1
                }
            },
            'imnet_spec': {
                'name': 'thera-simple',
                'args': {
                    'out_dim': 1,
                    'hidden_dim': 128,  # Smaller for testing
                    'num_frequencies': 32,
                    'kappa': 1.0
                }
            }
        }
    })
    
    # Create sample batch
    batch_size = 2
    sample_q = 100
    
    batch = {
        'inp': torch.randn(batch_size, 1, 48, 48),
        'coord': torch.randn(batch_size, sample_q, 2),
        'cell': torch.ones(batch_size, sample_q, 2) * 0.01,
        'gt': torch.randn(batch_size, sample_q, 1),
        'scale': torch.tensor([2.5, 3.2])
    }
    
    # Test forward pass
    model.train()
    pred = model(batch['inp'], batch['coord'], batch['cell'], batch['scale'])
    
    # Test loss computation
    loss_fn = torch.nn.L1Loss()
    loss = loss_fn(pred, batch['gt'])
    
    print(f"Prediction shape: {pred.shape}")
    print(f"GT shape: {batch['gt'].shape}")
    print(f"Loss: {loss.item():.6f}")
    
    # Test backward pass
    loss.backward()
    
    print("✓ Training step test passed!")


def main():
    """Run all tests"""
    print("Running Thera implementation tests...\n")
    
    try:
        # Test individual components
        test_thera_model()
        test_liif_with_thera()
        test_dataset_with_scale()
        test_training_step()
        
        print("\n" + "="*50)
        print("🎉 All tests passed! Thera implementation is ready.")
        print("You can now start training with:")
        print("sbatch train_thera_math.sbatch")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    main() 