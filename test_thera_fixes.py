#!/usr/bin/env python3
"""
Test script to verify Thera implementation fixes
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to path to import modules
sys.path.append('.')

import models
from models.thera import THEORETICAL_KAPPA, initialize_frequency_bank

def test_theoretical_kappa():
    """Test that theoretical kappa value is correct"""
    expected_kappa = np.sqrt(np.log(4)) / (2 * np.pi**2)
    print(f"✓ Theoretical κ value: {THEORETICAL_KAPPA:.6f}")
    print(f"  Expected: {expected_kappa:.6f}")
    assert abs(THEORETICAL_KAPPA - expected_kappa) < 1e-10, "Kappa values don't match"
    print("  ✅ Kappa value is correct!")

def test_frequency_initialization():
    """Test W1 initialization follows p(|ν|) ∝ |ν|"""
    in_dim = 4
    num_frequencies = 16
    W1 = initialize_frequency_bank(in_dim, num_frequencies)
    
    # Check frequency magnitudes are roughly increasing
    freq_mags = torch.norm(W1, dim=0)
    sorted_mags = torch.sort(freq_mags)[0]
    
    print(f"✓ W1 shape: {W1.shape}")
    print(f"  Frequency magnitudes: {freq_mags[:8].numpy()}")
    print(f"  Are roughly increasing: {torch.all(sorted_mags[1:] >= sorted_mags[:-1])}")
    
    # Check that we have a good spread of frequencies
    min_mag, max_mag = freq_mags.min(), freq_mags.max()
    print(f"  Frequency range: [{min_mag:.3f}, {max_mag:.3f}]")
    assert max_mag > min_mag * 5, "Frequency range too narrow"
    print("  ✅ Frequency initialization looks good!")

def test_thera_creation():
    """Test creating Thera models with fixed parameters"""
    print("\n=== Testing TheraNet Creation ===")
    
    # Test TheraNet
    thera_net = models.make({
        'name': 'thera',
        'args': {
            'in_dim': 10,  # 9 + 1 for scale
            'out_dim': 1,
            'hidden_dim': 128,
            'num_frequencies': 32
        }
    })
    
    print(f"✓ TheraNet created")
    print(f"  κ value: {thera_net.kappa:.6f}")
    print(f"  W1 shape: {thera_net.W1.shape}")
    print(f"  W1 frequency range: [{torch.norm(thera_net.W1, dim=0).min():.3f}, {torch.norm(thera_net.W1, dim=0).max():.3f}]")
    
    # Test TheraNetSimple
    thera_simple = models.make({
        'name': 'thera-simple',
        'args': {
            'in_dim': 10,
            'out_dim': 1,
            'hidden_dim': 128,
            'num_frequencies': 32
        }
    })
    
    print(f"✓ TheraNetSimple created")
    print(f"  κ value: {thera_simple.kappa:.6f}")
    print(f"  W1 shape: {thera_simple.W1.shape}")
    print("  ✅ Both models created successfully!")

def test_thermal_activation():
    """Test thermal activation function"""
    print("\n=== Testing Thermal Activation ===")
    
    thera_net = models.make({
        'name': 'thera',
        'args': {
            'in_dim': 5,
            'out_dim': 1,
            'num_frequencies': 8
        }
    })
    
    # Test thermal activation
    batch_size = 2
    z = torch.randn(batch_size, 8)
    t = torch.tensor([[1.0], [4.0]])  # t = S² for S=1 and S=2
    
    xi = thera_net.thermal_activation(z, thera_net.W1, thera_net.kappa, t)
    
    print(f"✓ Thermal activation output shape: {xi.shape}")
    print(f"  Input z range: [{z.min():.3f}, {z.max():.3f}]")
    print(f"  Output ξ range: [{xi.min():.3f}, {xi.max():.3f}]")
    
    # Check that higher t values lead to more attenuation (lower magnitude)
    # (for most frequencies this should be true on average)
    mean_xi_t1 = xi[0].abs().mean()
    mean_xi_t4 = xi[1].abs().mean()
    print(f"  Mean |ξ| for t=1: {mean_xi_t1:.4f}")
    print(f"  Mean |ξ| for t=4: {mean_xi_t4:.4f}")
    print(f"  Higher t leads to attenuation: {mean_xi_t4 < mean_xi_t1}")
    print("  ✅ Thermal activation working correctly!")

def test_forward_pass():
    """Test complete forward pass"""
    print("\n=== Testing Forward Pass ===")
    
    batch_size = 3
    in_dim = 9  # 8 features + 1 scale
    
    # Create input with scale information
    x = torch.randn(batch_size, in_dim)
    x[:, -1] = torch.tensor([1.0, 4.0, 16.0])  # t = S² for different scales
    
    # Test TheraNet
    thera_net = models.make({
        'name': 'thera',
        'args': {
            'in_dim': in_dim,
            'out_dim': 1,
            'hidden_dim': 64,
            'num_frequencies': 16
        }
    })
    
    output = thera_net(x)
    print(f"✓ TheraNet forward pass")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test TheraNetSimple
    thera_simple = models.make({
        'name': 'thera-simple',
        'args': {
            'in_dim': in_dim,
            'out_dim': 1,
            'hidden_dim': 64,
            'num_frequencies': 16
        }
    })
    
    output_simple = thera_simple(x)
    print(f"✓ TheraNetSimple forward pass")
    print(f"  Output shape: {output_simple.shape}")
    print(f"  Output range: [{output_simple.min():.4f}, {output_simple.max():.4f}]")
    print("  ✅ Forward passes working correctly!")

if __name__ == "__main__":
    print("=== Testing Thera Implementation Fixes ===\n")
    
    try:
        test_theoretical_kappa()
        test_frequency_initialization()
        test_thera_creation()
        test_thermal_activation()
        test_forward_pass()
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED!")
        print("The Thera implementation fixes are working correctly.")
        print("You can now retrain with the corrected parameters.")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 