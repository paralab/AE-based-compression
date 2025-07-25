#!/usr/bin/env python3
"""
Test adaptive scaling to verify it handles all scales properly
"""

import numpy as np
import sys
sys.path.append('.')

from datasets.adaptive_scaling_dataset_5x5x5 import AdaptiveScalingDataset5x5x5

# Test with synthetic data of different scales
def test_scaling():
    print("Testing adaptive scaling with synthetic data...")
    
    # Create test cases with different scales
    test_cases = [
        ("Normal scale", np.random.randn(10, 1, 5, 5, 5)),
        ("Small scale (1e-6)", np.random.randn(10, 1, 5, 5, 5) * 1e-6),
        ("Tiny scale (1e-9)", np.random.randn(10, 1, 5, 5, 5) * 1e-9),
        ("Large scale (1e3)", np.random.randn(10, 1, 5, 5, 5) * 1e3),
        ("Constant field", np.ones((10, 1, 5, 5, 5)) * 3.14e-8),
        ("Near-zero field", np.random.randn(10, 1, 5, 5, 5) * 1e-12),
    ]
    
    target_range = 10.0
    
    for name, data in test_cases:
        print(f"\n{name}:")
        print(f"  Original range: [{data.min():.2e}, {data.max():.2e}]")
        print(f"  Original std: {data.std():.2e}")
        
        # Simulate scaling
        data_abs = np.abs(data)
        data_abs_nonzero = data_abs[data_abs > 1e-12]
        
        if len(data_abs_nonzero) > 0:
            typical_magnitude = np.median(data_abs_nonzero)
            p95 = np.percentile(data_abs, 95)
            scale_factor = target_range / max(p95, 1e-10)
        else:
            typical_magnitude = 1e-12
            scale_factor = 1e10
        
        scaled_data = data * scale_factor
        
        print(f"  Scale factor: {scale_factor:.2e}")
        print(f"  Scaled range: [{scaled_data.min():.2f}, {scaled_data.max():.2f}]")
        print(f"  Scaled std: {scaled_data.std():.2f}")
        
        # Verify denormalization
        denorm_data = scaled_data / scale_factor
        error = np.abs(denorm_data - data).max()
        print(f"  Denormalization error: {error:.2e}")

if __name__ == "__main__":
    test_scaling()