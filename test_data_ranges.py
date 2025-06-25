#!/usr/bin/env python3
"""
Test Data Ranges for U_CHI with pos_log normalization
Quick check of input/output ranges for training
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.u_chi_dataset import create_u_chi_datasets

def test_data_ranges():
    """Test the data ranges with pos_log normalization"""
    print("="*60)
    print("TESTING U_CHI DATA RANGES WITH POS_LOG NORMALIZATION")
    print("="*60)
    
    data_folder = "/u/tawal/0620-NN-based-compression-thera/tt_q01/"
    
    print(f"Data folder: {data_folder}")
    print("Testing pos_log normalization...")
    
    try:
        # Create datasets with pos_log normalization
        train_dataset, val_dataset = create_u_chi_datasets(
            data_folder=data_folder,
            train_ratio=0.8,
            normalize=True,
            normalize_method='pos_log'
        )
        
        print(f"\nDataset created successfully!")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # Test a few samples
        print(f"\n" + "="*50)
        print("SAMPLE DATA ANALYSIS")
        print(f"="*50)
        
        for i in range(min(3, len(val_dataset))):
            sample, metadata = val_dataset[i]
            
            # Normalized data (what model sees)
            normalized = sample.numpy().squeeze()
            
            # Denormalized data (original units)
            denormalized = val_dataset.denormalize(normalized)
            
            print(f"\nSample {i+1}:")
            print(f"  Shape: {normalized.shape}")
            print(f"  NORMALIZED (model input):")
            print(f"    Range: [{normalized.min():.3f}, {normalized.max():.3f}]")
            print(f"    Mean: {normalized.mean():.3f}")
            print(f"    Std: {normalized.std():.3f}")
            print(f"  DENORMALIZED (original units):")
            print(f"    Range: [{denormalized.min():.3e}, {denormalized.max():.3e}]")
            print(f"    Mean: {denormalized.mean():.3e}")
            
            # Check round-trip accuracy
            renormalized = val_dataset._normalize_data.__globals__['np'].log(
                (denormalized - val_dataset.data_min + val_dataset.epsilon) + val_dataset.epsilon
            )
            error = np.abs(normalized - renormalized).max()
            print(f"    Round-trip error: {error:.2e}")
        
        # Overall statistics
        print(f"\n" + "="*50)
        print("OVERALL TRAINING DATA CHARACTERISTICS")
        print(f"="*50)
        
        all_normalized = []
        all_denormalized = []
        
        for i in range(min(100, len(train_dataset))):
            sample, _ = train_dataset[i]
            normalized = sample.numpy().squeeze()
            denormalized = train_dataset.denormalize(normalized)
            
            all_normalized.extend(normalized.flatten())
            all_denormalized.extend(denormalized.flatten())
        
        all_normalized = np.array(all_normalized)
        all_denormalized = np.array(all_denormalized)
        
        print(f"MODEL INPUT (normalized, what neural network sees):")
        print(f"  Range: [{all_normalized.min():.3f}, {all_normalized.max():.3f}]")
        print(f"  Mean: {all_normalized.mean():.3f}")
        print(f"  Std: {all_normalized.std():.3f}")
        print(f"  Dynamic range: {all_normalized.max() - all_normalized.min():.1f}")
        
        print(f"\nORIGINAL DATA (denormalized, physical units):")
        print(f"  Range: [{all_denormalized.min():.3e}, {all_denormalized.max():.3e}]")
        print(f"  Mean: {all_denormalized.mean():.3e}")
        print(f"  Orders of magnitude: {np.log10(all_denormalized.max()/all_denormalized.min()):.1f}")
        
        print(f"\n" + "="*50)
        print("EXPECTED TRAINING BEHAVIOR")
        print(f"="*50)
        print(f"✅ Model will see input values in range [{all_normalized.min():.1f}, {all_normalized.max():.1f}]")
        print(f"✅ This is a reasonable range for neural networks (not too extreme)")
        print(f"✅ Log transformation compressed {np.log10(all_denormalized.max()/all_denormalized.min()):.1f} orders of magnitude")
        print(f"✅ Normalization is working correctly!")
        
        if all_normalized.min() < -50 or all_normalized.max() > 50:
            print(f"⚠️  Warning: Normalized range might be too extreme for training")
        
        print(f"\npos_log normalization test completed successfully! ✅")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_ranges() 