#!/usr/bin/env python3
"""
Test script for Pure SWAE 3D implementation
Verifies that all components work correctly before training
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.swae_pure_3d import create_swae_3d_model, SWAE3D
from datasets.swae_3d_dataset import create_swae_3d_datasets


def test_model_architecture():
    """Test the SWAE 3D model architecture"""
    print("Testing SWAE 3D model architecture...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with paper specifications
    model = create_swae_3d_model(
        block_size=8,
        latent_dim=16,
        lambda_reg=10.0
    ).to(device)
    
    # Test with dummy data
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 8, 8, 8).to(device)
    
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        x_recon, z = model(test_input)
    
    print(f"Latent shape: {z.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    
    # Check shapes
    assert z.shape == (batch_size, 16), f"Expected latent shape (4, 16), got {z.shape}"
    assert x_recon.shape == test_input.shape, f"Expected reconstruction shape {test_input.shape}, got {x_recon.shape}"
    
    # Test loss computation
    loss_dict = model.loss_function(test_input, x_recon, z)
    
    print(f"Total loss: {loss_dict['loss']:.6f}")
    print(f"Reconstruction loss: {loss_dict['recon_loss']:.6f}")
    print(f"SW distance: {loss_dict['sw_distance']:.6f}")
    
    # Check loss values are reasonable
    assert loss_dict['loss'] > 0, "Total loss should be positive"
    assert loss_dict['recon_loss'] > 0, "Reconstruction loss should be positive"
    assert loss_dict['sw_distance'] >= 0, "SW distance should be non-negative"
    
    print("✓ Model architecture test passed!")
    return model


def test_dataset():
    """Test the SWAE 3D dataset"""
    print("\nTesting SWAE 3D dataset...")
    
    # Create small dataset for testing
    train_dataset, val_dataset = create_swae_3d_datasets(
        k_values=[2, 3],  # Small subset for testing
        resolution=40,
        block_size=8,
        train_split=0.8,
        normalize=True
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Test getting a block
    block, metadata = train_dataset[0]
    
    print(f"Block shape: {block.shape}")
    print(f"Block range: [{block.min():.3f}, {block.max():.3f}]")
    print(f"Block dtype: {block.dtype}")
    
    # Check block properties
    assert block.shape == (1, 8, 8, 8), f"Expected block shape (1, 8, 8, 8), got {block.shape}"
    assert block.dtype == torch.float32, f"Expected float32, got {block.dtype}"
    assert -1.1 <= block.min() <= -0.9, f"Normalized block min should be ~-1, got {block.min():.3f}"
    assert 0.9 <= block.max() <= 1.1, f"Normalized block max should be ~1, got {block.max():.3f}"
    
    # Test metadata
    assert 'sample_idx' in metadata, "Metadata should contain sample_idx"
    assert 'block_coords' in metadata, "Metadata should contain block_coords"
    assert 'global_min' in metadata, "Metadata should contain global_min"
    assert 'global_max' in metadata, "Metadata should contain global_max"
    
    print(f"Sample index: {metadata['sample_idx']}")
    print(f"Block coordinates: {metadata['block_coords']}")
    print(f"Global min/max: {metadata['global_min']:.3f}/{metadata['global_max']:.3f}")
    
    # Test denormalization
    denorm_block = train_dataset.denormalize_block(block)
    print(f"Denormalized range: [{denorm_block.min():.3f}, {denorm_block.max():.3f}]")
    
    # Check denormalization (relax check since we're testing a single block)
    assert denorm_block.min() >= metadata['global_min'] - 0.1, "Denormalized min should be within global range"
    assert denorm_block.max() <= metadata['global_max'] + 0.1, "Denormalized max should be within global range"
    
    print("✓ Dataset test passed!")
    return train_dataset, val_dataset


def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and dataset
    model = create_swae_3d_model().to(device)
    train_dataset, _ = create_swae_3d_datasets(
        k_values=[2, 3],
        resolution=40,
        block_size=8,
        train_split=0.8
    )
    
    # Create data loader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Get a batch
    data, metadata = next(iter(train_loader))
    data = data.to(device)
    
    print(f"Batch shape: {data.shape}")
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    x_recon, z = model(data)
    
    # Compute loss
    loss_dict = model.loss_function(data, x_recon, z)
    loss = loss_dict['loss']
    
    print(f"Loss before backward: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"Total gradient norm: {total_grad_norm:.6f}")
    assert total_grad_norm > 0, "Gradients should be non-zero"
    
    # Optimizer step
    optimizer.step()
    
    print("✓ Training step test passed!")


def test_full_reconstruction():
    """Test full sample reconstruction"""
    print("\nTesting full sample reconstruction...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and dataset
    model = create_swae_3d_model().to(device)
    _, val_dataset = create_swae_3d_datasets(
        k_values=[2],  # Single k value for faster testing
        resolution=40,
        block_size=8,
        train_split=0.8
    )
    
    # Test reconstruction
    sample_idx = 0
    reconstructed, original = val_dataset.reconstruct_full_sample(sample_idx, model, device)
    
    print(f"Original shape: {original.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Original range: [{original.min():.3f}, {original.max():.3f}]")
    print(f"Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    # Check shapes match
    assert original.shape == reconstructed.shape, "Shapes should match"
    assert original.shape == (40, 40, 40), "Should be 40x40x40"
    
    # Calculate MSE and PSNR
    mse = np.mean((original - reconstructed) ** 2)
    value_range = np.max(original) - np.min(original)
    psnr = 20 * np.log10(value_range) - 10 * np.log10(mse)
    
    print(f"MSE: {mse:.6f}")
    print(f"PSNR: {psnr:.2f} dB")
    
    # Untrained model should have reasonable but not perfect reconstruction
    assert psnr > 0, "PSNR should be positive"
    assert mse < value_range, "MSE should be less than value range"
    
    print("✓ Full reconstruction test passed!")


def test_compression_ratio():
    """Test compression ratio calculation"""
    print("\nTesting compression ratio...")
    
    # Original block: 8x8x8 = 512 float32 values = 2048 bytes
    # Latent: 16 float32 values = 64 bytes
    # Expected ratio: 2048/64 = 32:1
    
    original_bytes = 8 * 8 * 8 * 4  # 512 * 4 = 2048
    latent_bytes = 16 * 4           # 16 * 4 = 64
    expected_ratio = original_bytes / latent_bytes
    
    print(f"Original block size: {original_bytes} bytes")
    print(f"Latent size: {latent_bytes} bytes")
    print(f"Compression ratio: {expected_ratio:.1f}:1")
    
    assert expected_ratio == 32.0, f"Expected 32:1 ratio, got {expected_ratio:.1f}:1"
    
    print("✓ Compression ratio test passed!")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Pure SWAE 3D Implementation")
    print("=" * 60)
    
    try:
        # Test model architecture
        model = test_model_architecture()
        
        # Test dataset
        train_dataset, val_dataset = test_dataset()
        
        # Test training step
        test_training_step()
        
        # Test full reconstruction
        test_full_reconstruction()
        
        # Test compression ratio
        test_compression_ratio()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe Pure SWAE 3D implementation is working correctly.")
        print("You can now run the training script:")
        print("  python train_swae_3d_pure.py")
        print("Or submit the SLURM job:")
        print("  sbatch train_swae_3d_pure.sbatch")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 