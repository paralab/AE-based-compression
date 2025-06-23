import torch
import numpy as np
from torch.utils.data import Dataset
from .math_function_3d import MathFunction3DDataset


class SWAE3DBlockDataset(Dataset):
    """
    Dataset for SWAE 3D that splits 3D mathematical functions into 8x8x8 blocks
    Following the paper's approach of block-wise processing for scientific data
    """
    
    def __init__(self, base_dataset, block_size=8, normalize=True):
        """
        Args:
            base_dataset: MathFunction3DDataset instance
            block_size: Size of blocks (8x8x8 as per paper)
            normalize: Whether to normalize data to [-1, 1] as mentioned in paper
        """
        self.base_dataset = base_dataset
        self.block_size = block_size
        self.normalize = normalize
        
        # Get data dimensions
        sample_data = base_dataset[0]
        # Remove channel dimension if present: (1, 40, 40, 40) -> (40, 40, 40)
        if sample_data.dim() == 4 and sample_data.shape[0] == 1:
            sample_data = sample_data.squeeze(0)
        self.data_shape = sample_data.shape  # Should be (40, 40, 40)
        
        # Calculate number of blocks per dimension
        self.blocks_per_dim = self.data_shape[0] // block_size
        self.total_blocks_per_sample = self.blocks_per_dim ** 3
        
        # Calculate normalization parameters if needed
        if self.normalize:
            self._calculate_normalization_params()
        
        print(f"Dataset initialized:")
        print(f"  Data shape: {self.data_shape}")
        print(f"  Block size: {block_size}")
        print(f"  Blocks per dimension: {self.blocks_per_dim}")
        print(f"  Total blocks per sample: {self.total_blocks_per_sample}")
        print(f"  Total samples: {len(base_dataset)}")
        print(f"  Total blocks: {len(self)}")
    
    def _calculate_normalization_params(self):
        """Calculate global min/max for normalization as mentioned in paper"""
        print("Calculating normalization parameters...")
        
        all_values = []
        for i in range(len(self.base_dataset)):
            data = self.base_dataset[i]
            # Remove channel dimension if present
            if data.dim() == 4 and data.shape[0] == 1:
                data = data.squeeze(0)
            all_values.append(data.flatten())
        
        all_values = np.concatenate(all_values)
        self.global_min = float(np.min(all_values))
        self.global_max = float(np.max(all_values))
        
        print(f"  Global min: {self.global_min:.6f}")
        print(f"  Global max: {self.global_max:.6f}")
    
    def __len__(self):
        """Total number of blocks across all samples"""
        return len(self.base_dataset) * self.total_blocks_per_sample
    
    def __getitem__(self, idx):
        """
        Get a single 8x8x8 block
        
        Returns:
            block: torch.Tensor of shape (1, 8, 8, 8) - normalized to [-1, 1]
            metadata: dict with sample_idx, block_coords, and normalization info
        """
        # Calculate which sample and which block within that sample
        sample_idx = idx // self.total_blocks_per_sample
        block_idx = idx % self.total_blocks_per_sample
        
        # Get the full sample
        data = self.base_dataset[sample_idx]
        # Remove channel dimension if present
        if data.dim() == 4 and data.shape[0] == 1:
            data = data.squeeze(0)
        
        # Create basic metadata
        metadata = {'sample_idx': sample_idx}
        
        # Calculate 3D block coordinates
        block_z = block_idx // (self.blocks_per_dim ** 2)
        block_y = (block_idx % (self.blocks_per_dim ** 2)) // self.blocks_per_dim
        block_x = block_idx % self.blocks_per_dim
        
        # Extract the block
        z_start = block_z * self.block_size
        z_end = z_start + self.block_size
        y_start = block_y * self.block_size
        y_end = y_start + self.block_size
        x_start = block_x * self.block_size
        x_end = x_start + self.block_size
        
        # Convert to numpy if needed
        if torch.is_tensor(data):
            data_np = data.numpy()
        else:
            data_np = data
            
        block = data_np[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Normalize to [-1, 1] as mentioned in paper
        if self.normalize:
            # Linear normalization: (x - min) / (max - min) * 2 - 1
            block = (block - self.global_min) / (self.global_max - self.global_min) * 2 - 1
        
        # Convert to torch tensor and add channel dimension
        block = torch.from_numpy(block).float().unsqueeze(0)  # Shape: (1, 8, 8, 8)
        
        # Create metadata
        block_metadata = {
            'sample_idx': sample_idx,
            'block_coords': (block_x, block_y, block_z),
            'block_idx': block_idx,
            'original_metadata': metadata,
            'global_min': self.global_min if self.normalize else None,
            'global_max': self.global_max if self.normalize else None,
        }
        
        return block, block_metadata
    
    def denormalize_block(self, normalized_block):
        """
        Denormalize a block from [-1, 1] back to original range
        
        Args:
            normalized_block: torch.Tensor of shape (1, 8, 8, 8) or (8, 8, 8)
            
        Returns:
            Original scale block
        """
        if not self.normalize:
            return normalized_block
        
        # Remove channel dimension if present
        if normalized_block.dim() == 4:
            block = normalized_block.squeeze(0)
        else:
            block = normalized_block
        
        # Reverse normalization: (x + 1) / 2 * (max - min) + min
        block = (block + 1) / 2 * (self.global_max - self.global_min) + self.global_min
        
        return block
    
    def reconstruct_full_sample(self, sample_idx, model, device='cpu'):
        """
        Reconstruct a full 40x40x40 sample from its 8x8x8 blocks using the model
        
        Args:
            sample_idx: Index of the sample to reconstruct
            model: Trained SWAE model
            device: Device to run inference on
            
        Returns:
            reconstructed_data: numpy array of shape (40, 40, 40)
            original_data: numpy array of shape (40, 40, 40)
        """
        model.eval()
        
        # Initialize reconstruction array
        reconstructed = np.zeros(self.data_shape)
        
        # Get original data for comparison
        original_data = self.base_dataset[sample_idx]
        # Remove channel dimension if present
        if original_data.dim() == 4 and original_data.shape[0] == 1:
            original_data = original_data.squeeze(0)
        # Convert to numpy
        original_data = original_data.numpy()
        
        with torch.no_grad():
            # Process each block
            for block_z in range(self.blocks_per_dim):
                for block_y in range(self.blocks_per_dim):
                    for block_x in range(self.blocks_per_dim):
                        # Calculate block index
                        block_idx = (block_z * self.blocks_per_dim ** 2 + 
                                   block_y * self.blocks_per_dim + block_x)
                        
                        # Get the block
                        global_idx = sample_idx * self.total_blocks_per_sample + block_idx
                        block, metadata = self[global_idx]
                        
                        # Move to device and add batch dimension
                        block = block.unsqueeze(0).to(device)  # Shape: (1, 1, 8, 8, 8)
                        
                        # Reconstruct block
                        block_recon, _ = model(block)
                        
                        # Move back to CPU and remove batch/channel dimensions
                        block_recon = block_recon.squeeze(0).cpu()  # Shape: (1, 8, 8, 8)
                        
                        # Denormalize
                        block_recon = self.denormalize_block(block_recon)  # Shape: (8, 8, 8)
                        
                        # Place in reconstruction array
                        z_start = block_z * self.block_size
                        z_end = z_start + self.block_size
                        y_start = block_y * self.block_size
                        y_end = y_start + self.block_size
                        x_start = block_x * self.block_size
                        x_end = x_start + self.block_size
                        
                        reconstructed[z_start:z_end, y_start:y_end, x_start:x_end] = block_recon.numpy()
        
        return reconstructed, original_data


def create_swae_3d_datasets(k_values=[2, 3, 4, 5, 6], resolution=40, block_size=8, 
                           train_split=0.8, normalize=True):
    """
    Create train and validation datasets for SWAE 3D training
    
    Args:
        k_values: List of k values for mathematical functions
        resolution: Resolution of 3D functions (should be divisible by block_size)
        block_size: Size of blocks (8x8x8 as per paper)
        train_split: Fraction of data for training
        normalize: Whether to normalize data
        
    Returns:
        train_dataset, val_dataset: SWAE3DBlockDataset instances
    """
    # Check if resolution is divisible by block_size
    if resolution % block_size != 0:
        raise ValueError(f"Resolution {resolution} must be divisible by block_size {block_size}")
    
    # Create base dataset
    base_dataset = MathFunction3DDataset(
        k_values=k_values,
        resolution=resolution,
        num_functions=500,  # Total number of functions
        repeat=1,
        cache='in_memory',
        diagonal_only=True
    )
    
    # Split into train and validation indices
    total_samples = len(base_dataset)
    train_size = int(train_split * total_samples)
    
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_base = Subset(base_dataset, train_indices)
    val_base = Subset(base_dataset, val_indices)
    
    # Create block datasets
    train_dataset = SWAE3DBlockDataset(train_base, block_size=block_size, normalize=normalize)
    val_dataset = SWAE3DBlockDataset(val_base, block_size=block_size, normalize=normalize)
    
    # Use same normalization parameters for validation
    if normalize:
        val_dataset.global_min = train_dataset.global_min
        val_dataset.global_max = train_dataset.global_max
    
    print(f"\nDataset split:")
    print(f"  Train samples: {len(train_base)} ({len(train_dataset)} blocks)")
    print(f"  Val samples: {len(val_base)} ({len(val_dataset)} blocks)")
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Test the dataset
    print("Testing SWAE 3D Dataset...")
    
    # Create datasets
    train_dataset, val_dataset = create_swae_3d_datasets(
        k_values=[2, 3, 4],
        resolution=40,
        block_size=8,
        train_split=0.8
    )
    
    # Test getting a block
    block, metadata = train_dataset[0]
    print(f"\nBlock shape: {block.shape}")
    print(f"Block range: [{block.min():.3f}, {block.max():.3f}]")
    print(f"Metadata keys: {list(metadata.keys())}")
    print(f"Block coordinates: {metadata['block_coords']}")
    
    # Test denormalization
    denorm_block = train_dataset.denormalize_block(block)
    print(f"Denormalized range: [{denorm_block.min():.3f}, {denorm_block.max():.3f}]")
    
    print("\nDataset test completed successfully!") 