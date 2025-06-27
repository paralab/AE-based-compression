#!/usr/bin/env python3
"""
U_CHI Dataset for SWAE 3D Training
Loads U_CHI variable from GR simulation HDF5 files
Data shape: (num_samples, 1, 7, 7, 7)
"""

import os
import glob
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class UCHIDataset(Dataset):
    """
    Dataset for loading U_CHI variable from GR simulation HDF5 files
    
    Args:
        data_folder: Path to folder containing HDF5 files
        split: 'train', 'val', or 'test'
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation (rest goes to test)
        normalize: Whether to normalize the data
        normalize_method: 'minmax', 'zscore', 'pos_log', or 'none'
    """
    
    def __init__(self, data_folder, split='train', train_ratio=0.8, val_ratio=0.15,
                 normalize=True, normalize_method='minmax', shuffle=True, seed=42):
        self.data_folder = data_folder
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.shuffle = shuffle
        self.seed = seed
        
        # Validate ratios
        if self.test_ratio < 0:
            raise ValueError(f"Invalid ratios: train_ratio({train_ratio}) + val_ratio({val_ratio}) > 1.0")
        
        # Load U_CHI data from all HDF5 files
        self.data = self._load_u_chi_data()
        
        # CRITICAL: Fixed deterministic shuffle to ensure test set is always the same
        # This guarantees the test set is never contaminated across different runs
        if self.shuffle:
            # Use FIXED seed to ensure identical splits across all runs
            np.random.seed(self.seed)
            perm = np.random.permutation(self.data.shape[0])
            self.data = self.data[perm]
            print(f"🔒 Applied deterministic shuffle with seed={self.seed} for reproducible splits")
        
        # Split data into train/val/test
        self._split_data()
        
        # Apply normalization if requested
        if self.normalize:
            self._normalize_data()
            
        print(f"U_CHI Dataset initialized:")
        print(f"  Split: {split}")
        print(f"  Data shape: {self.data.shape}")
        print(f"  Split ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={self.test_ratio:.1%}")
        print(f"  Data range: [{self.data.min():.6f}, {self.data.max():.6f}]")
        print(f"  Data mean: {self.data.mean():.6f}")
        print(f"  Data std: {self.data.std():.6f}")
    
    def _load_u_chi_data(self):
        """Load U_CHI data from all HDF5 files"""
        # Find all HDF5 files
        hdf5_files = glob.glob(os.path.join(self.data_folder, "*.hdf5"))
        hdf5_files.sort()  # Sort for consistent ordering
        
        print(f"Found {len(hdf5_files)} HDF5 files in {self.data_folder}")
        
        # Load U_CHI data from all files
        all_u_chi_data = []
        
        for file_path in hdf5_files:
            print(f"Loading {os.path.basename(file_path)}...")
            
            with h5py.File(file_path, 'r') as f:
                # Check available keys
                if len(all_u_chi_data) == 0:  # Only print for first file
                    print(f"Available keys: {list(f.keys())}")
                
                # Get variable names
                var_names = [name.decode('utf-8') for name in f['vars'][:]]
                if len(all_u_chi_data) == 0:  # Only print for first file
                    print(f"Available variables: {var_names}")
                
                # Find U_CHI index
                try:
                    u_chi_idx = var_names.index('U_CHI')
                    print(f"Found U_CHI at index {u_chi_idx}")
                except ValueError:
                    raise ValueError(f"U_CHI not found in {file_path}. Available: {var_names}")
                
                # Extract U_CHI data
                var_data = f['var_data'][:]  # Shape: (variables, num_samples, 7, 7, 7)
                u_chi_data = var_data[u_chi_idx]  # Shape: (num_samples, 7, 7, 7)
                
                print(f"  Loaded U_CHI data shape: {u_chi_data.shape}")
                all_u_chi_data.append(u_chi_data)
        
        # Concatenate all data
        full_data = np.concatenate(all_u_chi_data, axis=0)  # (total_samples, 7, 7, 7)
        
        # Add channel dimension to make it (total_samples, 1, 7, 7, 7)
        full_data = full_data[:, np.newaxis, :, :, :]
        
        print(f"Total U_CHI data shape: {full_data.shape}")
        
        return full_data
        
    def _split_data(self):
        """Split data into train, validation, and test sets"""
        total_samples = self.data.shape[0]
        n_train = int(total_samples * self.train_ratio)
        n_val = int(total_samples * self.val_ratio)
        n_test = total_samples - n_train - n_val
        
        print(f"🔒 FIXED Data Split: {total_samples} total → {n_train} train ({self.train_ratio:.1%}), {n_val} val ({self.val_ratio:.1%}), {n_test} test ({self.test_ratio:.1%})")
        print(f"🎯 TEST SET: {n_test} samples (5%) - NEVER seen during training/validation")
        
        if self.split == 'train':
            self.data = self.data[:n_train]
            print(f"Using training data: {self.data.shape}")
        elif self.split == 'val':
            self.data = self.data[n_train:n_train+n_val]
            print(f"Using validation data: {self.data.shape}")
        elif self.split == 'test':
            self.data = self.data[n_train+n_val:]
            print(f"Using test data: {self.data.shape}")
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")
    
    def _normalize_data(self):
        """Normalize the data based on the specified method"""
        if self.normalize_method == 'minmax':
            # Min-max normalization to [0, 1]
            self.data_min = self.data.min()
            self.data_max = self.data.max()
            self.data = (self.data - self.data_min) / (self.data_max - self.data_min)
            print(f"Applied min-max normalization: [{self.data_min:.6f}, {self.data_max:.6f}] -> [0, 1]")
            
        elif self.normalize_method == 'zscore':
            # Z-score normalization
            self.data_mean = self.data.mean()
            self.data_std = self.data.std()
            self.data = (self.data - self.data_mean) / self.data_std
            print(f"Applied z-score normalization: mean={self.data_mean:.6f}, std={self.data_std:.6f}")
            
        elif self.normalize_method == 'pos_log':
            # FIXED: Positive shift then log transformation (PER-SAMPLE, not global)
            # This ensures we get the expected statistical properties (mean~4.5, std~1.1)
            self.epsilon = 1e-8
            
            # Store transformation parameters for each sample
            self.sample_data_mins = []
            normalized_data = []
            
            print(f"Applying per-sample positive-shift log normalization...")
            
            for i in range(self.data.shape[0]):
                sample = self.data[i]  # Shape: (1, 7, 7, 7)
                
                # Apply per-sample transformation
                sample_min = sample.min()
                data_shifted = sample - sample_min + self.epsilon
                log_sample = np.log(data_shifted + self.epsilon)  # Double epsilon like reference
                
                # Store parameters for denormalization
                self.sample_data_mins.append(sample_min)
                normalized_data.append(log_sample)
            
            # Replace data with normalized version
            self.data = np.array(normalized_data)
            
            print(f"Applied per-sample positive-shift log normalization:")
            print(f"  epsilon={self.epsilon}")
            print(f"  Per-sample data_min range: [{min(self.sample_data_mins):.6f}, {max(self.sample_data_mins):.6f}]")
            print(f"  Result should have mean~4.5, std~1.1 properties")
            
        elif self.normalize_method == 'none':
            print("No normalization applied")
            
        else:
            raise ValueError(f"Invalid normalize_method: {self.normalize_method}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the data sample
        sample = self.data[idx]  # Shape: (1, 7, 7, 7)
        
        # Convert to tensor
        sample = torch.from_numpy(sample).float()
        
        # Return sample and metadata (empty dict for compatibility)
        metadata = {
            'index': idx,
            'original_shape': (7, 7, 7)
        }
        
        return sample, metadata
    
    def denormalize(self, data, sample_idx=None):
        """Denormalize data back to original range"""
        if not self.normalize:
            return data
            
        if self.normalize_method == 'minmax':
            return data * (self.data_max - self.data_min) + self.data_min
        elif self.normalize_method == 'zscore':
            return data * self.data_std + self.data_mean
        elif self.normalize_method == 'pos_log':
            # FIXED: Per-sample denormalization with correct sample index
            if hasattr(self, 'sample_data_mins') and len(self.sample_data_mins) > 0:
                if sample_idx is not None and 0 <= sample_idx < len(self.sample_data_mins):
                    # Use the correct sample's parameters
                    sample_min = self.sample_data_mins[sample_idx]
                    return np.exp(data) - self.epsilon + sample_min
                else:
                    # Fallback: return in log scale with warning
                    print(f"Warning: No valid sample_idx provided for denormalization. Returning log-scale data.")
                    return data
            else:
                # Fallback to old method if parameters not available
                return np.exp(data) - self.epsilon + getattr(self, 'data_min', 0)
        else:
            return data
    
    def denormalize_batch(self, data_batch, start_idx=0):
        """Denormalize a batch of data with correct per-sample parameters"""
        if not self.normalize or self.normalize_method != 'pos_log':
            return data_batch
            
        if not hasattr(self, 'sample_data_mins'):
            print("Warning: No per-sample parameters available for batch denormalization")
            return data_batch
        
        denormalized_batch = []
        for i, data in enumerate(data_batch):
            sample_idx = start_idx + i
            if sample_idx < len(self.sample_data_mins):
                sample_min = self.sample_data_mins[sample_idx]
                denormalized = np.exp(data) - self.epsilon + sample_min
                denormalized_batch.append(denormalized)
            else:
                print(f"Warning: Sample index {sample_idx} out of range, returning log-scale data")
                denormalized_batch.append(data)
        
        return np.array(denormalized_batch)


def create_u_chi_datasets(data_folder, train_ratio=0.8, val_ratio=0.15, normalize=True, normalize_method='minmax', shuffle=True, seed=42):
    """
    Create train, validation, and test datasets for U_CHI data
    
    CRITICAL: Uses FIXED seed to ensure test set is always the same 5% of data
    across all training runs and inference. Test set is NEVER seen during training.
    
    Args:
        data_folder: Path to folder containing HDF5 files
        train_ratio: Ratio of data to use for training (default: 0.8 = 80%)
        val_ratio: Ratio of data to use for validation (default: 0.15 = 15%)
        normalize: Whether to normalize the data
        normalize_method: Normalization method ('minmax', 'zscore', 'pos_log', 'none')
        seed: FIXED seed (42) for reproducible splits - DO NOT CHANGE!
    
    Returns:
        train_dataset, val_dataset, test_dataset
        
    Data Split:
        - Training: 80% (used for model training)
        - Validation: 15% (used for hyperparameter tuning and early stopping)  
        - Test: 5% (held out for final unbiased evaluation - NEVER seen during training)
    """
    
    # Calculate test ratio
    test_ratio = 1.0 - train_ratio - val_ratio
    print(f"🔒 Creating datasets with FIXED splits: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
    print(f"🎯 Using FIXED seed={seed} to ensure test set is always the same samples")
    
    if test_ratio < 0:
        raise ValueError(f"Invalid ratios: train_ratio({train_ratio}) + val_ratio({val_ratio}) > 1.0")
    
    if test_ratio != 0.05:
        print(f"⚠️  Warning: Test ratio is {test_ratio:.1%}, not the intended 5%")
    train_dataset = UCHIDataset(
        data_folder=data_folder,
        split='train', 
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize=normalize,
        normalize_method=normalize_method,
        shuffle=shuffle,
        seed=seed
    )
    
    val_dataset = UCHIDataset(
        data_folder=data_folder,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize=normalize,
        normalize_method=normalize_method,
        shuffle=shuffle,
        seed=seed
    )
    
    test_dataset = UCHIDataset(
        data_folder=data_folder,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize=normalize,
        normalize_method=normalize_method,
        shuffle=shuffle,
        seed=seed
    )
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Test the dataset
    data_folder = "/u/tawal/0620-NN-based-compression-thera/tt_q01/"
    
    print("Testing U_CHI dataset...")
    train_dataset, val_dataset, test_dataset = create_u_chi_datasets(
        data_folder=data_folder,
        train_ratio=0.8,  # 80% train
        val_ratio=0.15,   # 15% val
        normalize=True,   # 5% test
        normalize_method='minmax'
    )
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test data loading
    sample, metadata = train_dataset[0]
    print(f"\nSample shape: {sample.shape}")
    print(f"Sample dtype: {sample.dtype}")
    print(f"Sample range: [{sample.min():.6f}, {sample.max():.6f}]")
    print(f"Metadata: {metadata}")
    
    print("\nU_CHI dataset test completed successfully!") 