import os
import glob
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class DualTransformDataset5x5x5(Dataset):
    """
    Dataset that splits variables into two groups with different transformations:
    - Group 1 (poslog): Standard variables that work well with poslog
    - Group 2 (asinh): Problematic variables that need asinh transformation
    
    This enables training two separate models optimized for each group.
    """
    
    # Variables that need special handling (asinh transformation)
    ASINH_VARIABLES = {
        'U_B2': {'scale': 0.001},
        'U_SYMAT2': {'scale': 0.01},
        'U_GT2': {'scale': 0.001},
        'U_SYMGT2': {'scale': 0.01},
        'U_SYMAT4': {'scale': 0.01},
        'U_SYMAT3': {'scale': 0.01}
    }
    
    def __init__(self, data_folder, group='poslog', split='train', train_ratio=0.8, 
                 val_ratio=0.15, normalize=True, shuffle=True, seed=42):
        """
        Args:
            data_folder: Path to folder containing HDF5 files
            group: 'poslog' for standard variables, 'asinh' for problematic variables
            split: 'train', 'val', or 'test'
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            normalize: Whether to normalize the data
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
        """
        self.data_folder = data_folder
        self.group = group
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.normalize = normalize
        self.shuffle = shuffle
        self.seed = seed
        
        # Validate ratios
        if self.test_ratio < 0:
            raise ValueError(f"Invalid ratios: train_ratio({train_ratio}) + val_ratio({val_ratio}) > 1.0")
        
        # Load data for the appropriate variable group
        self.data, self.var_names, self.sample_var_mapping = self._load_group_data()
        
        # Apply deterministic shuffle
        if self.shuffle:
            np.random.seed(self.seed)
            perm = np.random.permutation(self.data.shape[0])
            self.data = self.data[perm]
            if hasattr(self, 'sample_var_mapping'):
                self.sample_var_mapping = [self.sample_var_mapping[i] for i in perm]
        
        # Split data
        self._split_data()
        
        # Apply normalization
        if self.normalize:
            self._normalize_data()
        
        print(f"Dual Transform Dataset ({group} group) initialized:")
        print(f"  Split: {split}")
        print(f"  Data shape: {self.data.shape}")
        print(f"  Variables: {self.var_names}")
        print(f"  Transformation: {'poslog' if group == 'poslog' else 'asinh'}")
    
    def _load_group_data(self):
        """Load data for the specific variable group."""
        hdf5_files = glob.glob(os.path.join(self.data_folder, "*.hdf5"))
        hdf5_files.sort()
        
        print(f"Found {len(hdf5_files)} HDF5 files")
        
        # First, get all available variables
        with h5py.File(hdf5_files[0], 'r') as f:
            all_vars = [name.decode('utf-8') if isinstance(name, bytes) else name 
                       for name in f['vars'][:]]
            # Filter to U_ variables only
            u_vars = [v for v in all_vars if v.startswith('U_')]
        
        # Determine which variables to load based on group
        if self.group == 'asinh':
            target_vars = [v for v in u_vars if v in self.ASINH_VARIABLES]
        else:  # poslog group
            target_vars = [v for v in u_vars if v not in self.ASINH_VARIABLES]
        
        print(f"Loading {len(target_vars)} variables for {self.group} group: {target_vars}")
        
        # Load data
        all_data = []
        sample_var_mapping = []
        
        for file_path in hdf5_files:
            with h5py.File(file_path, 'r') as f:
                file_var_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                                 for name in f['vars'][:]]
                var_data = f['var_data'][:]  # Shape: (variables, num_samples, 7, 7, 7)
                
                # Load each target variable
                for var_name in target_vars:
                    if var_name in file_var_names:
                        var_idx = file_var_names.index(var_name)
                        var_samples = var_data[var_idx]  # (num_samples, 7, 7, 7)
                        
                        # Extract 5x5x5 center crop
                        var_samples_5x5x5 = var_samples[:, 1:6, 1:6, 1:6]
                        
                        # Add to data
                        all_data.append(var_samples_5x5x5)
                        
                        # Track which variable each sample belongs to
                        sample_var_mapping.extend([var_name] * var_samples_5x5x5.shape[0])
        
        # Concatenate all data
        full_data = np.concatenate(all_data, axis=0)
        full_data = full_data[:, np.newaxis, :, :, :]  # Add channel dimension
        
        return full_data, target_vars, sample_var_mapping
    
    def _split_data(self):
        """Split data into train/val/test sets."""
        n_samples = self.data.shape[0]
        n_train = int(n_samples * self.train_ratio)
        n_val = int(n_samples * self.val_ratio)
        
        if self.split == 'train':
            self.data = self.data[:n_train]
            if hasattr(self, 'sample_var_mapping'):
                self.sample_var_mapping = self.sample_var_mapping[:n_train]
        elif self.split == 'val':
            self.data = self.data[n_train:n_train + n_val]
            if hasattr(self, 'sample_var_mapping'):
                self.sample_var_mapping = self.sample_var_mapping[n_train:n_train + n_val]
        else:  # test
            self.data = self.data[n_train + n_val:]
            if hasattr(self, 'sample_var_mapping'):
                self.sample_var_mapping = self.sample_var_mapping[n_train + n_val:]
    
    def _normalize_data(self):
        """Apply appropriate normalization based on group."""
        
        if self.group == 'poslog':
            # Standard poslog transformation (per-sample)
            self.epsilon = 1e-8
            self.sample_data_mins = []
            normalized_data = []
            
            print(f"Applying per-sample positive-shift log normalization...")
            
            for i in range(self.data.shape[0]):
                sample = self.data[i]
                sample_min = sample.min()
                data_shifted = sample - sample_min + self.epsilon
                log_sample = np.log(data_shifted + self.epsilon)
                
                self.sample_data_mins.append(sample_min)
                normalized_data.append(log_sample)
            
            self.data = np.array(normalized_data)
            
        else:  # asinh group
            # Apply asinh transformation with variable-specific scales
            print(f"Applying asinh transformation with variable-specific scales...")
            
            self.transform_params = []
            
            for i in range(self.data.shape[0]):
                sample = self.data[i]
                var_name = self.sample_var_mapping[i]
                
                if var_name in self.ASINH_VARIABLES:
                    scale = self.ASINH_VARIABLES[var_name]['scale']
                else:
                    scale = 0.01  # Default scale
                
                # Apply asinh transformation
                self.data[i] = np.arcsinh(sample / scale)
                self.transform_params.append({'type': 'asinh', 'scale': scale})
    
    def denormalize(self, data, sample_idx=None):
        """Denormalize data back to original scale."""
        if not self.normalize:
            return data
        
        if self.group == 'poslog':
            # Reverse poslog transformation
            if sample_idx is not None:
                data_min = self.sample_data_mins[sample_idx]
                return np.exp(data) - self.epsilon + data_min
            else:
                # Batch denormalization not implemented for poslog
                return data
                
        else:  # asinh group
            # Reverse asinh transformation
            if sample_idx is not None:
                params = self.transform_params[sample_idx]
                scale = params['scale']
                return np.sinh(data) * scale
            else:
                # Batch denormalization
                return np.sinh(data) * 0.01  # Use default scale
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # Return data and metadata
        metadata = {
            'sample_idx': idx,
            'variable': self.sample_var_mapping[idx] if hasattr(self, 'sample_var_mapping') else 'unknown',
            'group': self.group
        }
        return torch.from_numpy(self.data[idx]).float(), metadata


def create_dual_datasets(data_folder, train_ratio=0.8, val_ratio=0.15,
                        normalize=True, batch_size=64, num_workers=4):
    """Create datasets for both variable groups."""
    
    datasets = {}
    dataloaders = {}
    
    for group in ['poslog', 'asinh']:
        # Create datasets for this group
        train_dataset = DualTransformDataset5x5x5(
            data_folder, group=group, split='train', 
            train_ratio=train_ratio, val_ratio=val_ratio,
            normalize=normalize
        )
        
        val_dataset = DualTransformDataset5x5x5(
            data_folder, group=group, split='val',
            train_ratio=train_ratio, val_ratio=val_ratio,
            normalize=normalize
        )
        
        test_dataset = DualTransformDataset5x5x5(
            data_folder, group=group, split='test',
            train_ratio=train_ratio, val_ratio=val_ratio,
            normalize=normalize
        )
        
        # Store datasets
        datasets[group] = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        
        # Create dataloaders
        dataloaders[group] = {
            'train': torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True
            ),
            'val': torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            )
        }
    
    return datasets, dataloaders