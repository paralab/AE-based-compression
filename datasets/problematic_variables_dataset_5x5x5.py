import os
import glob
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class ProblematicVariablesDataset5x5x5(Dataset):
    """
    Dataset for loading problematic variables from GR simulation HDF5 files
    with improved transformations (symmetric log and asinh).
    
    Specifically handles: U_B2, U_SYMAT2, U_GT2, U_SYMGT2, U_SYMAT4, U_SYMAT3
    
    Args:
        data_folder: Path to folder containing HDF5 files
        split: 'train', 'val', or 'test'
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation (rest goes to test)
        normalize: Whether to normalize the data
        normalize_method: 'symlog', 'asinh', 'robust', 'zscore', or 'none'
        target_vars: List of specific variables to load (default: problematic variables)
    """
    
    # Default problematic variables with their recommended transformations
    # Based on analysis, all variables work best with asinh transformation
    PROBLEMATIC_VARS = {
        'U_B2': {'transform': 'asinh', 'scale': 0.001},
        'U_SYMAT2': {'transform': 'asinh', 'scale': 0.01},
        'U_GT2': {'transform': 'asinh', 'scale': 0.001},
        'U_SYMGT2': {'transform': 'asinh', 'scale': 0.01},
        'U_SYMAT4': {'transform': 'asinh', 'scale': 0.01},
        'U_SYMAT3': {'transform': 'asinh', 'scale': 0.01}
    }
    
    def __init__(self, data_folder, split='train', train_ratio=0.8, val_ratio=0.15,
                 normalize=True, normalize_method='auto', shuffle=True, seed=42, 
                 target_vars=None):
        self.data_folder = data_folder
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.shuffle = shuffle
        self.seed = seed
        
        # Use specified variables or default to problematic ones
        if target_vars is None:
            self.target_vars = list(self.PROBLEMATIC_VARS.keys())
        else:
            self.target_vars = target_vars
        
        # Validate ratios
        if self.test_ratio < 0:
            raise ValueError(f"Invalid ratios: train_ratio({train_ratio}) + val_ratio({val_ratio}) > 1.0")
        
        # Load data
        self.data, self.var_names, self.sample_var_mapping = self._load_variables_data()
        
        # Apply deterministic shuffle
        if self.shuffle:
            np.random.seed(self.seed)
            perm = np.random.permutation(self.data.shape[0])
            self.data = self.data[perm]
            if hasattr(self, 'sample_var_mapping'):
                self.sample_var_mapping = [self.sample_var_mapping[i] for i in perm]
            print(f"🔒 Applied deterministic shuffle with seed={self.seed}")
        
        # Split data
        self._split_data()
        
        # Apply normalization
        if self.normalize:
            self._normalize_data()
            
        print(f"Problematic Variables Dataset (5x5x5) initialized:")
        print(f"  Split: {split}")
        print(f"  Data shape: {self.data.shape}")
        print(f"  Variables included: {self.var_names}")
        print(f"  Data range: [{self.data.min():.6f}, {self.data.max():.6f}]")
        print(f"  Data mean: {self.data.mean():.6f}")
        print(f"  Data std: {self.data.std():.6f}")
    
    def _load_variables_data(self):
        """Load specific variables data from HDF5 files"""
        hdf5_files = glob.glob(os.path.join(self.data_folder, "*.hdf5"))
        hdf5_files.sort()
        
        print(f"Found {len(hdf5_files)} HDF5 files")
        print(f"Loading variables: {self.target_vars}")
        
        all_data = []
        sample_var_mapping = []
        
        for file_path in hdf5_files:
            with h5py.File(file_path, 'r') as f:
                file_var_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                                 for name in f['vars'][:]]
                var_data = f['var_data'][:]  # Shape: (variables, num_samples, 7, 7, 7)
                
                # Load each target variable
                for var_name in self.target_vars:
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
        
        return full_data, self.target_vars, sample_var_mapping
    
    def _split_data(self):
        """Split data into train/val/test sets"""
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
        """Apply appropriate normalization based on variable type"""
        
        if self.normalize_method == 'auto':
            # Apply per-sample transformations based on variable type
            print("Applying automatic transformations based on variable type...")
            
            transformed_data = []
            self.transform_params = []
            
            for i in range(self.data.shape[0]):
                sample = self.data[i]  # (1, 5, 5, 5)
                var_name = self.sample_var_mapping[i]
                
                if var_name in self.PROBLEMATIC_VARS:
                    var_config = self.PROBLEMATIC_VARS[var_name]
                    
                    if var_config['transform'] == 'symlog':
                        # Symmetric log transformation
                        c = var_config['scale']
                        transformed = np.sign(sample) * np.log1p(np.abs(sample) / c) * c
                        transform_info = {'type': 'symlog', 'scale': c}
                        
                    elif var_config['transform'] == 'asinh':
                        # Inverse hyperbolic sine
                        scale = var_config['scale']
                        transformed = np.arcsinh(sample / scale)
                        transform_info = {'type': 'asinh', 'scale': scale}
                    
                    else:
                        # Default to standardization
                        mean = sample.mean()
                        std = sample.std()
                        transformed = (sample - mean) / (std + 1e-8)
                        transform_info = {'type': 'zscore', 'mean': mean, 'std': std}
                else:
                    # Unknown variable - use standardization
                    mean = sample.mean()
                    std = sample.std()
                    transformed = (sample - mean) / (std + 1e-8)
                    transform_info = {'type': 'zscore', 'mean': mean, 'std': std}
                
                transformed_data.append(transformed)
                self.transform_params.append(transform_info)
            
            self.data = np.array(transformed_data)
            print(f"Applied automatic transformations to {len(self.data)} samples")
            
        elif self.normalize_method == 'symlog':
            # Global symmetric log
            scale = 0.1  # Default scale
            self.data = np.sign(self.data) * np.log1p(np.abs(self.data) / scale) * scale
            self.transform_scale = scale
            print(f"Applied global symmetric log transformation with scale={scale}")
            
        elif self.normalize_method == 'asinh':
            # Global asinh
            scale = 0.01  # Default scale
            self.data = np.arcsinh(self.data / scale)
            self.transform_scale = scale
            print(f"Applied global asinh transformation with scale={scale}")
            
        elif self.normalize_method == 'robust':
            # Robust scaling using median and IQR
            median = np.median(self.data)
            q25, q75 = np.percentile(self.data, [25, 75])
            iqr = q75 - q25
            
            if iqr > 0:
                self.data = (self.data - median) / iqr
            else:
                self.data = self.data - median
                
            self.transform_median = median
            self.transform_iqr = iqr
            print(f"Applied robust scaling: median={median:.6f}, IQR={iqr:.6f}")
            
        elif self.normalize_method == 'zscore':
            # Standard z-score normalization
            self.data_mean = self.data.mean()
            self.data_std = self.data.std()
            self.data = (self.data - self.data_mean) / (self.data_std + 1e-8)
            print(f"Applied z-score normalization: mean={self.data_mean:.6f}, std={self.data_std:.6f}")
            
        elif self.normalize_method == 'none':
            print("No normalization applied")
    
    def denormalize(self, data, sample_indices=None):
        """Denormalize data back to original scale"""
        if not self.normalize:
            return data
        
        if self.normalize_method == 'auto' and sample_indices is not None:
            # Apply per-sample denormalization
            denorm_data = []
            for i, idx in enumerate(sample_indices):
                sample = data[i:i+1]
                params = self.transform_params[idx]
                
                if params['type'] == 'symlog':
                    c = params['scale']
                    denorm = np.sign(sample) * (np.expm1(np.abs(sample) / c) * c)
                elif params['type'] == 'asinh':
                    scale = params['scale']
                    denorm = np.sinh(sample) * scale
                elif params['type'] == 'zscore':
                    denorm = sample * params['std'] + params['mean']
                else:
                    denorm = sample
                    
                denorm_data.append(denorm)
            return np.concatenate(denorm_data, axis=0)
            
        elif self.normalize_method == 'symlog':
            c = self.transform_scale
            return np.sign(data) * (np.expm1(np.abs(data) / c) * c)
            
        elif self.normalize_method == 'asinh':
            return np.sinh(data) * self.transform_scale
            
        elif self.normalize_method == 'robust':
            return data * self.transform_iqr + self.transform_median
            
        elif self.normalize_method == 'zscore':
            return data * self.data_std + self.data_mean
            
        else:
            return data
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # Return data and metadata
        metadata = {
            'sample_idx': idx,
            'variable': self.sample_var_mapping[idx] if hasattr(self, 'sample_var_mapping') else 'unknown'
        }
        return torch.from_numpy(self.data[idx]).float(), metadata


def create_problematic_variables_datasets(data_folder, train_ratio=0.8, val_ratio=0.15,
                                        normalize=True, normalize_method='auto',
                                        batch_size=64, num_workers=4, target_vars=None):
    """Create train, val, and test datasets and dataloaders for problematic variables"""
    
    # Create datasets
    train_dataset = ProblematicVariablesDataset5x5x5(
        data_folder, split='train', train_ratio=train_ratio, val_ratio=val_ratio,
        normalize=normalize, normalize_method=normalize_method, target_vars=target_vars
    )
    
    val_dataset = ProblematicVariablesDataset5x5x5(
        data_folder, split='val', train_ratio=train_ratio, val_ratio=val_ratio,
        normalize=normalize, normalize_method=normalize_method, target_vars=target_vars
    )
    
    test_dataset = ProblematicVariablesDataset5x5x5(
        data_folder, split='test', train_ratio=train_ratio, val_ratio=val_ratio,
        normalize=normalize, normalize_method=normalize_method, target_vars=target_vars
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader