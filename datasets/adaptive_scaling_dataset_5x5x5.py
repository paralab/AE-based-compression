#!/usr/bin/env python3
"""
Dataset with adaptive scaling to bring all variables to learnable range
"""

import os
import glob
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import json


class AdaptiveScalingDataset5x5x5(Dataset):
    """
    Dataset that adaptively scales each variable to a standard range for better learning
    
    Key features:
    - Each variable is scaled to approximately [-10, 10] range
    - Handles near-zero and constant fields specially
    - Stores scaling parameters for exact reconstruction
    """
    
    def __init__(self, data_folder, split='train', train_ratio=0.8, val_ratio=0.15,
                 target_range=10.0, min_scale=1e-10, constant_threshold=1e-12,
                 variables=None, exclude_vars=None, shuffle=True, seed=42):
        
        self.data_folder = data_folder
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.target_range = target_range
        self.min_scale = min_scale
        self.constant_threshold = constant_threshold
        self.shuffle = shuffle
        self.seed = seed
        
        # Load data
        self.data, self.var_names, self.sample_var_mapping = self._load_data(variables, exclude_vars)
        
        # Calculate per-variable scaling parameters
        self.scaling_params = self._calculate_scaling_params()
        
        # Apply scaling
        self._apply_scaling()
        
        # Split data
        self._split_data()
        
        print(f"Adaptive Scaling Dataset initialized:")
        print(f"  Split: {split}")
        print(f"  Data shape: {self.data.shape}")
        print(f"  Variables: {len(self.var_names)}")
        print(f"  Target range: [-{target_range}, {target_range}]")
        
    def _load_data(self, variables, exclude_vars):
        """Load data from HDF5 files"""
        hdf5_files = glob.glob(os.path.join(self.data_folder, "*.hdf5"))
        hdf5_files.sort()
        
        all_data = []
        all_var_names = []
        sample_var_mapping = []
        
        print(f"Loading from {len(hdf5_files)} files...")
        
        for file_path in hdf5_files:  # Load all files
            with h5py.File(file_path, 'r') as f:
                var_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                            for name in f['vars'][:]]
                var_data = f['var_data'][:]  # Shape: (num_vars, num_samples, 7, 7, 7)
                
                # Filter variables
                for i, var_name in enumerate(var_names):
                    if not var_name.startswith('U_'):
                        continue
                    if variables and var_name not in variables:
                        continue
                    if exclude_vars and var_name in exclude_vars:
                        continue
                    
                    # Extract 5x5x5 center
                    var_samples = var_data[i][:, 1:6, 1:6, 1:6]
                    var_samples = var_samples[:, np.newaxis, :, :, :]  # Add channel dim
                    
                    all_data.append(var_samples)
                    sample_var_mapping.extend([var_name] * var_samples.shape[0])
                    
                    if var_name not in all_var_names:
                        all_var_names.append(var_name)
        
        # Concatenate all data
        data = np.concatenate(all_data, axis=0)
        
        # Shuffle if requested
        if self.shuffle:
            np.random.seed(self.seed)
            perm = np.random.permutation(data.shape[0])
            data = data[perm]
            sample_var_mapping = [sample_var_mapping[i] for i in perm]
        
        return data, all_var_names, sample_var_mapping
    
    def _calculate_scaling_params(self):
        """Calculate scaling parameters for each variable"""
        scaling_params = {}
        
        print("\nCalculating scaling parameters for each variable...")
        
        for var_name in self.var_names:
            # Get all samples for this variable
            var_indices = [i for i, v in enumerate(self.sample_var_mapping) if v == var_name]
            if not var_indices:
                continue
                
            var_data = self.data[var_indices]
            
            # Calculate statistics
            var_flat = var_data.flatten()
            var_abs = np.abs(var_flat)
            var_abs_nonzero = var_abs[var_abs > self.constant_threshold]
            
            if len(var_abs_nonzero) == 0:
                # All zeros or near-zeros
                scale_type = 'constant'
                scale_factor = 1.0 / self.min_scale
                shift = 0.0
                typical_magnitude = self.min_scale
            else:
                # Calculate robust statistics
                typical_magnitude = np.median(var_abs_nonzero)
                var_std = np.std(var_flat)
                var_range = np.max(var_flat) - np.min(var_flat)
                
                if var_range < self.constant_threshold:
                    # Essentially constant field
                    scale_type = 'constant'
                    scale_factor = 1.0
                    shift = -np.mean(var_flat)
                elif typical_magnitude < 1e-6:
                    # Very small values - use magnitude scaling
                    scale_type = 'magnitude'
                    scale_factor = self.target_range / max(typical_magnitude, self.min_scale)
                    shift = 0.0
                else:
                    # Normal scaling based on percentiles
                    scale_type = 'percentile'
                    p95 = np.percentile(var_abs, 95)
                    scale_factor = self.target_range / max(p95, self.min_scale)
                    shift = 0.0
            
            scaling_params[var_name] = {
                'scale_type': scale_type,
                'scale_factor': scale_factor,
                'shift': shift,
                'typical_magnitude': typical_magnitude,
                'original_min': np.min(var_flat),
                'original_max': np.max(var_flat)
            }
            
            print(f"  {var_name}: type={scale_type}, scale={scale_factor:.2e}, "
                  f"magnitude={typical_magnitude:.2e}, range=[{scaling_params[var_name]['original_min']:.2e}, "
                  f"{scaling_params[var_name]['original_max']:.2e}]")
        
        return scaling_params
    
    def _apply_scaling(self):
        """Apply scaling followed by poslog transformation"""
        print("\nApplying adaptive scaling + poslog...")
        
        scaled_data = np.zeros_like(self.data)
        self.poslog_params = []  # Store poslog parameters for denormalization
        
        for i, var_name in enumerate(self.sample_var_mapping):
            if var_name not in self.scaling_params:
                scaled_data[i] = self.data[i]
                self.poslog_params.append(None)
                continue
                
            params = self.scaling_params[var_name]
            sample = self.data[i]
            
            # Step 1: Apply scaling to bring to reasonable range
            if params['scale_type'] == 'constant':
                # For constant fields, just center and scale slightly
                scaled_sample = (sample + params['shift']) * params['scale_factor']
            else:
                # Standard scaling
                scaled_sample = sample * params['scale_factor']
            
            # Step 2: Apply poslog transformation
            # Shift to positive and apply log
            sample_min = scaled_sample.min()
            epsilon = 1e-8
            shifted_sample = scaled_sample - sample_min + epsilon
            poslog_sample = np.log(shifted_sample + epsilon)
            
            scaled_data[i] = poslog_sample
            
            # Store poslog parameters
            self.poslog_params.append({
                'min_value': sample_min,
                'epsilon': epsilon
            })
        
        # Replace original data with scaled data
        self.data = scaled_data
        
        # Calculate final statistics
        print(f"\nScaled + poslog data statistics:")
        print(f"  Overall range: [{self.data.min():.2f}, {self.data.max():.2f}]")
        print(f"  Overall mean: {self.data.mean():.2f}")
        print(f"  Overall std: {self.data.std():.2f}")
    
    def _split_data(self):
        """Split data into train/val/test"""
        n_samples = self.data.shape[0]
        n_train = int(n_samples * self.train_ratio)
        n_val = int(n_samples * self.val_ratio)
        
        if self.split == 'train':
            self.data = self.data[:n_train]
            self.sample_var_mapping = self.sample_var_mapping[:n_train]
        elif self.split == 'val':
            self.data = self.data[n_train:n_train + n_val]
            self.sample_var_mapping = self.sample_var_mapping[n_train:n_train + n_val]
        else:  # test
            self.data = self.data[n_train + n_val:]
            self.sample_var_mapping = self.sample_var_mapping[n_train + n_val:]
    
    def denormalize(self, data, sample_indices):
        """Denormalize data back to original scale"""
        if isinstance(sample_indices, int):
            sample_indices = [sample_indices]
        
        denorm_data = np.zeros_like(data)
        
        for i, idx in enumerate(sample_indices):
            var_name = self.sample_var_mapping[idx]
            if var_name not in self.scaling_params:
                denorm_data[i] = data[i]
                continue
                
            params = self.scaling_params[var_name]
            poslog_params = self.poslog_params[idx] if idx < len(self.poslog_params) else None
            
            # Step 1: Inverse poslog transformation
            if poslog_params is not None:
                # Inverse of log: exp
                exp_data = np.exp(data[i])
                # Remove epsilon and shift back
                unshifted_data = exp_data - poslog_params['epsilon']
                # Add back the minimum value
                inverse_poslog = unshifted_data + poslog_params['min_value']
            else:
                inverse_poslog = data[i]
            
            # Step 2: Inverse scaling
            if params['scale_type'] == 'constant':
                denorm_data[i] = (inverse_poslog / params['scale_factor']) - params['shift']
            else:
                denorm_data[i] = inverse_poslog / params['scale_factor']
        
        return denorm_data.squeeze() if len(sample_indices) == 1 else denorm_data
    
    def save_scaling_params(self, filepath):
        """Save scaling parameters to file"""
        # Convert poslog params list to a dict for saving
        poslog_params_dict = {}
        for i, var_name in enumerate(self.sample_var_mapping):
            if i < len(self.poslog_params) and self.poslog_params[i] is not None:
                if var_name not in poslog_params_dict:
                    poslog_params_dict[var_name] = self.poslog_params[i]
        
        save_data = {
            'scaling_params': self.scaling_params,
            'poslog_params': poslog_params_dict
        }
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data[idx]).float()
        
        metadata = {
            'index': idx,
            'variable_name': self.sample_var_mapping[idx],
            'scale_factor': self.scaling_params[self.sample_var_mapping[idx]]['scale_factor'],
            'scale_type': self.scaling_params[self.sample_var_mapping[idx]]['scale_type']
        }
        
        return sample, metadata


def create_adaptive_scaling_datasets(data_folder, train_ratio=0.8, val_ratio=0.15,
                                   target_range=10.0, batch_size=64, num_workers=4,
                                   variables=None, exclude_vars=None):
    """Create train, val, and test datasets with adaptive scaling"""
    
    # Create datasets
    train_dataset = AdaptiveScalingDataset5x5x5(
        data_folder, split='train', train_ratio=train_ratio, val_ratio=val_ratio,
        target_range=target_range, variables=variables, exclude_vars=exclude_vars
    )
    
    val_dataset = AdaptiveScalingDataset5x5x5(
        data_folder, split='val', train_ratio=train_ratio, val_ratio=val_ratio,
        target_range=target_range, variables=variables, exclude_vars=exclude_vars
    )
    
    test_dataset = AdaptiveScalingDataset5x5x5(
        data_folder, split='test', train_ratio=train_ratio, val_ratio=val_ratio,
        target_range=target_range, variables=variables, exclude_vars=exclude_vars
    )
    
    # Save scaling parameters
    train_dataset.save_scaling_params('./scaling_params.json')
    
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