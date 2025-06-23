import torch
import numpy as np
from torch.utils.data import Dataset

from datasets import register


@register('math-function-3d')
class MathFunction3DDataset(Dataset):
    """
    Dataset for 3D mathematical functions of the form sin(2πk1*x)*sin(2πk2*y)*sin(2πk3*z)
    """
    
    def __init__(self, k_values=[2, 3, 4, 5, 6], resolution=40, 
                 num_functions=1000, repeat=1, cache='in_memory', diagonal_only=True):
        """
        Args:
            k_values: List of frequency values for k1, k2, and k3
            resolution: Resolution of the generated functions (resolution x resolution x resolution)
            num_functions: Number of functions to generate
            repeat: Number of times to repeat the dataset
            cache: Caching strategy ('in_memory', 'none')
            diagonal_only: If True, only use combinations where k1=k2=k3
        """
        self.k_values = k_values
        self.resolution = resolution
        self.num_functions = num_functions
        self.repeat = repeat
        self.cache = cache
        self.diagonal_only = diagonal_only
        
        # Generate k1, k2, k3 combinations
        self.k_combinations = []
        if diagonal_only:
            # Only diagonal combinations: (k, k, k)
            for k in k_values:
                self.k_combinations.append((k, k, k))
        else:
            # All possible combinations
            for k1 in k_values:
                for k2 in k_values:
                    for k3 in k_values:
                        self.k_combinations.append((k1, k2, k3))
        
        print(f"3D Math function dataset: Using {len(self.k_combinations)} combinations: {self.k_combinations}")
        
        # Pre-generate functions if caching in memory
        if cache == 'in_memory':
            self.functions = []
            for i in range(num_functions):
                func = self._generate_function(i)
                self.functions.append(func)
    
    def _generate_function(self, idx):
        """Generate a 3D mathematical function"""
        # Select k1, k2, k3 combination
        k1, k2, k3 = self.k_combinations[idx % len(self.k_combinations)]
        
        # Create coordinate grids
        x = np.linspace(-1, 1, self.resolution)
        y = np.linspace(-1, 1, self.resolution)
        z = np.linspace(-1, 1, self.resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Generate function: sin(2πk1*x) * sin(2πk2*y) * sin(2πk3*z)
        func = (np.sin(2 * np.pi * k1 * X) * 
                np.sin(2 * np.pi * k2 * Y) * 
                np.sin(2 * np.pi * k3 * Z))
        
        # Convert to tensor and add channel dimension
        func_tensor = torch.FloatTensor(func).unsqueeze(0)  # (1, D, H, W)
        
        return func_tensor
    
    def __len__(self):
        return self.num_functions * self.repeat
    
    def __getitem__(self, idx):
        # Handle repeat
        actual_idx = idx % self.num_functions
        
        if self.cache == 'in_memory':
            return self.functions[actual_idx]
        else:
            return self._generate_function(actual_idx)

    @staticmethod
    def generate_superposition(k_triplets, resolution=80):
        """
        Generate a superposition of multiple 3D mathematical functions
        
        Args:
            k_triplets: List of (k1, k2, k3) tuples for each component
            resolution: Resolution of the output function
        """
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        z = np.linspace(-1, 1, resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Sum all component functions
        func = np.zeros_like(X)
        for k1, k2, k3 in k_triplets:
            func += (np.sin(2 * np.pi * k1 * X) * 
                    np.sin(2 * np.pi * k2 * Y) * 
                    np.sin(2 * np.pi * k3 * Z))
        
        # Normalize to [-1, 1] range
        func = func / len(k_triplets)
        
        # Convert to tensor and add channel dimension
        func_tensor = torch.FloatTensor(func).unsqueeze(0)  # (1, D, H, W)
        
        return func_tensor 