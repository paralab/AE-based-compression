import torch
import numpy as np
from torch.utils.data import Dataset

from datasets import register


@register('math-function')
class MathFunctionDataset(Dataset):
    """
    Dataset for mathematical functions of the form sin(2πk1*x)*sin(2πk2*y)
    """
    
    def __init__(self, k_values=[2, 3, 4, 5, 6], resolution=1024, 
                 num_functions=1000, repeat=1, cache='in_memory'):
        """
        Args:
            k_values: List of frequency values for k1 and k2
            resolution: Resolution of the generated functions (resolution x resolution)
            num_functions: Number of functions to generate
            repeat: Number of times to repeat the dataset
            cache: Caching strategy ('in_memory', 'none')
        """
        self.k_values = k_values
        self.resolution = resolution
        self.num_functions = num_functions
        self.repeat = repeat
        self.cache = cache
        
        # Generate all possible k1, k2 combinations
        self.k_combinations = []
        for k1 in k_values:
            for k2 in k_values:
                self.k_combinations.append((k1, k2))
        
        # Pre-generate functions if caching in memory
        if cache == 'in_memory':
            self.functions = []
            for i in range(num_functions):
                func = self._generate_function(i)
                self.functions.append(func)
    
    def _generate_function(self, idx):
        """Generate a mathematical function"""
        # Select k1, k2 combination
        k1, k2 = self.k_combinations[idx % len(self.k_combinations)]
        
        # Create coordinate grids
        x = np.linspace(-1, 1, self.resolution)
        y = np.linspace(-1, 1, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        # Generate function: sin(2πk1*x) * sin(2πk2*y)
        func = np.sin(2 * np.pi * k1 * X) * np.sin(2 * np.pi * k2 * Y)
        
        # Convert to tensor and add channel dimension
        func_tensor = torch.FloatTensor(func).unsqueeze(0)  # (1, H, W)
        
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