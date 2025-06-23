import torch
import torch.nn as nn
import numpy as np

from models import register


@register('thera-3d')
class TheraNet3D(nn.Module):
    """
    3D Thera Neural Heat Field implementation
    Φ(x, t) = W2 · ξ(W1x + b1, ν(W1), κ, t) + b2
    where ξ(z, ν, κ, t) = sin(z) · exp(-|ν|²κt)
    """
    
    def __init__(self, in_dim, out_dim, hidden_dim=256, num_frequencies=64, kappa=1.0):
        super().__init__()
        self.in_dim = in_dim - 1  # Subtract 1 for scale parameter t
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_frequencies = num_frequencies
        self.kappa = kappa
        
        # Global frequency bank W1 (shared across all positions)
        self.W1 = nn.Parameter(torch.randn(self.in_dim, num_frequencies) * 0.1)
        
        # Hypernetwork to predict phase shifts b1 and output weights W2
        hypernetwork_input_dim = self.in_dim
        self.hypernetwork = nn.Sequential(
            nn.Linear(hypernetwork_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_frequencies + num_frequencies * out_dim)  # b1 + W2
        )
        
        # Output bias b2
        self.b2 = nn.Parameter(torch.zeros(out_dim))
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Better weight initialization"""
        # Initialize frequency bank with smaller values
        nn.init.normal_(self.W1, 0, 0.01)
        
        # Initialize hypernetwork
        for module in self.hypernetwork:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        
    def thermal_activation(self, z, nu, kappa, t):
        """
        Thermal activation function: ξ(z, ν, κ, t) = sin(z) · exp(-|ν|²κt)
        
        Args:
            z: Input after linear transformation W1x + b1, shape (..., num_frequencies)
            nu: Frequency magnitudes |ν| from W1, shape (num_frequencies,)
            kappa: Thermal diffusivity parameter
            t: Time parameter (scale squared), shape (..., 1)
        """
        # Compute frequency magnitudes |ν|² for each frequency
        nu_squared = torch.sum(nu ** 2, dim=0)  # (num_frequencies,)
        
        # Expand dimensions for broadcasting
        nu_squared = nu_squared.unsqueeze(0).expand(z.shape[0], -1)  # (batch_size, num_frequencies)
        t_expanded = t.unsqueeze(-1).expand(-1, self.num_frequencies)  # (batch_size, num_frequencies)
        
        # Thermal decay: exp(-|ν|²κt)
        thermal_decay = torch.exp(-nu_squared * kappa * t_expanded)
        
        # Apply thermal activation: sin(z) · exp(-|ν|²κt)
        return torch.sin(z) * thermal_decay
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor, shape (batch_size, in_dim)
               Last dimension is the scale parameter t = S²
        """
        batch_size = x.shape[0]
        
        # Split input: coordinates/features and scale parameter
        coords_features = x[:, :-1]  # (batch_size, in_dim-1)
        t = x[:, -1:]  # (batch_size, 1) - scale parameter t = S²
        
        # Linear transformation: W1x
        z_base = torch.matmul(coords_features, self.W1)  # (batch_size, num_frequencies)
        
        # Use hypernetwork to predict phase shifts b1 and output weights W2
        hyper_output = self.hypernetwork(coords_features)  # (batch_size, num_frequencies + num_frequencies * out_dim)
        
        # Split hypernetwork output
        b1 = hyper_output[:, :self.num_frequencies]  # (batch_size, num_frequencies)
        W2_flat = hyper_output[:, self.num_frequencies:]  # (batch_size, num_frequencies * out_dim)
        W2 = W2_flat.view(batch_size, self.num_frequencies, self.out_dim)  # (batch_size, num_frequencies, out_dim)
        
        # Add phase shifts: W1x + b1
        z = z_base + b1  # (batch_size, num_frequencies)
        
        # Apply thermal activation
        xi = self.thermal_activation(z, self.W1, self.kappa, t)  # (batch_size, num_frequencies)
        
        # Final output: W2 · ξ + b2
        # xi: (batch_size, num_frequencies), W2: (batch_size, num_frequencies, out_dim)
        output = torch.bmm(xi.unsqueeze(1), W2).squeeze(1)  # (batch_size, out_dim)
        output = output + self.b2  # Add bias
        
        return output


@register('thera-3d-simple')
class TheraNet3DSimple(nn.Module):
    """
    Simplified 3D Thera implementation with fixed frequency bank
    """
    
    def __init__(self, in_dim, out_dim, hidden_dim=256, num_frequencies=64, kappa=1.0):
        super().__init__()
        self.in_dim = in_dim - 1  # Subtract 1 for scale parameter t
        self.out_dim = out_dim
        self.num_frequencies = num_frequencies
        self.kappa = kappa
        
        # Fixed frequency bank for 3D - better initialization
        self.register_buffer('W1', torch.randn(self.in_dim, num_frequencies) * 0.01)
        
        # MLP to predict output weights with better architecture
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim + num_frequencies, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Add layer normalization
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Better weight initialization"""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Split input
        coords_features = x[:, :-1]  # (batch_size, in_dim-1)
        t = x[:, -1:]  # (batch_size, 1)
        
        # Linear transformation
        z = torch.matmul(coords_features, self.W1)  # (batch_size, num_frequencies)
        
        # Thermal activation with fixed frequencies
        nu_squared = torch.sum(self.W1 ** 2, dim=0)  # (num_frequencies,)
        nu_squared = nu_squared.unsqueeze(0).expand(batch_size, -1)
        t_expanded = t.expand(-1, self.num_frequencies)
        
        # Clamp t to avoid numerical issues
        t_clamped = torch.clamp(t_expanded, min=1e-6, max=100.0)
        
        thermal_decay = torch.exp(-nu_squared * self.kappa * t_clamped)
        xi = torch.sin(z) * thermal_decay
        
        # Concatenate features and thermal activations
        features = torch.cat([coords_features, xi], dim=-1)
        
        return self.mlp(features) 