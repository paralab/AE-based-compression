import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import register


def sliced_wasserstein_distance(X, Y, num_projections=50, p=2, device='cuda'):
    """
    Compute the sliced Wasserstein distance between two sets of samples.
    
    Args:
        X: tensor of shape (batch_size, latent_dim)
        Y: tensor of shape (batch_size, latent_dim) 
        num_projections: number of random projections
        p: power for the Wasserstein distance calculation
        device: device to run computation on
    """
    dim = X.shape[1]
    
    # Generate random projections from unit sphere
    projections = torch.randn(num_projections, dim, device=device)
    projections = projections / torch.norm(projections, dim=1, keepdim=True)
    
    # Project both sets of samples
    X_projections = torch.matmul(X, projections.T)  # (batch_size, num_projections)
    Y_projections = torch.matmul(Y, projections.T)  # (batch_size, num_projections)
    
    # Sort projections
    X_sorted, _ = torch.sort(X_projections, dim=0)
    Y_sorted, _ = torch.sort(Y_projections, dim=0)
    
    # Compute Wasserstein distance for each projection
    wasserstein_distances = torch.mean(torch.abs(X_sorted - Y_sorted) ** p, dim=0)
    
    # Return mean over all projections
    return torch.mean(wasserstein_distances)


@register('swae-3d')
class SWAE3D(nn.Module):
    """
    3D Sliced-Wasserstein Autoencoder for scientific data compression
    Based on the paper "Exploring Autoencoder-based Error-bounded Compression for Scientific Data"
    """
    
    def __init__(self, input_channels=1, latent_dim=64, hidden_channels=[32, 64, 128], 
                 num_projections=50, lambda_reg=10.0):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_projections = num_projections
        self.lambda_reg = lambda_reg
        self.input_channels = input_channels
        
        # Encoder - 3D Convolutional layers
        encoder_layers = []
        in_ch = input_channels
        
        for hidden_ch in hidden_channels:
            encoder_layers.extend([
                nn.Conv3d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(hidden_ch),
                nn.ReLU(inplace=True)
            ])
            in_ch = hidden_ch
        
        self.encoder_conv = nn.Sequential(*encoder_layers)
        
        # Calculate the size after convolutions for the fully connected layer
        # Assuming input size is (40, 40, 40), after 3 stride-2 convolutions: (5, 5, 5)
        self.feature_size = hidden_channels[-1] * (5 * 5 * 5)  # 125 * hidden_channels[-1]
        
        # Encoder fully connected layer
        self.encoder_fc = nn.Linear(self.feature_size, latent_dim)
        
        # Decoder fully connected layer
        self.decoder_fc = nn.Linear(latent_dim, self.feature_size)
        
        # Decoder - 3D Transposed Convolutional layers
        decoder_layers = []
        hidden_channels_reversed = hidden_channels[::-1]
        
        for i, hidden_ch in enumerate(hidden_channels_reversed[:-1]):
            next_ch = hidden_channels_reversed[i + 1]
            decoder_layers.extend([
                nn.ConvTranspose3d(hidden_ch, next_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm3d(next_ch),
                nn.ReLU(inplace=True)
            ])
        
        # Final layer to get back to input channels
        decoder_layers.append(
            nn.ConvTranspose3d(hidden_channels_reversed[-1], input_channels, 
                             kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        decoder_layers.append(nn.Tanh())  # Output in [-1, 1] range
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
        
        # Store the spatial dimensions for reshaping
        self.spatial_dims = (5, 5, 5)  # After 3 stride-2 convolutions from (40,40,40)
        
    def encode(self, x):
        """Encode input to latent space"""
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        z = self.encoder_fc(x)
        return z
    
    def decode(self, z):
        """Decode latent representation to output"""
        x = self.decoder_fc(z)
        x = x.view(x.size(0), -1, *self.spatial_dims)  # Reshape to 3D
        x = self.decoder_conv(x)
        return x
    
    def forward(self, x):
        """Forward pass through the autoencoder"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def compute_loss(self, x, x_recon, z, prior_samples=None):
        """
        Compute SWAE loss: reconstruction loss + sliced Wasserstein loss
        
        Args:
            x: input data
            x_recon: reconstructed data
            z: latent codes
            prior_samples: samples from prior distribution (if None, use standard normal)
        """
        batch_size = x.size(0)
        
        # Reconstruction loss (L2)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # Generate prior samples if not provided (standard normal distribution)
        if prior_samples is None:
            prior_samples = torch.randn(batch_size, self.latent_dim, device=x.device)
        
        # Sliced Wasserstein distance between latent codes and prior
        sw_loss = sliced_wasserstein_distance(
            z, prior_samples, 
            num_projections=self.num_projections, 
            device=x.device
        )
        
        # Total loss
        total_loss = recon_loss + self.lambda_reg * sw_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'sw_loss': sw_loss
        }


@register('swae-encoder-3d')
class SWAE3DEncoder(nn.Module):
    """
    3D SWAE Encoder for use with LIIF+Thera
    """
    
    def __init__(self, input_channels=1, hidden_channels=[32, 64, 128], no_upsampling=True):
        super().__init__()
        
        self.no_upsampling = no_upsampling
        
        # Encoder - 3D Convolutional layers
        encoder_layers = []
        in_ch = input_channels
        
        for hidden_ch in hidden_channels:
            encoder_layers.extend([
                nn.Conv3d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(hidden_ch),
                nn.ReLU(inplace=True)
            ])
            in_ch = hidden_ch
        
        self.encoder_conv = nn.Sequential(*encoder_layers)
        
        # Output dimension for LIIF
        self.out_dim = hidden_channels[-1]
    
    def forward(self, x):
        """Forward pass through encoder"""
        features = self.encoder_conv(x)
        return features 