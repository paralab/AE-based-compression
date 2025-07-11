#!/usr/bin/env python3
"""
SWAE 3D Model for 5x5x5 Blocks (U_CHI Dataset)
Adapted from the original 7x7x7 SWAE implementation
Based on "Exploring Autoencoder-based Error-bounded Compression for Scientific Data"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GDN(nn.Module):
    """Generalized Divisive Normalization layer as used in the paper"""
    def __init__(self, channels, inverse=False, beta_min=1e-6, gamma_init=0.1):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.channels = channels
        
        # Learnable parameters
        self.beta = nn.Parameter(torch.ones(channels))
        self.gamma = nn.Parameter(gamma_init * torch.eye(channels))
        
    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        
        # Reshape for matrix operations
        x_flat = x.view(B, C, -1)  # (B, C, D*H*W)
        
        # Compute normalization with numerical stability
        beta = torch.clamp(self.beta, min=self.beta_min)
        
        if self.inverse:
            # iGDN: x * sqrt(beta + gamma @ x^2)
            x_sq = torch.clamp(x_flat ** 2, min=1e-12)  # Prevent zeros
            norm = torch.sqrt(beta.view(1, -1, 1) + torch.einsum('ij,bjk->bik', torch.abs(self.gamma), x_sq) + 1e-12)
            y = x_flat * norm
        else:
            # GDN: x / sqrt(beta + gamma @ x^2)
            x_sq = torch.clamp(x_flat ** 2, min=1e-12)  # Prevent zeros
            norm = torch.sqrt(beta.view(1, -1, 1) + torch.einsum('ij,bjk->bik', torch.abs(self.gamma), x_sq) + 1e-12)
            y = x_flat / (norm + 1e-12)  # Prevent division by zero
            
        return y.view(B, C, D, H, W)


class Conv3DBlock(nn.Module):
    """3D Convolutional block as specified in the paper"""
    def __init__(self, in_channels, out_channels, stride=2):
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # Use BatchNorm instead of GDN for better stability
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DeConv3DBlock(nn.Module):
    """3D Deconvolution block with batch norm and ReLU"""
    def __init__(self, in_channels, out_channels, stride=2):
        super(DeConv3DBlock, self).__init__()
        
        # For stride=1, we don't need output padding
        output_padding = 1 if stride > 1 else 0
        
        self.deconv1 = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1, output_padding=output_padding
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class SWAE3DEncoder5x5x5(nn.Module):
    """3D SWAE Encoder for 5x5x5 blocks"""
    def __init__(self, channels=[32, 64, 128, 256], latent_dim=16):
        super(SWAE3DEncoder5x5x5, self).__init__()
        
        # Input: 5x5x5x1 -> Conv blocks
        self.conv_blocks = nn.ModuleList()
        
        # First block: 1 -> first channel (5x5x5 -> 3x3x3)
        self.conv_blocks.append(Conv3DBlock(1, channels[0], stride=2))
        
        # Second block: stride=2 (3x3x3 -> 2x2x2)
        self.conv_blocks.append(Conv3DBlock(channels[0], channels[1], stride=2))
        
        # Remaining blocks use stride=1 to maintain 2x2x2 spatial dimensions
        for i in range(1, len(channels)-1):
            self.conv_blocks.append(Conv3DBlock(channels[i], channels[i+1], stride=1))
        
        # Calculate FC input size: channels[-1] * 2 * 2 * 2
        self.fc_input_size = channels[-1] * 2 * 2 * 2
        
        # Fully connected layer to latent space
        self.fc = nn.Linear(self.fc_input_size, latent_dim)
        
    def forward(self, x):
        # x: (B, 1, 5, 5, 5)
        for block in self.conv_blocks:
            x = block(x)
        
        # x: (B, 256, 2, 2, 2)
        x = x.view(x.size(0), -1)  # Flatten to (B, 256*2*2*2)
        x = self.fc(x)  # (B, latent_dim)
        
        return x


class SWAE3DDecoder5x5x5(nn.Module):
    """3D SWAE Decoder for 5x5x5 blocks"""
    def __init__(self, channels=[256, 128, 64, 32], latent_dim=16):
        super(SWAE3DDecoder5x5x5, self).__init__()
        
        # Calculate FC output size: channels[0] * 2 * 2 * 2
        self.fc_output_size = channels[0] * 2 * 2 * 2
        
        # Fully connected layer from latent space
        self.fc = nn.Linear(latent_dim, self.fc_output_size)
        
        # Deconv blocks (reverse of encoder)
        self.deconv_blocks = nn.ModuleList()
        
        # First blocks use stride=1 to maintain 2x2x2 spatial dimensions
        for i in range(len(channels)-2):
            self.deconv_blocks.append(DeConv3DBlock(channels[i], channels[i+1], stride=1))
        
        # Second-to-last block: stride=2 (2x2x2 -> 4x4x4)
        self.deconv_blocks.append(DeConv3DBlock(channels[-2], channels[-1], stride=2))
        
        # Final deconv block: 4x4x4 -> 5x5x5 using kernel_size=2, stride=1, padding=0
        self.final_deconv = nn.ConvTranspose3d(channels[-1], 1, kernel_size=2, stride=1, padding=0)
        
    def forward(self, x):
        # x: (B, latent_dim)
        x = self.fc(x)  # (B, fc_output_size)
        x = x.view(x.size(0), -1, 2, 2, 2)  # (B, channels[0], 2, 2, 2)
        
        # Apply deconv blocks
        for block in self.deconv_blocks:
            x = block(x)
        
        # Final deconv to get exact 5x5x5
        x = self.final_deconv(x)  # (B, 1, 5, 5, 5)
        return x


def sliced_wasserstein_distance(encoded_samples, prior_samples, num_projections=50):
    """
    Compute sliced Wasserstein distance as specified in the paper
    O(n log n) complexity as mentioned in the paper
    """
    batch_size, latent_dim = encoded_samples.shape
    device = encoded_samples.device
    
    # Sample random projections from unit sphere (S^{d-1})
    projections = torch.randn(num_projections, latent_dim, device=device)
    projections = F.normalize(projections, dim=1)
    
    # Project samples onto random directions
    encoded_projections = torch.matmul(encoded_samples, projections.t())  # (batch_size, num_projections)
    prior_projections = torch.matmul(prior_samples, projections.t())      # (batch_size, num_projections)
    
    # Sort projections (this is the O(n log n) step mentioned in paper)
    encoded_projections, _ = torch.sort(encoded_projections, dim=0)
    prior_projections, _ = torch.sort(prior_projections, dim=0)
    
    # Compute L2 distance between sorted projections
    wasserstein_distance = torch.mean((encoded_projections - prior_projections) ** 2)
    
    return wasserstein_distance


class SWAE3D5x5x5(nn.Module):
    """
    Pure SWAE 3D Autoencoder for 5x5x5 U_CHI Data
    Based on "Exploring Autoencoder-based Error-bounded Compression for Scientific Data"
    """
    def __init__(self, latent_dim=16, lambda_reg=0.9,
                 encoder_cls=None, decoder_cls=None,
                 channels=[32, 64, 128, 256]):
        super(SWAE3D5x5x5, self).__init__()
        
        # Pick architecture
        if encoder_cls is None:
            encoder_cls = SWAE3DEncoder5x5x5  # conv default
        if decoder_cls is None:
            decoder_cls = SWAE3DDecoder5x5x5
        
        # Instantiate encoder
        if hasattr(encoder_cls, '__init__'):
            import inspect
            sig = inspect.signature(encoder_cls.__init__)
            if 'channels' in sig.parameters:
                self.encoder = encoder_cls(channels, latent_dim)
            else:
                self.encoder = encoder_cls(latent_dim=latent_dim)
        else:
            self.encoder = encoder_cls(latent_dim=latent_dim)
        
        # Instantiate decoder
        if hasattr(decoder_cls, '__init__'):
            import inspect
            sig = inspect.signature(decoder_cls.__init__)
            if 'channels' in sig.parameters:
                self.decoder = decoder_cls(list(reversed(channels)), latent_dim)
            else:
                self.decoder = decoder_cls(latent_dim=latent_dim)
        else:
            self.decoder = decoder_cls(latent_dim=latent_dim)
        
        self.lambda_reg = lambda_reg
        self.latent_dim = latent_dim
        
    def forward(self, x):
        # Handle different architectures
        if hasattr(self.encoder, 'q') and hasattr(self.encoder.q, '__class__'):
            # mlp_opt architecture returns (q, scale) tuple
            q, scale = self.encoder(x)
            x_recon = self.decoder(q, scale)
            latent = q  # Use quantized latent for loss computation
        else:
            # Standard architectures (conv, mlp, gmlp)
            latent = self.encoder(x)
            x_recon = self.decoder(latent)
        
        return x_recon, latent
    
    def loss_function(self, x, x_recon, z, prior_dist='normal'):
        """
        SWAE Loss function as specified in Equation (1) of the paper:
        L(φ, ψ) = (1/M) * Σ c(x_m, ψ(φ(x_m))) + (λ/LM) * Σ c(θ_l · z_i[m], θ_l · φ(x_j[m]))
        
        Where c(x, y) = ||x - y||^2 (Equation 2)
        
        Handles both standard float latents and INT8 quantized latents
        """
        batch_size = x.size(0)
        
        # Reconstruction loss: ||x - x_recon||^2
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # Convert quantized latent to float if needed
        if z.dtype == torch.int8:
            z_float = z.float()
        else:
            z_float = z
        
        # Sample from prior distribution (standard normal as commonly used)
        if prior_dist == 'normal':
            prior_samples = torch.randn(batch_size, self.latent_dim, device=x.device)
        else:
            # Could implement other priors here
            prior_samples = torch.randn(batch_size, self.latent_dim, device=x.device)
        
        # Sliced Wasserstein distance between encoded samples and prior
        sw_distance = sliced_wasserstein_distance(z_float, prior_samples)
        
        # Total loss (Equation 3 in paper)
        total_loss = recon_loss + self.lambda_reg * sw_distance
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'sw_distance': sw_distance
        }
    
    def encode(self, x):
        """Encode input to latent space"""
        if hasattr(self.encoder, 'q') and hasattr(self.encoder.q, '__class__'):
            # mlp_opt architecture returns (q, scale) tuple
            return self.encoder(x)
        else:
            # Standard architectures
            return self.encoder(x)
    
    def decode(self, z, scale=None):
        """Decode from latent space"""
        if hasattr(self.encoder, 'q') and hasattr(self.encoder.q, '__class__'):
            # mlp_opt architecture needs (q, scale) tuple
            if scale is None:
                raise ValueError("mlp_opt architecture requires scale parameter for decoding")
            return self.decoder(z, scale)
        else:
            # Standard architectures
            return self.decoder(z)
    
    def reconstruct(self, x):
        """Full reconstruction pipeline"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon


def create_swae_3d_5x5x5_model(latent_dim=16, lambda_reg=10.0, encoder_channels=None, arch: str = "conv"):
    """
    Create SWAE 3D model for 5x5x5 blocks
    
    Args:
        latent_dim: Latent vector dimension (16 as per Table VI)
        lambda_reg: Regularization weight for SW distance (10.0 default)
        encoder_channels: List of channel sizes for the encoder (will be reversed for decoder)
        arch: Architecture type ('conv', 'mlp', 'gmlp')
    
    Returns:
        SWAE3D5x5x5 model
    """
    # Default channels if not specified
    if encoder_channels is None:
        encoder_channels = [32, 64, 128, 256]
    
    # pick enc/dec classes
    if arch == "conv":
        encoder_cls, decoder_cls = SWAE3DEncoder5x5x5, SWAE3DDecoder5x5x5
    elif arch == "mlp":
        from .swae_mlp_3d_5x5x5 import MLPEncoder5x5x5, MLPDecoder5x5x5
        encoder_cls, decoder_cls = MLPEncoder5x5x5, MLPDecoder5x5x5
    elif arch == "gmlp":
        from .swae_mlp_3d_5x5x5 import gMLPEncoder5x5x5, gMLPDecoder5x5x5
        encoder_cls, decoder_cls = gMLPEncoder5x5x5, gMLPDecoder5x5x5
    elif arch == "mlp_opt":
        from .swae_mlp_opt_3d_5x5x5 import (
            FastMLPEncoder5x5x5 as Enc,
            FastMLPDecoder5x5x5 as Dec,
        )
        encoder_cls, decoder_cls = Enc, Dec
    else:
        raise ValueError(f"Unknown arch {arch}")
    
    model = SWAE3D5x5x5(
        channels=encoder_channels,
        latent_dim=latent_dim,
        lambda_reg=lambda_reg,
        encoder_cls=encoder_cls,
        decoder_cls=decoder_cls
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_swae_3d_5x5x5_model().to(device)
    
    # Test with dummy data (batch of 5x5x5 blocks)
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 5, 5, 5).to(device)
    
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass
    x_recon, z = model(test_input)
    
    print(f"Latent shape: {z.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    
    # Compute loss
    loss_dict = model.loss_function(test_input, x_recon, z)
    
    print(f"Total loss: {loss_dict['loss']:.6f}")
    print(f"Reconstruction loss: {loss_dict['recon_loss']:.6f}")
    print(f"SW distance: {loss_dict['sw_distance']:.6f}")
    
    print("\nSWAE 3D 5x5x5 model created successfully!") 