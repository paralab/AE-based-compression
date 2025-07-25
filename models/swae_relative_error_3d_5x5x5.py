"""
SWAE model with relative error loss option for 3D data (5x5x5)
Especially useful for variables with values near zero
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .swae_pure_3d_5x5x5_opt import SWAE3D5x5x5


class SWAERelativeError3D5x5x5(SWAE3D5x5x5):
    """SWAE with relative error reconstruction loss option"""
    
    def __init__(self, latent_dim=16, lambda_reg=0.1, channels=[32, 64, 128, 256], 
                 encoder_cls=None, decoder_cls=None, use_relative_error=True, rel_error_epsilon=1e-6):
        super().__init__(latent_dim=latent_dim, lambda_reg=lambda_reg, 
                         encoder_cls=encoder_cls, decoder_cls=decoder_cls, channels=channels)
        
        self.use_relative_error = use_relative_error
        self.rel_error_epsilon = rel_error_epsilon
        
    def loss_function(self, x, x_recon, z):
        """
        Compute loss with optional relative error
        
        Args:
            x: original input
            x_recon: reconstructed input
            z: latent representation
            
        Returns:
            dict with loss components
        """
        batch_size = x.shape[0]
        
        # Reconstruction loss
        if self.use_relative_error:
            # Hybrid loss: Weighted combination of MSE and bounded relative error
            mse_loss = F.mse_loss(x_recon, x, reduction='mean')
            
            # Bounded relative error with larger epsilon and clamping
            abs_diff = torch.abs(x - x_recon)
            abs_x = torch.abs(x)
            
            # Use a larger epsilon based on data scale
            epsilon = torch.maximum(
                torch.tensor(self.rel_error_epsilon).to(x.device),
                0.1 * torch.std(x)  # Adaptive epsilon based on data scale
            )
            
            # Compute relative error with bounds
            relative_errors = abs_diff / (abs_x + epsilon)
            # Clamp extreme values to prevent explosion
            relative_errors = torch.clamp(relative_errors, min=0, max=10.0)
            rel_loss = torch.mean(relative_errors)
            
            # Weighted combination (mostly MSE with some relative guidance)
            recon_loss = 0.9 * mse_loss + 0.1 * rel_loss
        else:
            # Standard MSE loss
            recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # Sliced Wasserstein distance (unchanged)
        z_prior = torch.randn_like(z)
        # Import the function from the parent module
        from .swae_pure_3d_5x5x5_opt import sliced_wasserstein_distance
        sw_distance = sliced_wasserstein_distance(z, z_prior)
        
        # Total loss
        loss = recon_loss + self.lambda_reg * sw_distance
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'sw_distance': sw_distance
        }


def create_swae_relative_error_model(latent_dim=16, lambda_reg=0.1, encoder_channels=[32, 64, 128, 256],
                                    arch='conv', use_relative_error=True):
    """Factory function to create SWAE model with relative error option"""
    # Import encoder/decoder classes based on architecture
    encoder_cls = None
    decoder_cls = None
    
    if arch == "mlp":
        from .swae_mlp_3d_5x5x5 import MLPEncoder5x5x5, MLPDecoder5x5x5
        encoder_cls, decoder_cls = MLPEncoder5x5x5, MLPDecoder5x5x5
    elif arch == "gmlp":
        from .swae_mlp_3d_5x5x5 import gMLPEncoder5x5x5, gMLPDecoder5x5x5
        encoder_cls, decoder_cls = gMLPEncoder5x5x5, gMLPDecoder5x5x5
    # else: use default conv architecture (None will trigger default in parent class)
    
    return SWAERelativeError3D5x5x5(
        latent_dim=latent_dim,
        lambda_reg=lambda_reg,
        channels=encoder_channels,
        encoder_cls=encoder_cls,
        decoder_cls=decoder_cls,
        use_relative_error=use_relative_error
    )