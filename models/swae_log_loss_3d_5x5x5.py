"""
SWAE model with log-space loss to handle vanishing gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .swae_pure_3d_5x5x5_opt import SWAE3D5x5x5


class SWAELogLoss3D5x5x5(SWAE3D5x5x5):
    """SWAE with log-space loss to prevent vanishing gradients"""
    
    def __init__(self, latent_dim=16, lambda_reg=0.1, channels=[32, 64, 128, 256], 
                 encoder_cls=None, decoder_cls=None, loss_scale=1000.0, use_log_loss=True):
        super().__init__(latent_dim=latent_dim, lambda_reg=lambda_reg, 
                         encoder_cls=encoder_cls, decoder_cls=decoder_cls, channels=channels)
        
        self.loss_scale = loss_scale
        self.use_log_loss = use_log_loss
        
    def loss_function(self, x, x_recon, z):
        """
        Compute loss with log-space or scaled MSE to prevent vanishing gradients
        
        Args:
            x: original input
            x_recon: reconstructed input
            z: latent representation
            
        Returns:
            dict with loss components
        """
        batch_size = x.shape[0]
        
        # Reconstruction loss
        mse = F.mse_loss(x_recon, x, reduction='mean')
        
        if self.use_log_loss:
            # Log-space MSE: Use log(1 + scaled_mse) which is smooth and prevents vanishing gradients
            # This formulation:
            # - For small MSE: log(1 + x) ≈ x (linear behavior, good gradients)
            # - For large MSE: log(1 + x) ≈ log(x) (logarithmic growth, prevents explosion)
            # - Always positive and smooth
            
            # Scale MSE to reasonable range before log
            scaled_mse = mse * self.loss_scale
            
            # Apply smooth log transformation
            recon_loss = torch.log1p(scaled_mse)  # log(1 + x) is more stable than log(x)
            
        else:
            # Simple scaling approach
            recon_loss = self.loss_scale * mse
        
        # Sliced Wasserstein distance (unchanged)
        z_prior = torch.randn_like(z)
        from .swae_pure_3d_5x5x5_opt import sliced_wasserstein_distance
        sw_distance = sliced_wasserstein_distance(z, z_prior)
        
        # Total loss
        if self.use_log_loss:
            # Apply same log transformation to SW distance
            scaled_sw = sw_distance * self.loss_scale
            sw_loss = torch.log1p(scaled_sw)
            loss = recon_loss + self.lambda_reg * sw_loss
        else:
            loss = recon_loss + self.lambda_reg * self.loss_scale * sw_distance
        
        # Return actual MSE for monitoring (not the transformed version)
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'sw_distance': sw_distance,
            'actual_mse': mse  # Track actual MSE for monitoring
        }


def create_swae_log_loss_model(latent_dim=16, lambda_reg=0.1, encoder_channels=[32, 64, 128, 256],
                               arch='conv', loss_scale=1000.0, use_log_loss=True):
    """Factory function to create SWAE model with log-space loss"""
    # Import encoder/decoder classes based on architecture
    encoder_cls = None
    decoder_cls = None
    
    if arch == "mlp":
        from .swae_mlp_3d_5x5x5 import MLPEncoder5x5x5, MLPDecoder5x5x5
        encoder_cls, decoder_cls = MLPEncoder5x5x5, MLPDecoder5x5x5
    elif arch == "gmlp":
        from .swae_mlp_3d_5x5x5 import gMLPEncoder5x5x5, gMLPDecoder5x5x5
        encoder_cls, decoder_cls = gMLPEncoder5x5x5, gMLPDecoder5x5x5
    
    return SWAELogLoss3D5x5x5(
        latent_dim=latent_dim,
        lambda_reg=lambda_reg,
        channels=encoder_channels,
        encoder_cls=encoder_cls,
        decoder_cls=decoder_cls,
        loss_scale=loss_scale,
        use_log_loss=use_log_loss
    )