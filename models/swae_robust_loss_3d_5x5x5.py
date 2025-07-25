"""
SWAE model with robust loss functions for variables near zero
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .swae_pure_3d_5x5x5_opt import SWAE3D5x5x5


class SWAERobustLoss3D5x5x5(SWAE3D5x5x5):
    """SWAE with robust loss options for problematic variables"""
    
    def __init__(self, latent_dim=16, lambda_reg=0.1, channels=[32, 64, 128, 256], 
                 encoder_cls=None, decoder_cls=None, loss_type='log_cosh', huber_delta=1.0):
        super().__init__(latent_dim=latent_dim, lambda_reg=lambda_reg, 
                         encoder_cls=encoder_cls, decoder_cls=decoder_cls, channels=channels)
        
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        
    def log_cosh_loss(self, y_true, y_pred):
        """Log-cosh loss: log(cosh(pred - true))
        Behaves like L2 for small errors and L1 for large errors"""
        def log_cosh(x):
            # Numerically stable computation
            return x + F.softplus(-2. * x) - torch.log(torch.tensor(2.0))
        
        diff = y_pred - y_true
        return torch.mean(log_cosh(diff))
    
    def huber_loss(self, y_true, y_pred):
        """Huber loss: quadratic for small errors, linear for large errors"""
        return F.smooth_l1_loss(y_pred, y_true, beta=self.huber_delta)
    
    def loss_function(self, x, x_recon, z):
        """
        Compute loss with robust loss functions
        
        Args:
            x: original input
            x_recon: reconstructed input
            z: latent representation
            
        Returns:
            dict with loss components
        """
        batch_size = x.shape[0]
        
        # Reconstruction loss
        if self.loss_type == 'log_cosh':
            recon_loss = self.log_cosh_loss(x, x_recon)
        elif self.loss_type == 'huber':
            recon_loss = self.huber_loss(x, x_recon)
        elif self.loss_type == 'mae':
            recon_loss = F.l1_loss(x_recon, x, reduction='mean')
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


def create_swae_robust_loss_model(latent_dim=16, lambda_reg=0.1, encoder_channels=[32, 64, 128, 256],
                                  arch='conv', loss_type='log_cosh'):
    """Factory function to create SWAE model with robust loss"""
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
    
    return SWAERobustLoss3D5x5x5(
        latent_dim=latent_dim,
        lambda_reg=lambda_reg,
        channels=encoder_channels,
        encoder_cls=encoder_cls,
        decoder_cls=decoder_cls,
        loss_type=loss_type
    )