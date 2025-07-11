#!/usr/bin/env python3
"""
MLP-based SWAE components for 5×5×5 blocks.
Keeps the SWAE loss — only the architecture is different.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 1-a  Simple GELU MLP variant -----------------------------------------
class MLPEncoder5x5x5(nn.Module):
    """
    Encoder: 125 → hidden → latent (default 512→16).
    """
    def __init__(self, latent_dim: int = 16, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                            # (B, 125)
            nn.Linear(125, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim)            # (B, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class MLPDecoder5x5x5(nn.Module):
    """
    Decoder mirrors the encoder: latent → hidden → 125 → reshape.
    """
    def __init__(self, latent_dim: int = 16, hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, 125)

    def forward(self, z):
        x = self.act(self.fc1(z))
        x = self.fc2(x)                              # (B, 125)
        return x.view(-1, 1, 5, 5, 5)


# --- 1-b  (Optional) gMLP block drop-in ------------------------------------
class gMLPBlock(nn.Module):
    """Single Multi-Gate Perceptron block (d_model must divide by 2)."""
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        u, v = self.fc1(x).chunk(2, dim=-1)          # gating
        return self.proj(F.gelu(u) * v) + self.fc2(x)


class gMLPEncoder5x5x5(nn.Module):
    """
    Encoder with two gMLP blocks before the bottleneck.
    """
    def __init__(self, latent_dim: int = 16, d_model: int = 256):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Flatten(),                            # (B, 125)
            nn.Linear(125, d_model),
            nn.GELU()
        )
        self.block1 = gMLPBlock(d_model)
        self.block2 = gMLPBlock(d_model)
        self.to_latent = nn.Linear(d_model, latent_dim)

    def forward(self, x):
        x = self.pre(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.to_latent(x)


class gMLPDecoder5x5x5(nn.Module):
    """
    Decoder with gMLP blocks: latent → d_model → two gMLP blocks → 125 → reshape.
    """
    def __init__(self, latent_dim: int = 16, d_model: int = 256):
        super().__init__()
        self.from_latent = nn.Linear(latent_dim, d_model)
        self.block1 = gMLPBlock(d_model)
        self.block2 = gMLPBlock(d_model)
        self.post = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, 125)
        )

    def forward(self, z):
        x = self.from_latent(z)
        x = self.block1(x)
        x = self.block2(x)
        x = self.post(x)                             # (B, 125)
        return x.view(-1, 1, 5, 5, 5)


if __name__ == "__main__":
    # Test the MLP models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test MLP encoder/decoder
    print("Testing MLP Encoder/Decoder...")
    mlp_encoder = MLPEncoder5x5x5(latent_dim=16, hidden=512).to(device)
    mlp_decoder = MLPDecoder5x5x5(latent_dim=16, hidden=512).to(device)
    
    # Test with dummy data
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 5, 5, 5).to(device)
    
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass
    latent = mlp_encoder(test_input)
    reconstructed = mlp_decoder(latent)
    
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Count parameters
    mlp_encoder_params = sum(p.numel() for p in mlp_encoder.parameters())
    mlp_decoder_params = sum(p.numel() for p in mlp_decoder.parameters())
    total_mlp_params = mlp_encoder_params + mlp_decoder_params
    
    print(f"MLP Encoder parameters: {mlp_encoder_params:,}")
    print(f"MLP Decoder parameters: {mlp_decoder_params:,}")
    print(f"Total MLP parameters: {total_mlp_params:,}")
    
    # Test gMLP encoder/decoder
    print("\nTesting gMLP Encoder/Decoder...")
    gmlp_encoder = gMLPEncoder5x5x5(latent_dim=16, d_model=256).to(device)
    gmlp_decoder = gMLPDecoder5x5x5(latent_dim=16, d_model=256).to(device)
    
    # Forward pass
    latent = gmlp_encoder(test_input)
    reconstructed = gmlp_decoder(latent)
    
    print(f"gMLP Latent shape: {latent.shape}")
    print(f"gMLP Reconstructed shape: {reconstructed.shape}")
    
    # Count parameters
    gmlp_encoder_params = sum(p.numel() for p in gmlp_encoder.parameters())
    gmlp_decoder_params = sum(p.numel() for p in gmlp_decoder.parameters())
    total_gmlp_params = gmlp_encoder_params + gmlp_decoder_params
    
    print(f"gMLP Encoder parameters: {gmlp_encoder_params:,}")
    print(f"gMLP Decoder parameters: {gmlp_decoder_params:,}")
    print(f"Total gMLP parameters: {total_gmlp_params:,}")
    
    print("\nMLP models created successfully!") 