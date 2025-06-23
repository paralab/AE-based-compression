#!/usr/bin/env python3
"""
SWAE 3D Architecture Visualization
Creates detailed diagrams showing the pure SWAE reconstruction architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_swae_architecture_diagram():
    """Create a comprehensive SWAE architecture diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#e1f5fe',
        'encoder': '#f3e5f5', 
        'latent': '#fff3e0',
        'decoder': '#e8f5e8',
        'loss': '#ffebee',
        'output': '#f1f8e9'
    }
    
    # Title
    ax.text(10, 13.5, 'Pure SWAE 3D Reconstruction Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    ax.text(10, 13, 'Based on "Exploring Autoencoder-based Error-bounded Compression for Scientific Data"', 
            fontsize=12, ha='center', style='italic')
    
    # Input Data Section
    input_box = FancyBboxPatch((0.5, 11), 3, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 11.75, '3D Scientific Data\n(40×40×40)', 
            fontsize=11, fontweight='bold', ha='center', va='center')
    
    # Block Partitioning
    partition_box = FancyBboxPatch((4.5, 11), 3, 1.5, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['input'], 
                                   edgecolor='black', linewidth=2)
    ax.add_patch(partition_box)
    ax.text(6, 11.75, 'Block Partitioning\n8×8×8 blocks\n(125 blocks total)', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    
    # Encoder Section
    encoder_y = 9
    encoder_blocks = [
        ('Input Block\n(B,1,8,8,8)', 1),
        ('Conv3D+BN+ReLU\n1→32 ch, s=2\n4×4×4', 2.5),
        ('Conv3D+BN+ReLU\n32→64 ch, s=2\n2×2×2', 4),
        ('Conv3D+BN+ReLU\n64→128 ch, s=2\n1×1×1', 5.5),
        ('Flatten+FC\n128→16', 7)
    ]
    
    for i, (text, x_pos) in enumerate(encoder_blocks):
        box = FancyBboxPatch((x_pos-0.6, encoder_y-0.7), 1.2, 1.4, 
                             boxstyle="round,pad=0.05", 
                             facecolor=colors['encoder'], 
                             edgecolor='purple', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x_pos, encoder_y, text, fontsize=9, ha='center', va='center')
        
        # Add arrows between blocks
        if i < len(encoder_blocks) - 1:
            next_x = encoder_blocks[i+1][1]
            arrow = ConnectionPatch((x_pos+0.6, encoder_y), (next_x-0.6, encoder_y), 
                                  "data", "data", arrowstyle="->", 
                                  shrinkA=0, shrinkB=0, mutation_scale=20, fc="black")
            ax.add_artist(arrow)
    
    # Latent Space
    latent_box = FancyBboxPatch((8.5, 8.3), 2, 1.4, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['latent'], 
                                edgecolor='orange', linewidth=3)
    ax.add_patch(latent_box)
    ax.text(9.5, 9, 'Latent Code z\n(B, 16)\nCompressed\nRepresentation', 
            fontsize=11, fontweight='bold', ha='center', va='center')
    
    # Decoder Section
    decoder_y = 6.5
    decoder_blocks = [
        ('FC+Reshape\n16→128\n(B,128,1,1,1)', 7),
        ('DeConv3D+BN+ReLU\n128→64 ch, s=2\n2×2×2', 5.5),
        ('DeConv3D+BN+ReLU\n64→32 ch, s=2\n4×4×4', 4),
        ('DeConv3D (final)\n32→1 ch, s=2\n8×8×8', 2.5),
        ('Output Block\n(B,1,8,8,8)', 1)
    ]
    
    for i, (text, x_pos) in enumerate(decoder_blocks):
        box = FancyBboxPatch((x_pos-0.6, decoder_y-0.7), 1.2, 1.4, 
                             boxstyle="round,pad=0.05", 
                             facecolor=colors['decoder'], 
                             edgecolor='green', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x_pos, decoder_y, text, fontsize=9, ha='center', va='center')
        
        # Add arrows between blocks
        if i < len(decoder_blocks) - 1:
            next_x = decoder_blocks[i+1][1]
            arrow = ConnectionPatch((x_pos-0.6, decoder_y), (next_x+0.6, decoder_y), 
                                  "data", "data", arrowstyle="->", 
                                  shrinkA=0, shrinkB=0, mutation_scale=20, fc="black")
            ax.add_artist(arrow)
    
    # Loss Function Section
    loss_y = 4
    
    # Reconstruction Loss
    recon_loss_box = FancyBboxPatch((11.5, loss_y), 3, 1.2, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=colors['loss'], 
                                    edgecolor='red', linewidth=2)
    ax.add_patch(recon_loss_box)
    ax.text(13, loss_y+0.6, 'Reconstruction Loss\nL_recon = MSE(x, x̂)\n= ||x - x̂||²', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    
    # Prior Samples
    prior_box = FancyBboxPatch((15.5, 8.3), 2, 1.4, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#f0f0f0', 
                               edgecolor='gray', linewidth=2)
    ax.add_patch(prior_box)
    ax.text(16.5, 9, 'Prior Samples\nz_prior ~ N(0,I)\n(B, 16)', 
            fontsize=10, ha='center', va='center')
    
    # Sliced Wasserstein Distance
    sw_loss_box = FancyBboxPatch((15.5, loss_y), 3, 1.2, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['loss'], 
                                 edgecolor='red', linewidth=2)
    ax.add_patch(sw_loss_box)
    ax.text(17, loss_y+0.6, 'Sliced Wasserstein Distance\nL_SW = E[||sort(θᵀz) - sort(θᵀz_prior)||²]\nθ ~ Uniform(S^(d-1))', 
            fontsize=9, fontweight='bold', ha='center', va='center')
    
    # Total Loss
    total_loss_box = FancyBboxPatch((13.5, 2.5), 3, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor='#ffcdd2', 
                                    edgecolor='darkred', linewidth=3)
    ax.add_patch(total_loss_box)
    ax.text(15, 3, 'Total Loss\nL = L_recon + λ·L_SW\nλ = 10.0', 
            fontsize=11, fontweight='bold', ha='center', va='center')
    
    # Block Assembly
    assembly_box = FancyBboxPatch((0.5, 4.5), 3, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['output'], 
                                  edgecolor='darkgreen', linewidth=2)
    ax.add_patch(assembly_box)
    ax.text(2, 5.25, 'Block Assembly\n125 blocks → \n40×40×40 volume', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    
    # Final Output
    output_box = FancyBboxPatch((0.5, 2.5), 3, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], 
                                edgecolor='darkgreen', linewidth=2)
    ax.add_patch(output_box)
    ax.text(2, 3.25, 'Reconstructed\n3D Data\n(40×40×40)', 
            fontsize=11, fontweight='bold', ha='center', va='center')
    
    # Add connecting arrows
    # Input to encoder
    arrow1 = ConnectionPatch((3.5, 11.75), (0.4, 9), 
                            "data", "data", arrowstyle="->", 
                            shrinkA=0, shrinkB=0, mutation_scale=20, fc="black")
    ax.add_artist(arrow1)
    
    # Encoder to latent
    arrow2 = ConnectionPatch((7.6, 9), (8.5, 9), 
                            "data", "data", arrowstyle="->", 
                            shrinkA=0, shrinkB=0, mutation_scale=20, fc="black")
    ax.add_artist(arrow2)
    
    # Latent to decoder
    arrow3 = ConnectionPatch((8.5, 9), (7.6, 6.5), 
                            "data", "data", arrowstyle="->", 
                            shrinkA=0, shrinkB=0, mutation_scale=20, fc="black")
    ax.add_artist(arrow3)
    
    # Decoder to assembly
    arrow4 = ConnectionPatch((0.4, 6.5), (2, 6), 
                            "data", "data", arrowstyle="->", 
                            shrinkA=0, shrinkB=0, mutation_scale=20, fc="black")
    ax.add_artist(arrow4)
    
    # Assembly to output
    arrow5 = ConnectionPatch((2, 4.5), (2, 4), 
                            "data", "data", arrowstyle="->", 
                            shrinkA=0, shrinkB=0, mutation_scale=20, fc="black")
    ax.add_artist(arrow5)
    
    # Loss connections
    # Input to recon loss
    arrow6 = ConnectionPatch((0.4, 9), (11.5, 4.6), 
                            "data", "data", arrowstyle="->", 
                            shrinkA=0, shrinkB=0, mutation_scale=15, fc="red", 
                            linestyle='--', alpha=0.7)
    ax.add_artist(arrow6)
    
    # Output to recon loss
    arrow7 = ConnectionPatch((0.4, 6.5), (11.5, 4.6), 
                            "data", "data", arrowstyle="->", 
                            shrinkA=0, shrinkB=0, mutation_scale=15, fc="red", 
                            linestyle='--', alpha=0.7)
    ax.add_artist(arrow7)
    
    # Latent to SW loss
    arrow8 = ConnectionPatch((10.5, 9), (15.5, 4.6), 
                            "data", "data", arrowstyle="->", 
                            shrinkA=0, shrinkB=0, mutation_scale=15, fc="red", 
                            linestyle='--', alpha=0.7)
    ax.add_artist(arrow8)
    
    # Prior to SW loss
    arrow9 = ConnectionPatch((16.5, 8.3), (16.5, 5.2), 
                            "data", "data", arrowstyle="->", 
                            shrinkA=0, shrinkB=0, mutation_scale=15, fc="red", 
                            linestyle='--', alpha=0.7)
    ax.add_artist(arrow9)
    
    # Losses to total loss
    arrow10 = ConnectionPatch((13, 4), (14, 3.5), 
                             "data", "data", arrowstyle="->", 
                             shrinkA=0, shrinkB=0, mutation_scale=15, fc="darkred")
    ax.add_artist(arrow10)
    
    arrow11 = ConnectionPatch((17, 4), (16, 3.5), 
                             "data", "data", arrowstyle="->", 
                             shrinkA=0, shrinkB=0, mutation_scale=15, fc="darkred")
    ax.add_artist(arrow11)
    
    # Add mathematical formulations box
    math_box = FancyBboxPatch((11.5, 0.5), 7, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#f5f5f5', 
                              edgecolor='black', linewidth=1)
    ax.add_patch(math_box)
    ax.text(15, 1.25, 'Key Equations:\n' +
                      '• Encoder: φ(x) → z ∈ ℝ¹⁶\n' +
                      '• Decoder: ψ(z) → x̂ ∈ ℝ⁸ˣ⁸ˣ⁸\n' +
                      '• SW Distance: SW(P_z, P_prior) = E[W₁(P_z^θ, P_prior^θ)]', 
            fontsize=10, ha='center', va='center', family='monospace')
    
    # Add specifications box
    spec_box = FancyBboxPatch((0.5, 0.5), 7, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#f0f8ff', 
                              edgecolor='blue', linewidth=1)
    ax.add_patch(spec_box)
    ax.text(4, 1.25, 'Implementation Specifications:\n' +
                     '• Block Size: 8×8×8 (as per paper Table VI)\n' +
                     '• Latent Dimension: 16 (compression ratio ~32:1)\n' +
                     '• Architecture: [32, 64, 128] channels\n' +
                     '• Regularization: λ = 10.0, 50 projections', 
            fontsize=10, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('swae_3d_architecture_complete.png', dpi=300, bbox_inches='tight')
    plt.savefig('swae_3d_architecture_complete.pdf', bbox_inches='tight')
    print("Saved SWAE architecture diagram: swae_3d_architecture_complete.png")
    
    return fig

def create_sliced_wasserstein_diagram():
    """Create a detailed diagram explaining the Sliced Wasserstein Distance"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sliced Wasserstein Distance in SWAE', fontsize=16, fontweight='bold')
    
    # 1. Random Projections
    ax1 = axes[0, 0]
    ax1.set_title('1. Random Projections from Unit Sphere', fontsize=12, fontweight='bold')
    
    # Draw unit circle (2D representation of unit sphere)
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax1.add_patch(circle)
    
    # Draw random projection vectors
    np.random.seed(42)
    angles = np.random.uniform(0, 2*np.pi, 8)
    for i, angle in enumerate(angles):
        x, y = np.cos(angle), np.sin(angle)
        ax1.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, 
                 fc=f'C{i}', ec=f'C{i}', linewidth=2, alpha=0.7)
        ax1.text(x*1.15, y*1.15, f'θ_{i+1}', fontsize=10, ha='center', va='center')
    
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.text(0, -1.3, 'θ ~ Uniform(S^(d-1))', fontsize=11, ha='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    
    # 2. Projection Process
    ax2 = axes[0, 1]
    ax2.set_title('2. Project Latent Codes onto Random Directions', fontsize=12, fontweight='bold')
    
    # Simulate latent codes and projections
    z_encoded = np.random.randn(50, 2)  # 2D for visualization
    z_prior = np.random.randn(50, 2)
    
    theta = np.array([0.7, 0.7])  # Example projection direction
    theta = theta / np.linalg.norm(theta)
    
    # Project onto theta
    proj_encoded = np.dot(z_encoded, theta)
    proj_prior = np.dot(z_prior, theta)
    
    ax2.scatter(z_encoded[:, 0], z_encoded[:, 1], alpha=0.6, label='Encoded z', color='red')
    ax2.scatter(z_prior[:, 0], z_prior[:, 1], alpha=0.6, label='Prior z', color='blue')
    ax2.arrow(0, 0, theta[0]*3, theta[1]*3, head_width=0.1, head_length=0.1, 
             fc='green', ec='green', linewidth=3, label='Projection θ')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Sort Projections
    ax3 = axes[1, 0]
    ax3.set_title('3. Sort Projected Values', fontsize=12, fontweight='bold')
    
    proj_encoded_sorted = np.sort(proj_encoded)
    proj_prior_sorted = np.sort(proj_prior)
    
    x_vals = np.arange(len(proj_encoded_sorted))
    ax3.plot(x_vals, proj_encoded_sorted, 'ro-', label='Sorted Encoded Projections', alpha=0.7)
    ax3.plot(x_vals, proj_prior_sorted, 'bo-', label='Sorted Prior Projections', alpha=0.7)
    ax3.fill_between(x_vals, proj_encoded_sorted, proj_prior_sorted, 
                     alpha=0.3, color='yellow', label='L2 Distance')
    ax3.set_xlabel('Sample Index (sorted)')
    ax3.set_ylabel('Projected Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Final Algorithm
    ax4 = axes[1, 1]
    ax4.set_title('4. Sliced Wasserstein Algorithm', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    algorithm_text = """
Algorithm: Sliced Wasserstein Distance

Input: z_encoded, z_prior, num_projections=50

1. For l = 1 to num_projections:
   • Sample θ_l ~ Uniform(S^(d-1))
   • Compute p_l^enc = θ_l^T · z_encoded  
   • Compute p_l^prior = θ_l^T · z_prior
   • Sort: p̃_l^enc = sort(p_l^enc)
   • Sort: p̃_l^prior = sort(p_l^prior)
   • Compute: d_l = ||p̃_l^enc - p̃_l^prior||²

2. Return: SW = (1/L) Σ d_l

Complexity: O(n log n) per projection
Total: O(L·n log n) where L=50, n=batch_size
    """
    
    ax4.text(0.05, 0.95, algorithm_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('sliced_wasserstein_explanation.png', dpi=300, bbox_inches='tight')
    plt.savefig('sliced_wasserstein_explanation.pdf', bbox_inches='tight')
    print("Saved Sliced Wasserstein explanation: sliced_wasserstein_explanation.png")
    
    return fig

if __name__ == "__main__":
    print("Creating SWAE Architecture Diagrams...")
    
    # Create main architecture diagram
    fig1 = create_swae_architecture_diagram()
    
    # Create Sliced Wasserstein explanation
    fig2 = create_sliced_wasserstein_diagram()
    
    plt.show()
    print("\nAll diagrams created successfully!")
    print("Files generated:")
    print("• swae_3d_architecture_complete.png")
    print("• swae_3d_architecture_complete.pdf") 
    print("• sliced_wasserstein_explanation.png")
    print("• sliced_wasserstein_explanation.pdf") 