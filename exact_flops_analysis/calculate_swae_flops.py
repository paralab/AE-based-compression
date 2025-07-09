#!/usr/bin/env python3
"""
Exact FLOP Calculation for SWAE 7x7x7 Architecture
Breaks down every operation to verify the 2.4B FLOP claim
"""

import math

def calculate_conv3d_flops(batch_size, in_channels, out_channels, kernel_size, input_shape, output_shape):
    """
    Calculate FLOPs for a 3D convolution layer
    FLOPs = batch_size × output_volume × (input_channels × kernel_volume + bias_ops)
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    
    # Output volume
    output_volume = output_shape[0] * output_shape[1] * output_shape[2] * out_channels
    
    # Kernel volume  
    kernel_volume = kernel_size[0] * kernel_size[1] * kernel_size[2]
    
    # FLOPs per output element = (input_channels × kernel_volume) + 1 bias
    flops_per_output = (in_channels * kernel_volume) + 1
    
    # Total FLOPs
    total_flops = batch_size * output_volume * flops_per_output
    
    return total_flops

def calculate_batchnorm3d_flops(batch_size, channels, spatial_shape):
    """
    Calculate FLOPs for 3D Batch Normalization
    BN: (x - mean) / sqrt(var + eps) * gamma + beta
    Operations per element: subtract, divide, multiply, add = 4 ops
    """
    volume = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
    total_elements = batch_size * channels * volume
    
    # 4 operations per element (subtract, divide, multiply, add)
    return total_elements * 4

def calculate_fc_flops(batch_size, input_features, output_features):
    """
    Calculate FLOPs for fully connected layer
    FC: y = Wx + b
    FLOPs = batch_size × output_features × (input_features + 1)
    """
    return batch_size * output_features * (input_features + 1)

def calculate_relu_flops(batch_size, channels, spatial_shape):
    """
    Calculate FLOPs for ReLU activation
    ReLU: max(0, x) = 1 comparison per element
    """
    volume = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
    total_elements = batch_size * channels * volume
    return total_elements  # 1 comparison per element

def calculate_sliced_wasserstein_flops(batch_size, latent_dim, num_projections=50):
    """
    Calculate FLOPs for Sliced Wasserstein Distance
    For each projection:
    1. Random projection: latent_dim operations per sample
    2. Sort: O(batch_size * log(batch_size))
    3. L2 distance: batch_size operations
    """
    flops_per_projection = (
        batch_size * latent_dim +  # Random projection
        batch_size * math.log2(batch_size) * 10 +  # Sort (approximate)
        batch_size  # L2 distance
    )
    
    return num_projections * flops_per_projection

def analyze_swae_architecture(batch_size=32, latent_dim=16):
    """
    Complete FLOP analysis of SWAE 7x7x7 architecture
    """
    
    print("="*80)
    print("EXACT SWAE 7x7x7 FLOP CALCULATION")
    print("="*80)
    print(f"Batch Size: {batch_size}")
    print(f"Latent Dimension: {latent_dim}")
    print(f"Input Shape: (1, 7, 7, 7)")
    print()
    
    total_flops = 0
    
    # =================================================================
    # ENCODER
    # =================================================================
    print("🔍 ENCODER ANALYSIS:")
    print("-" * 40)
    
    # Block 1: 1→32 channels, 7×7×7 → 4×4×4
    print("Block 1 (7×7×7 → 4×4×4):")
    
    # Conv3D 1: 1→32, kernel=3, stride=2
    conv1_flops = calculate_conv3d_flops(batch_size, 1, 32, 3, (7,7,7), (4,4,4))
    print(f"  Conv3D 1→32:     {conv1_flops:,} FLOPs")
    total_flops += conv1_flops
    
    # Conv3D 2: 32→32, kernel=3, stride=1  
    conv2_flops = calculate_conv3d_flops(batch_size, 32, 32, 3, (4,4,4), (4,4,4))
    print(f"  Conv3D 32→32:    {conv2_flops:,} FLOPs")
    total_flops += conv2_flops
    
    # BatchNorm3D: 32 channels
    bn1_flops = calculate_batchnorm3d_flops(batch_size, 32, (4,4,4))
    print(f"  BatchNorm3D:     {bn1_flops:,} FLOPs")
    total_flops += bn1_flops
    
    # ReLU
    relu1_flops = calculate_relu_flops(batch_size, 32, (4,4,4))
    print(f"  ReLU:            {relu1_flops:,} FLOPs")
    total_flops += relu1_flops
    
    block1_total = conv1_flops + conv2_flops + bn1_flops + relu1_flops
    print(f"  Block 1 Total:   {block1_total:,} FLOPs")
    print()
    
    # Block 2: 32→64 channels, 4×4×4 → 2×2×2
    print("Block 2 (4×4×4 → 2×2×2):")
    
    # Conv3D 1: 32→64, kernel=3, stride=2
    conv3_flops = calculate_conv3d_flops(batch_size, 32, 64, 3, (4,4,4), (2,2,2))
    print(f"  Conv3D 32→64:    {conv3_flops:,} FLOPs")
    total_flops += conv3_flops
    
    # Conv3D 2: 64→64, kernel=3, stride=1
    conv4_flops = calculate_conv3d_flops(batch_size, 64, 64, 3, (2,2,2), (2,2,2))
    print(f"  Conv3D 64→64:    {conv4_flops:,} FLOPs")
    total_flops += conv4_flops
    
    # BatchNorm3D: 64 channels
    bn2_flops = calculate_batchnorm3d_flops(batch_size, 64, (2,2,2))
    print(f"  BatchNorm3D:     {bn2_flops:,} FLOPs")
    total_flops += bn2_flops
    
    # ReLU
    relu2_flops = calculate_relu_flops(batch_size, 64, (2,2,2))
    print(f"  ReLU:            {relu2_flops:,} FLOPs")
    total_flops += relu2_flops
    
    block2_total = conv3_flops + conv4_flops + bn2_flops + relu2_flops
    print(f"  Block 2 Total:   {block2_total:,} FLOPs")
    print()
    
    # Block 3: 64→128 channels, 2×2×2 → 1×1×1
    print("Block 3 (2×2×2 → 1×1×1):")
    
    # Conv3D 1: 64→128, kernel=3, stride=2
    conv5_flops = calculate_conv3d_flops(batch_size, 64, 128, 3, (2,2,2), (1,1,1))
    print(f"  Conv3D 64→128:   {conv5_flops:,} FLOPs")
    total_flops += conv5_flops
    
    # Conv3D 2: 128→128, kernel=3, stride=1
    conv6_flops = calculate_conv3d_flops(batch_size, 128, 128, 3, (1,1,1), (1,1,1))
    print(f"  Conv3D 128→128:  {conv6_flops:,} FLOPs")
    total_flops += conv6_flops
    
    # BatchNorm3D: 128 channels
    bn3_flops = calculate_batchnorm3d_flops(batch_size, 128, (1,1,1))
    print(f"  BatchNorm3D:     {bn3_flops:,} FLOPs")
    total_flops += bn3_flops
    
    # ReLU
    relu3_flops = calculate_relu_flops(batch_size, 128, (1,1,1))
    print(f"  ReLU:            {relu3_flops:,} FLOPs")
    total_flops += relu3_flops
    
    block3_total = conv5_flops + conv6_flops + bn3_flops + relu3_flops
    print(f"  Block 3 Total:   {block3_total:,} FLOPs")
    print()
    
    # FC Layer: 128 → latent_dim
    fc_encoder_flops = calculate_fc_flops(batch_size, 128, latent_dim)
    print(f"FC Layer 128→{latent_dim}:  {fc_encoder_flops:,} FLOPs")
    total_flops += fc_encoder_flops
    
    encoder_total = block1_total + block2_total + block3_total + fc_encoder_flops
    print(f"ENCODER TOTAL:     {encoder_total:,} FLOPs")
    print()
    
    # =================================================================
    # DECODER  
    # =================================================================
    print("🔧 DECODER ANALYSIS:")
    print("-" * 40)
    
    # FC Layer: latent_dim → 128
    fc_decoder_flops = calculate_fc_flops(batch_size, latent_dim, 128)
    print(f"FC Layer {latent_dim}→128:  {fc_decoder_flops:,} FLOPs")
    total_flops += fc_decoder_flops
    
    # DeConv Block 1: 128→64, 1×1×1 → 2×2×2
    print("DeConv Block 1 (1×1×1 → 2×2×2):")
    
    # Transposed Conv3D: 128→64, kernel=3, stride=2
    deconv1_flops = calculate_conv3d_flops(batch_size, 128, 64, 3, (1,1,1), (2,2,2))
    print(f"  DeConv3D 128→64: {deconv1_flops:,} FLOPs")
    total_flops += deconv1_flops
    
    # Conv3D: 64→64, kernel=3, stride=1
    conv7_flops = calculate_conv3d_flops(batch_size, 64, 64, 3, (2,2,2), (2,2,2))
    print(f"  Conv3D 64→64:    {conv7_flops:,} FLOPs")
    total_flops += conv7_flops
    
    # BatchNorm3D: 64 channels
    bn4_flops = calculate_batchnorm3d_flops(batch_size, 64, (2,2,2))
    print(f"  BatchNorm3D:     {bn4_flops:,} FLOPs")
    total_flops += bn4_flops
    
    # ReLU
    relu4_flops = calculate_relu_flops(batch_size, 64, (2,2,2))
    print(f"  ReLU:            {relu4_flops:,} FLOPs")
    total_flops += relu4_flops
    
    deconv_block1_total = deconv1_flops + conv7_flops + bn4_flops + relu4_flops
    print(f"  DeConv Block 1:  {deconv_block1_total:,} FLOPs")
    print()
    
    # DeConv Block 2: 64→32, 2×2×2 → 4×4×4
    print("DeConv Block 2 (2×2×2 → 4×4×4):")
    
    # Transposed Conv3D: 64→32, kernel=3, stride=2
    deconv2_flops = calculate_conv3d_flops(batch_size, 64, 32, 3, (2,2,2), (4,4,4))
    print(f"  DeConv3D 64→32:  {deconv2_flops:,} FLOPs")
    total_flops += deconv2_flops
    
    # Conv3D: 32→32, kernel=3, stride=1
    conv8_flops = calculate_conv3d_flops(batch_size, 32, 32, 3, (4,4,4), (4,4,4))
    print(f"  Conv3D 32→32:    {conv8_flops:,} FLOPs")
    total_flops += conv8_flops
    
    # BatchNorm3D: 32 channels
    bn5_flops = calculate_batchnorm3d_flops(batch_size, 32, (4,4,4))
    print(f"  BatchNorm3D:     {bn5_flops:,} FLOPs")
    total_flops += bn5_flops
    
    # ReLU
    relu5_flops = calculate_relu_flops(batch_size, 32, (4,4,4))
    print(f"  ReLU:            {relu5_flops:,} FLOPs")
    total_flops += relu5_flops
    
    deconv_block2_total = deconv2_flops + conv8_flops + bn5_flops + relu5_flops
    print(f"  DeConv Block 2:  {deconv_block2_total:,} FLOPs")
    print()
    
    # Final DeConv: 32→1, 4×4×4 → 7×7×7
    final_deconv_flops = calculate_conv3d_flops(batch_size, 32, 1, 4, (4,4,4), (7,7,7))
    print(f"Final DeConv 32→1: {final_deconv_flops:,} FLOPs")
    total_flops += final_deconv_flops
    
    decoder_total = fc_decoder_flops + deconv_block1_total + deconv_block2_total + final_deconv_flops
    print(f"DECODER TOTAL:     {decoder_total:,} FLOPs")
    print()
    
    # =================================================================
    # SLICED WASSERSTEIN DISTANCE
    # =================================================================
    print("⚡ SLICED WASSERSTEIN DISTANCE:")
    print("-" * 40)
    
    sw_flops = calculate_sliced_wasserstein_flops(batch_size, latent_dim, num_projections=50)
    print(f"SW Distance (50 proj): {sw_flops:,} FLOPs")
    total_flops += sw_flops
    print()
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print("📊 FINAL SUMMARY:")
    print("=" * 40)
    print(f"Encoder FLOPs:     {encoder_total:,}")
    print(f"Decoder FLOPs:     {decoder_total:,}")
    print(f"SW Distance FLOPs: {sw_flops:,}")
    print(f"TOTAL FLOPs:       {total_flops:,}")
    print()
    print(f"Total in Billions: {total_flops / 1e9:.2f}B")
    print(f"Per Sample FLOPs:  {total_flops / batch_size:,}")
    print()
    
    # Compute time estimation
    print("🎯 A100 GPU TIMING ESTIMATION:")
    print("-" * 40)
    
    a100_peak_flops = 19.5e12  # 19.5 TFLOPS for mixed precision
    efficiency_factors = [0.6, 0.65, 0.7, 0.75]
    
    for eff in efficiency_factors:
        effective_flops = a100_peak_flops * eff
        compute_time_seconds = total_flops / effective_flops
        compute_time_ms = compute_time_seconds * 1000
        print(f"  {eff*100:.0f}% efficiency: {compute_time_ms:.2f} ms")
    
    print()
    
    # Verify against claimed 0.23ms
    claimed_time_ms = 0.23
    required_efficiency = (total_flops / batch_size) / (claimed_time_ms / 1000) / a100_peak_flops
    print(f"To achieve 0.23ms per sample:")
    print(f"  Required efficiency: {required_efficiency*100:.1f}%")
    print(f"  This is {'REALISTIC' if required_efficiency < 0.8 else 'OPTIMISTIC'}")

if __name__ == "__main__":
    analyze_swae_architecture(batch_size=32, latent_dim=16) 