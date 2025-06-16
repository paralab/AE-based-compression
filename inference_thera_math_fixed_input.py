#!/usr/bin/env python3

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import argparse

import models
import datasets
from utils import make_coord, to_pixel_samples


def load_model(model_path, config_path=None):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    model_spec = checkpoint['model']
    model = models.make(model_spec, load_sd=True)
    model = model.cuda()
    model.eval()
    return model


def create_test_function(k1, k2, resolution=256):
    """Create a test mathematical function at high resolution"""
    if isinstance(k1, str) and k1.startswith('super'):
        # Handle superposition case
        k_pairs = [(2, 2), (6, 6)]  # Superposition of (2,2) and (6,6)
        return datasets.math_function.MathFunctionDataset.generate_superposition(k_pairs, resolution)
    
    # Regular single-frequency case
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create function: sin(2πk1*x) * sin(2πk2*y)
    func = np.sin(2 * np.pi * k1 * X) * np.sin(2 * np.pi * k2 * Y)
    
    # Convert to tensor and add channel dimension
    func_tensor = torch.FloatTensor(func).unsqueeze(0)  # (1, H, W)
    return func_tensor


def bicubic_upsampling(lr_func, scale_factor):
    """Perform bicubic upsampling"""
    lr_func = lr_func.unsqueeze(0)  # Add batch dimension
    hr_func = F.interpolate(lr_func, scale_factor=scale_factor, mode='bicubic', align_corners=False)
    return hr_func.squeeze(0)  # Remove batch dimension


def test_model_on_function(model, lr_func, scale_factor, hr_resolution):
    """Test model on a low-resolution function"""
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension to input
        lr_func_batch = lr_func.unsqueeze(0).cuda()  # (1, 1, H, W)
        
        # Generate features from low-resolution input
        model.gen_feat(lr_func_batch)
        
        # Create coordinates for high-resolution output
        coord = make_coord((hr_resolution, hr_resolution)).cuda()
        coord = coord.unsqueeze(0)  # Add batch dimension: (1, H*W, 2)
        
        # Create cell information - represents the area each pixel covers
        cell = torch.ones_like(coord)
        cell[:, :, 0] *= 2 / hr_resolution  # Cell width
        cell[:, :, 1] *= 2 / hr_resolution  # Cell height
        
        # Create scale information - this should be consistent with training
        scale = torch.tensor([scale_factor], dtype=torch.float32).cuda()
        
        # Query high-resolution values
        # Note: The model expects (batch, num_points, 2) for coord and cell
        pred = model.query_rgb(coord, cell, scale)
        
        # Reshape to image format
        pred = pred.view(1, hr_resolution, hr_resolution, 1)
        pred = pred.permute(0, 3, 1, 2).squeeze(0)  # (1, H, W)
        
    return pred.cpu()


def calculate_metrics(pred, gt):
    """Calculate L1 loss and PSNR"""
    l1_loss = F.l1_loss(pred, gt).item()
    
    # Calculate PSNR
    mse = F.mse_loss(pred, gt).item()
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    return l1_loss, psnr


def visualize_results(gt, lr, bicubic, thera, liif, k1, k2, scale_factor, save_path):
    """Visualize and save comparison results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Convert tensors to numpy for visualization
    gt_np = gt.squeeze().numpy()
    lr_np = lr.squeeze().numpy()
    bicubic_np = bicubic.squeeze().numpy()
    thera_np = thera.squeeze().numpy()
    liif_np = liif.squeeze().numpy() if liif is not None else None
    
    # Plot images
    im1 = axes[0, 0].imshow(gt_np, cmap='viridis', vmin=-1, vmax=1)
    if isinstance(k1, str) and k1 == 'super':
        axes[0, 0].set_title(f'Ground Truth ({gt.shape[-1]}×{gt.shape[-2]})\nSuperposition: sin(4πx)sin(4πy) + sin(12πx)sin(12πy)')
    else:
        axes[0, 0].set_title(f'Ground Truth ({gt.shape[-1]}×{gt.shape[-2]})\nsin(2π{k1}x)sin(2π{k2}y)')
    axes[0, 0].axis('off')
    
    im2 = axes[0, 1].imshow(lr_np, cmap='viridis', vmin=-1, vmax=1)
    axes[0, 1].set_title(f'Low Resolution (64×64)\nDownsampled Input')
    axes[0, 1].axis('off')
    
    im3 = axes[0, 2].imshow(bicubic_np, cmap='viridis', vmin=-1, vmax=1)
    axes[0, 2].set_title(f'Bicubic Upsampling\n{scale_factor}× scale → {bicubic_np.shape[0]}×{bicubic_np.shape[1]}')
    axes[0, 2].axis('off')
    
    im4 = axes[1, 0].imshow(thera_np, cmap='viridis', vmin=-1, vmax=1)
    axes[1, 0].set_title(f'Thera (Neural Heat Fields)\n{scale_factor}× scale → {thera_np.shape[0]}×{thera_np.shape[1]}')
    axes[1, 0].axis('off')
    
    if liif_np is not None:
        im5 = axes[1, 1].imshow(liif_np, cmap='viridis', vmin=-1, vmax=1)
        axes[1, 1].set_title('Original LIIF')
        axes[1, 1].axis('off')
    else:
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, 'LIIF model\nnot available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    # Error visualization
    thera_error = np.abs(thera_np - gt_np)
    im6 = axes[1, 2].imshow(thera_error, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 2].set_title('Thera Error (|pred - gt|)')
    axes[1, 2].axis('off')
    
    # Add colorbar
    plt.colorbar(im1, ax=axes[0, :], orientation='horizontal', fraction=0.05, pad=0.1)
    plt.colorbar(im6, ax=axes[1, 2], orientation='horizontal', fraction=0.05, pad=0.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def parse_test_function(func_spec):
    """Parse test function specification"""
    if func_spec == 'super,0':
        return 'super', 0  # Special case for superposition
    else:
        k1, k2 = map(int, func_spec.split(','))
        return k1, k2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thera_model', required=True, help='Path to trained Thera model')
    parser.add_argument('--liif_model', default=None, help='Path to trained LIIF model for comparison')
    parser.add_argument('--output_dir', default='results/thera_fixed_input_corrected', help='Output directory')
    parser.add_argument('--scale_factors', nargs='+', type=int, default=[2, 4, 6, 8], help='Scale factors to test')
    parser.add_argument('--test_functions', nargs='+', default=['2,2', '3,3', '4,4', '5,5', '6,6', 'super,0'], 
                       help='Test functions as k1,k2 pairs or special cases')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading Thera model...")
    thera_model = load_model(args.thera_model)
    
    liif_model = None
    if args.liif_model:
        print("Loading LIIF model...")
        liif_model = load_model(args.liif_model)
    
    # Test parameters - FIXED INPUT SIZE WITH PROPER DOWNSAMPLING
    input_resolution = 64  # Fixed 64x64 input
    
    print("Fixed Input Resolution (Corrected): 64×64 (downsampled from high-res GT)")
    print(f"Scale factors: {args.scale_factors}")
    print("=" * 60)
    
    results = []
    
    for scale_factor in args.scale_factors:
        hr_resolution = input_resolution * scale_factor  # 128, 256, 512, 1024, 2048
        print(f"\nTesting scale factor: {scale_factor}× (64×64 → {hr_resolution}×{hr_resolution})")
        
        for func_spec in args.test_functions:
            # Parse function specification
            k1, k2 = parse_test_function(func_spec)
            
            # Print appropriate message based on function type
            if isinstance(k1, str) and k1 == 'super':
                print(f"  Testing function: Superposition of sin(4πx)sin(4πy) + sin(12πx)sin(12πy)")
            else:
                print(f"  Testing function: sin(2π{k1}x)sin(2π{k2}y)")
            
            # CORRECTED APPROACH: Create GT at target resolution, then downsample to 64x64
            gt_func = create_test_function(k1, k2, hr_resolution)
            
            # Downsample GT to create 64x64 LR input (proper super-resolution setup)
            lr_func = F.interpolate(gt_func.unsqueeze(0), size=(input_resolution, input_resolution), 
                                  mode='bilinear', align_corners=False).squeeze(0)
            
            print(f"    LR size: {lr_func.shape}, GT size: {gt_func.shape}")
            print(f"    (LR is downsampled from GT, maintaining proper SR relationship)")
            
            # Bicubic upsampling
            bicubic_result = bicubic_upsampling(lr_func, scale_factor)
            
            # Thera prediction
            thera_result = test_model_on_function(thera_model, lr_func, scale_factor, hr_resolution)
            
            # LIIF prediction (if available)
            liif_result = None
            if liif_model:
                liif_result = test_model_on_function(liif_model, lr_func, scale_factor, hr_resolution)
            
            # Calculate metrics
            bicubic_l1, bicubic_psnr = calculate_metrics(bicubic_result, gt_func)
            thera_l1, thera_psnr = calculate_metrics(thera_result, gt_func)
            
            liif_l1, liif_psnr = None, None
            if liif_result is not None:
                liif_l1, liif_psnr = calculate_metrics(liif_result, gt_func)
            
            # Store results
            result = {
                'scale': scale_factor,
                'input_res': input_resolution,
                'output_res': hr_resolution,
                'k1': str(k1),  # Convert to string to handle 'super' case
                'k2': str(k2),
                'bicubic_l1': bicubic_l1, 'bicubic_psnr': bicubic_psnr,
                'thera_l1': thera_l1, 'thera_psnr': thera_psnr,
                'liif_l1': liif_l1, 'liif_psnr': liif_psnr
            }
            results.append(result)
            
            # Print metrics
            print(f"    Bicubic  - L1: {bicubic_l1:.6f}, PSNR: {bicubic_psnr:.2f} dB")
            print(f"    Thera    - L1: {thera_l1:.6f}, PSNR: {thera_psnr:.2f} dB")
            if liif_result is not None:
                print(f"    LIIF     - L1: {liif_l1:.6f}, PSNR: {liif_psnr:.2f} dB")
            
            # Visualize results
            save_path = os.path.join(args.output_dir, 
                                   f'fixed_input_corrected_{"super" if k1=="super" else f"k{k1}k{k2}"}_scale{scale_factor}x.png')
            visualize_results(gt_func, lr_func, bicubic_result, thera_result, liif_result,
                            k1, k2, scale_factor, save_path)
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, 'fixed_input_corrected_results.txt')
    with open(results_file, 'w') as f:
        f.write("Fixed Input Size Results (CORRECTED) - Thera vs Bicubic (64×64 downsampled input)\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            if result['k1'] == 'super':
                f.write(f"Scale: {result['scale']}×, Input: {result['input_res']}×{result['input_res']} → Output: {result['output_res']}×{result['output_res']}\n")
                f.write(f"Function: Superposition of sin(4πx)sin(4πy) + sin(12πx)sin(12πy)\n")
            else:
                f.write(f"Scale: {result['scale']}×, Input: {result['input_res']}×{result['input_res']} → Output: {result['output_res']}×{result['output_res']}\n")
                f.write(f"Function: sin(2π{result['k1']}x)sin(2π{result['k2']}y)\n")
            f.write(f"  Bicubic  - L1: {result['bicubic_l1']:.6f}, PSNR: {result['bicubic_psnr']:.2f} dB\n")
            f.write(f"  Thera    - L1: {result['thera_l1']:.6f}, PSNR: {result['thera_psnr']:.2f} dB\n")
            if result['liif_l1'] is not None:
                f.write(f"  LIIF     - L1: {result['liif_l1']:.6f}, PSNR: {result['liif_psnr']:.2f} dB\n")
            f.write("\n")
    
    print(f"\nResults saved to {args.output_dir}")
    print("Fixed input size inference (CORRECTED) completed!")


if __name__ == '__main__':
    main() 