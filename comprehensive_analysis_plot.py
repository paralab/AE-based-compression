#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def compute_psnr(img1, img2):
    """Compute PSNR between two images"""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def process_image_at_scale(gt_img, scale_factor, input_size=(512, 512)):
    """Process image at a specific scale factor"""
    # Create low-res version first (simulating input condition)
    lr_img = gt_img.resize(input_size, Image.BICUBIC)
    
    # Calculate target size based on scale factor
    target_size = (input_size[0] * scale_factor, input_size[1] * scale_factor)
    
    # Create bicubic upsampled version from low-res to target size
    bicubic_img = lr_img.resize(target_size, Image.BICUBIC)
    
    # Load LIIF output
    output_path = f'outputs/demo_output_{target_size[0]}x{target_size[1]}.png'
    if not os.path.exists(output_path):
        print(f"Warning: {output_path} not found")
        return None, None, None, None
        
    output_img = Image.open(output_path)
    
    # Ground Truth resized to target resolution for comparison
    gt_resized = gt_img.resize(target_size, Image.BICUBIC)
    
    return lr_img, bicubic_img, output_img, gt_resized

def get_center_crop(img, size=400):
    """Get center crop of an image"""
    cx, cy = img.size[0]//2, img.size[1]//2
    return img.crop((cx-size//2, cy-size//2, cx+size//2, cy+size//2))

def normalize_and_compute_diff(img1, img2):
    """Compute normalized difference between two images"""
    arr1 = np.array(img1, dtype=np.float32) / 255.0
    arr2 = np.array(img2, dtype=np.float32) / 255.0
    diff = np.abs(arr1 - arr2)
    return diff, np.mean(diff)

def create_comprehensive_plot(image_path):
    """Create comprehensive Input-Output-GT-Losses visualization with bicubic comparison at multiple scales"""
    
    # Define the scale factors to analyze
    scale_factors = [2, 4, 8]  # 2x, 4x, 8x
    input_size = (512, 512)  # Base input size
    
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found")
        return
            
    print(f"\nProcessing {os.path.basename(image_path)}")
    gt_img = Image.open(image_path)
    
    # Create figure
    fig = plt.figure(figsize=(25, 15))  # Adjusted size for 3 rows
    
    for idx, scale in enumerate(scale_factors):
        print(f"  Processing {scale}x scale")
        
        # Process image at current scale
        lr_img, bicubic_img, output_img, gt_resized = process_image_at_scale(gt_img, scale, input_size)
        
        if output_img is None:
            print(f"  Warning: LIIF output not found for {scale}x scale")
            continue
        
        # Get crops for visualization
        gt_crop = get_center_crop(gt_resized)
        bicubic_crop = get_center_crop(bicubic_img)
        output_crop = get_center_crop(output_img)
        lr_crop = get_center_crop(lr_img)
        
        # Compute PSNR
        bicubic_psnr = compute_psnr(bicubic_img, gt_resized)
        liif_psnr = compute_psnr(output_img, gt_resized)
        
        # Plot the results for this scale
        # Input (Low-res)
        ax1 = plt.subplot(3, 6, idx*6 + 1)
        ax1.imshow(lr_crop)
        ax1.set_title(f'Input ({input_size[0]}×{input_size[1]})\n{scale}x upscaling', 
                     fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        # Bicubic Upsampled
        ax2 = plt.subplot(3, 6, idx*6 + 2)
        ax2.imshow(bicubic_crop)
        ax2.set_title(f'Bicubic {scale}x\nPSNR: {bicubic_psnr:.2f} dB', 
                     fontsize=11, fontweight='bold', color='blue')
        ax2.axis('off')
        
        # LIIF Output
        ax3 = plt.subplot(3, 6, idx*6 + 3)
        ax3.imshow(output_crop)
        ax3.set_title(f'LIIF {scale}x\nPSNR: {liif_psnr:.2f} dB', 
                     fontsize=11, fontweight='bold', color='red')
        ax3.axis('off')
        
        # Ground Truth
        ax4 = plt.subplot(3, 6, idx*6 + 4)
        ax4.imshow(gt_crop)
        ax4.set_title(f'Ground Truth\n{gt_resized.size[0]}×{gt_resized.size[1]}', 
                     fontsize=11, fontweight='bold', color='green')
        ax4.axis('off')
        
        # Bicubic Difference Map
        ax5 = plt.subplot(3, 6, idx*6 + 5)
        bicubic_diff, bicubic_diff_mag = normalize_and_compute_diff(bicubic_crop, gt_crop)
        ax5.imshow(bicubic_diff, cmap='hot', vmin=0, vmax=0.2)
        ax5.set_title(f'Bicubic Diff\nMag: {bicubic_diff_mag:.4f}', 
                     fontsize=10, fontweight='bold', color='blue')
        ax5.axis('off')
        
        # LIIF Difference Map
        ax6 = plt.subplot(3, 6, idx*6 + 6)
        liif_diff, liif_diff_mag = normalize_and_compute_diff(output_crop, gt_crop)
        ax6.imshow(liif_diff, cmap='hot', vmin=0, vmax=0.2)
        ax6.set_title(f'LIIF Diff\nMag: {liif_diff_mag:.4f}', 
                     fontsize=10, fontweight='bold', color='red')
        ax6.axis('off')
        
        print(f"  PSNR values:")
        print(f"    Bicubic: {bicubic_psnr:.2f} dB")
        print(f"    LIIF: {liif_psnr:.2f} dB")
    
    # Overall title
    plt.suptitle(f'LIIF vs Bicubic Analysis: Multiple Scale Comparison (2x, 4x, 8x)\nImage: {os.path.basename(image_path)}', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot
    output_filename = f'outputs/multiscale_analysis_{os.path.splitext(os.path.basename(image_path))[0]}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved {output_filename}")
    
    # Show summary statistics
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   • Image: {os.path.basename(image_path)}")
    print(f"   • Scale factors: {scale_factors}")
    print(f"   • Base input size: {input_size}")
    print(f"   • Output sizes: {[f'{input_size[0]*s}x{input_size[1]*s}' for s in scale_factors]}")
    
    plt.show()

if __name__ == '__main__':
    # Example usage with a single image
    image_path = 'load/div2k/DIV2K_valid_HR/0855.png'
    create_comprehensive_plot(image_path) 