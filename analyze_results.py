#!/usr/bin/env python3

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import re
import os

def analyze_images():
    """Analyze input and output image dimensions"""
    print("="*60)
    print("LIIF IMAGE ANALYSIS")
    print("="*60)
    
    # Check ground truth dimensions
    gt_0801 = Image.open('load/div2k/DIV2K_valid_HR/0801.png')
    gt_0802 = Image.open('load/div2k/DIV2K_valid_HR/0802.png')
    
    print('\n📸 GROUND TRUTH (INPUT) IMAGES:')
    print(f'  • 0801.png: {gt_0801.size[0]}×{gt_0801.size[1]} pixels ({gt_0801.mode})')
    print(f'  • 0802.png: {gt_0802.size[0]}×{gt_0802.size[1]} pixels ({gt_0802.mode})')
    
    # Check output dimensions
    out_1024 = Image.open('outputs/demo_output_1024x1024.png')
    out_2048 = Image.open('outputs/demo_output_2048x2048.png') 
    out_1536 = Image.open('outputs/demo_output_0802_1536x1536.png')
    
    print('\n🎨 GENERATED SUPER-RESOLUTION OUTPUTS:')
    print(f'  • demo_output_1024x1024.png: {out_1024.size[0]}×{out_1024.size[1]} pixels')
    print(f'  • demo_output_2048x2048.png: {out_2048.size[0]}×{out_2048.size[1]} pixels')
    print(f'  • demo_output_0802_1536x1536.png: {out_1536.size[0]}×{out_1536.size[1]} pixels')
    
    # Calculate scaling factors
    print('\n📊 SCALING ANALYSIS:')
    original_size_0801 = gt_0801.size[0] * gt_0801.size[1]
    original_size_0802 = gt_0802.size[0] * gt_0802.size[1]
    
    out_1024_size = 1024 * 1024
    out_2048_size = 2048 * 2048
    out_1536_size = 1536 * 1536
    
    scale_1024 = out_1024_size / original_size_0801
    scale_2048 = out_2048_size / original_size_0801  
    scale_1536 = out_1536_size / original_size_0802
    
    print(f'  • 0801 → 1024×1024: {scale_1024:.2f}× scaling')
    print(f'  • 0801 → 2048×2048: {scale_2048:.2f}× scaling') 
    print(f'  • 0802 → 1536×1536: {scale_1536:.2f}× scaling')
    
    return gt_0801, gt_0802, out_1024, out_2048, out_1536

def extract_training_metrics():
    """Extract training metrics from log file"""
    print('\n📈 TRAINING METRICS EXTRACTION:')
    
    epochs = []
    train_losses = []
    val_psnrs = []
    
    with open('save/test_run/log.txt', 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Extract training loss: "epoch X/50, train: loss=Y.ZZZZ"
        train_match = re.search(r'epoch (\d+)/50, train: loss=([\d.]+)', line)
        if train_match:
            epoch = int(train_match.group(1))
            loss = float(train_match.group(2))
            epochs.append(epoch)
            train_losses.append(loss)
        
        # Extract validation PSNR: "val: psnr=XX.XXXX"
        val_match = re.search(r'val: psnr=([\d.]+)', line)
        if val_match:
            psnr = float(val_match.group(1))
            val_psnrs.append(psnr)
    
    print(f'  • Extracted {len(train_losses)} training loss values')
    print(f'  • Extracted {len(val_psnrs)} validation PSNR values')
    print(f'  • Final training loss: {train_losses[-1]:.4f}')
    print(f'  • Best validation PSNR: {max(val_psnrs):.4f} dB')
    
    return epochs, train_losses, val_psnrs

def create_training_plots(epochs, train_losses, val_psnrs):
    """Create training metric plots"""
    print('\n📊 CREATING TRAINING PLOTS...')
    
    # Validation epochs (every 5 epochs)
    val_epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    val_epochs = val_epochs[:len(val_psnrs)]  # Ensure we don't exceed available data
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training Loss Plot
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('LIIF Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(train_losses) * 1.1)
    
    # Add annotations for key points
    min_loss_epoch = epochs[np.argmin(train_losses)]
    min_loss_value = min(train_losses)
    ax1.annotate(f'Min Loss: {min_loss_value:.4f}\nEpoch: {min_loss_epoch}', 
                xy=(min_loss_epoch, min_loss_value), 
                xytext=(min_loss_epoch + 5, min_loss_value + 0.01),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    # Validation PSNR Plot
    ax2.plot(val_epochs, val_psnrs, 'g-o', linewidth=2, markersize=6, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation PSNR (dB)')
    ax2.set_title('LIIF Validation PSNR Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations for best PSNR
    best_psnr_idx = np.argmax(val_psnrs)
    best_psnr_epoch = val_epochs[best_psnr_idx]
    best_psnr_value = max(val_psnrs)
    ax2.annotate(f'Best PSNR: {best_psnr_value:.4f} dB\nEpoch: {best_psnr_epoch}', 
                xy=(best_psnr_epoch, best_psnr_value), 
                xytext=(best_psnr_epoch - 10, best_psnr_value - 0.5),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig('outputs/training_metrics.png', dpi=300, bbox_inches='tight')
    print('  ✅ Saved training_metrics.png')
    
def create_image_comparison():
    """Create image comparison visualization"""
    print('\n🖼️  CREATING IMAGE COMPARISON...')
    
    # Load images
    gt_0801 = Image.open('load/div2k/DIV2K_valid_HR/0801.png')
    gt_0802 = Image.open('load/div2k/DIV2K_valid_HR/0802.png')
    out_1024 = Image.open('outputs/demo_output_1024x1024.png')
    out_2048 = Image.open('outputs/demo_output_2048x2048.png')
    out_1536 = Image.open('outputs/demo_output_0802_1536x1536.png')
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Image 0801 comparisons
    # Ground truth 0801 (crop for visualization)
    gt_0801_crop = gt_0801.crop((500, 300, 1000, 800))  # 500x500 crop
    axes[0, 0].imshow(gt_0801_crop)
    axes[0, 0].set_title(f'Ground Truth 0801\n(Crop: 500×500)\nOriginal: {gt_0801.size[0]}×{gt_0801.size[1]}', fontsize=12)
    axes[0, 0].axis('off')
    
    # 1024x1024 output (crop for comparison)
    out_1024_crop = out_1024.crop((200, 150, 700, 650))  # 500x500 crop  
    axes[0, 1].imshow(out_1024_crop)
    axes[0, 1].set_title(f'LIIF Output 1024×1024\n(Crop: 500×500)\nScale: ~0.37× area', fontsize=12)
    axes[0, 1].axis('off')
    
    # 2048x2048 output (crop for comparison)
    out_2048_crop = out_2048.crop((400, 300, 900, 800))  # 500x500 crop
    axes[0, 2].imshow(out_2048_crop)
    axes[0, 2].set_title(f'LIIF Output 2048×2048\n(Crop: 500×500)\nScale: ~1.48× area', fontsize=12)
    axes[0, 2].axis('off')
    
    # Row 2: Image 0802 comparison + metrics
    # Ground truth 0802 (crop for visualization)
    gt_0802_crop = gt_0802.crop((400, 200, 900, 700))  # 500x500 crop
    axes[1, 0].imshow(gt_0802_crop)
    axes[1, 0].set_title(f'Ground Truth 0802\n(Crop: 500×500)\nOriginal: {gt_0802.size[0]}×{gt_0802.size[1]}', fontsize=12)
    axes[1, 0].axis('off')
    
    # 1536x1536 output (crop for comparison)
    out_1536_crop = out_1536.crop((300, 150, 800, 650))  # 500x500 crop
    axes[1, 1].imshow(out_1536_crop)
    axes[1, 1].set_title(f'LIIF Output 1536×1536\n(Crop: 500×500)\nScale: ~0.85× area', fontsize=12)
    axes[1, 1].axis('off')
    
    # Performance summary
    axes[1, 2].text(0.1, 0.9, '🏆 LIIF Performance Summary', fontsize=16, weight='bold', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.8, '📊 Training Results:', fontsize=14, weight='bold', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.75, '• Epochs: 50', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.7, '• Final Loss: ~0.038', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.65, '• Best PSNR: 32.84 dB', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.6, '• Model Size: 1.6M params', fontsize=12, transform=axes[1, 2].transAxes)
    
    axes[1, 2].text(0.1, 0.5, '🎯 Capabilities:', fontsize=14, weight='bold', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.45, '• Arbitrary resolution', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.4, '• Single model all scales', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.35, '• Continuous representation', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.3, '• High quality outputs', fontsize=12, transform=axes[1, 2].transAxes)
    
    axes[1, 2].text(0.1, 0.2, '📏 Output Resolutions:', fontsize=14, weight='bold', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.15, '• 1024×1024 (1M pixels)', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.1, '• 2048×2048 (4M pixels)', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.05, '• 1536×1536 (2.4M pixels)', fontsize=12, transform=axes[1, 2].transAxes)
    
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.suptitle('LIIF: Ground Truth vs Super-Resolution Outputs', fontsize=20, weight='bold')
    plt.tight_layout()
    plt.savefig('outputs/image_comparison.png', dpi=300, bbox_inches='tight')
    print('  ✅ Saved image_comparison.png')

def main():
    """Main analysis function"""
    # Analyze images
    analyze_images()
    
    # Extract and plot training metrics
    epochs, train_losses, val_psnrs = extract_training_metrics()
    create_training_plots(epochs, train_losses, val_psnrs)
    
    # Create image comparisons
    create_image_comparison()
    
    print('\n' + '='*60)
    print('✅ ANALYSIS COMPLETE!')
    print('Generated files:')
    print('  • outputs/training_metrics.png - Training loss and validation PSNR plots')
    print('  • outputs/image_comparison.png - Visual comparison of inputs vs outputs')
    print('='*60)

if __name__ == '__main__':
    main() 