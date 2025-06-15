import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

import models
import datasets
from utils import make_coord
from datasets.math_function import MathFunctionDataset

def batched_predict(model, inp, coord, cell, bsize):
    """Predict values in batches to avoid memory issues"""
    with torch.no_grad():
        model.eval()
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model(inp, coord[:, ql:qr, :], cell[:, ql:qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

def bicubic_upscale(inp, size):
    """Bicubic upsampling of the input"""
    inp_tensor = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float()
    up = F.interpolate(inp_tensor, size=(size, size), mode='bicubic', align_corners=False)
    return up.squeeze().numpy()

def compute_metrics(pred, gt):
    """Compute various error metrics"""
    mse = np.mean((pred - gt) ** 2)
    psnr = -10 * np.log10(mse)
    mae = np.mean(np.abs(pred - gt))  # This is L1 loss
    max_error = np.max(np.abs(pred - gt))
    l1_per_freq = np.mean(np.abs(pred - gt), axis=(0, 1))  # Average L1 loss
    return {
        'PSNR': psnr,
        'L1': mae,  # Renamed MAE to L1 for clarity
        'Max Error': max_error
    }

def eval_function(model, k1, k2, resolution=256, scale_factor=4, device='cuda'):
    """Evaluate a specific function with given k1, k2 values"""
    # Generate low-res input
    dataset = MathFunctionDataset(resolution=resolution//scale_factor)
    inp = dataset._generate_function(k1, k2)
    inp = inp.squeeze().numpy()  # Remove channel dimension for bicubic
    inp_tensor = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dims for LIIF

    # Generate coordinates for high-res output
    h = resolution
    w = resolution
    coord = make_coord((h, w)).to(device)
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    
    # Add batch dimension to coord and cell
    coord = coord.unsqueeze(0)
    cell = cell.unsqueeze(0)

    # Generate LIIF prediction
    pred = batched_predict(model, inp_tensor, coord, cell, bsize=30000)
    pred = pred.view(h, w).cpu().numpy()
    
    # Generate bicubic prediction
    pred_bicubic = bicubic_upscale(inp, resolution)
    
    # Generate ground truth for comparison
    dataset_hr = MathFunctionDataset(resolution=resolution)
    gt = dataset_hr._generate_function(k1, k2)
    gt = gt.squeeze().numpy()
    
    return inp, pred, pred_bicubic, gt

def plot_results(inp, pred, pred_bicubic, gt, k1, k2, save_path):
    """Plot input, predictions, ground truth, and errors"""
    plt.figure(figsize=(25, 15))  # Made figure taller for L1 loss plot
    
    # First row: Images
    plt.subplot(341)
    plt.imshow(inp, cmap='viridis')
    plt.title(f'Input (Low-res)\nk1={k1}, k2={k2}')
    plt.colorbar()
    
    plt.subplot(342)
    plt.imshow(pred, cmap='viridis')
    plt.title('LIIF Prediction (High-res)')
    plt.colorbar()
    
    plt.subplot(343)
    plt.imshow(pred_bicubic, cmap='viridis')
    plt.title('Bicubic Upsampling')
    plt.colorbar()
    
    plt.subplot(344)
    plt.imshow(gt, cmap='viridis')
    plt.title('Ground Truth (High-res)')
    plt.colorbar()
    
    # Second row: Error maps and metrics
    plt.subplot(345)
    plt.text(0.1, 0.5, 'Error Metrics:', fontsize=12)
    plt.text(0.1, 0.3, f'Resolution Scale: {gt.shape[0]}x{gt.shape[1]}', fontsize=10)
    plt.axis('off')
    
    plt.subplot(346)
    error_liif = np.abs(pred - gt)
    plt.imshow(error_liif, cmap='hot')
    metrics_liif = compute_metrics(pred, gt)
    plt.title(f'LIIF Error\nPSNR: {metrics_liif["PSNR"]:.2f}dB\nL1 Loss: {metrics_liif["L1"]:.4f}\nMax Error: {metrics_liif["Max Error"]:.4f}')
    plt.colorbar()
    
    plt.subplot(347)
    error_bicubic = np.abs(pred_bicubic - gt)
    plt.imshow(error_bicubic, cmap='hot')
    metrics_bicubic = compute_metrics(pred_bicubic, gt)
    plt.title(f'Bicubic Error\nPSNR: {metrics_bicubic["PSNR"]:.2f}dB\nL1 Loss: {metrics_bicubic["L1"]:.4f}\nMax Error: {metrics_bicubic["Max Error"]:.4f}')
    plt.colorbar()
    
    plt.subplot(348)
    error_diff = error_liif - error_bicubic
    plt.imshow(error_diff, cmap='bwr')
    plt.title('Error Difference\n(LIIF - Bicubic)\nRed: LIIF worse\nBlue: LIIF better')
    plt.colorbar()
    
    # Third row: L1 Loss Analysis
    plt.subplot(313)
    x = np.arange(gt.shape[0])
    l1_liif_rows = np.mean(error_liif, axis=1)
    l1_bicubic_rows = np.mean(error_bicubic, axis=1)
    
    plt.plot(x, l1_liif_rows, label='LIIF L1', color='blue', alpha=0.7)
    plt.plot(x, l1_bicubic_rows, label='Bicubic L1', color='red', alpha=0.7)
    plt.fill_between(x, l1_liif_rows, l1_bicubic_rows, 
                     where=(l1_liif_rows < l1_bicubic_rows),
                     color='blue', alpha=0.3, label='LIIF Better')
    plt.fill_between(x, l1_liif_rows, l1_bicubic_rows,
                     where=(l1_liif_rows >= l1_bicubic_rows),
                     color='red', alpha=0.3, label='Bicubic Better')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f'L1 Loss Comparison Across Vertical Axis\nAvg LIIF L1: {np.mean(l1_liif_rows):.4f}, Avg Bicubic L1: {np.mean(l1_bicubic_rows):.4f}')
    plt.xlabel('Vertical Position (pixels)')
    plt.ylabel('L1 Loss')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics_liif, metrics_bicubic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train-div2k/train_edsr-baseline-liif_math.yaml')
    parser.add_argument('--model', default='save/math_function_training/epoch-best.pth')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Load model
    model = models.make(config['model']).cuda()
    model.load_state_dict(torch.load(args.model)['model']['sd'])
    
    # Create output directory
    save_dir = 'results/math_function_inference'
    os.makedirs(save_dir, exist_ok=True)
    
    # Test different k1, k2 combinations
    test_combinations = [
        (2, 2), (2, 8),  # Low-low and low-high frequency
        (5, 5),          # Mid-mid frequency
        (8, 2), (8, 8)   # High-low and high-high frequency
    ]
    
    print(f"Running inference at {args.resolution}x{args.resolution} resolution...")
    print(f"Scale factor: {args.scale}x")
    
    # Store metrics for all combinations
    all_metrics = []
    
    for k1, k2 in tqdm(test_combinations):
        print(f"\nTesting k1={k1}, k2={k2}")
        inp, pred, pred_bicubic, gt = eval_function(
            model, k1, k2,
            resolution=args.resolution,
            scale_factor=args.scale
        )
        
        save_path = os.path.join(save_dir, f'result_k1_{k1}_k2_{k2}_scale_{args.scale}x.png')
        metrics_liif, metrics_bicubic = plot_results(inp, pred, pred_bicubic, gt, k1, k2, save_path)
        print(f"Results saved to {save_path}")
        
        # Store metrics
        all_metrics.append({
            'k1': k1,
            'k2': k2,
            'scale': args.scale,
            'LIIF': metrics_liif,
            'Bicubic': metrics_bicubic
        })
        
        # Print comparison with focus on L1 loss
        print("\nL1 Loss Comparison:")
        print(f"LIIF L1:     {metrics_liif['L1']:.6f}")
        print(f"Bicubic L1:  {metrics_bicubic['L1']:.6f}")
        print(f"Difference:  {metrics_liif['L1'] - metrics_bicubic['L1']:.6f}")
        print(f"Improvement: {((metrics_bicubic['L1'] - metrics_liif['L1'])/metrics_bicubic['L1']*100):.2f}%")
    
    # Save metrics to file with focus on L1 loss
    metrics_file = os.path.join(save_dir, f'l1_metrics_scale_{args.scale}x.txt')
    with open(metrics_file, 'w') as f:
        f.write("L1 Loss Analysis Summary\n")
        f.write("=====================\n\n")
        for m in all_metrics:
            f.write(f"k1={m['k1']}, k2={m['k2']}, scale={m['scale']}x\n")
            f.write(f"LIIF L1:     {m['LIIF']['L1']:.6f}\n")
            f.write(f"Bicubic L1:  {m['Bicubic']['L1']:.6f}\n")
            improvement = ((m['Bicubic']['L1'] - m['LIIF']['L1'])/m['Bicubic']['L1']*100)
            f.write(f"Improvement: {improvement:.2f}%\n\n")
    
    print(f"\nInference completed! Results and L1 metrics saved in: {save_dir}")

if __name__ == '__main__':
    main() 