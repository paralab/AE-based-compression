#!/usr/bin/env python3
"""
Analyze Mean Magnitudes of U_CHI Dataset Samples
Calculate mean magnitude for each sample and create histogram
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.u_chi_dataset import create_u_chi_datasets

def analyze_sample_mean_magnitudes():
    """Analyze mean magnitude for each sample in the dataset"""
    print("="*70)
    print("ANALYZING MEAN MAGNITUDES OF U_CHI DATASET SAMPLES")
    print("="*70)
    
    data_folder = "/u/tawal/0620-NN-based-compression-thera/tt_q01/"
    output_dir = "mean_magnitude_analysis"
    
    print(f"Data folder: {data_folder}")
    print("Loading dataset (without normalization to see raw values)...")
    
    try:
        # Create datasets without normalization to see raw values
        train_dataset, val_dataset = create_u_chi_datasets(
            data_folder=data_folder,
            train_ratio=0.8,
            normalize=False,  # Use raw data
            normalize_method='none'
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # Combine both datasets for comprehensive analysis
        all_datasets = {
            'train': train_dataset,
            'val': val_dataset
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        for dataset_name, dataset in all_datasets.items():
            print(f"\n" + "="*50)
            print(f"ANALYZING {dataset_name.upper()} DATASET ({len(dataset)} samples)")
            print(f"="*50)
            
            # Collect statistics for each sample
            sample_means = []
            sample_mean_magnitudes = []
            sample_stds = []
            sample_mins = []
            sample_maxs = []
            sample_dynamic_ranges = []
            
            print("Processing samples...")
            for i in tqdm(range(len(dataset)), desc=f"Processing {dataset_name}"):
                sample, metadata = dataset[i]
                
                # Remove channel dimension: (1, 7, 7, 7) -> (7, 7, 7)
                data = sample.numpy().squeeze()
                
                # Calculate statistics
                sample_mean = np.mean(data)
                sample_mean_magnitude = np.mean(np.abs(data))
                sample_std = np.std(data)
                sample_min = np.min(data)
                sample_max = np.max(data)
                
                # Calculate dynamic range (avoiding division by zero)
                if sample_min > 0:
                    dynamic_range = sample_max / sample_min
                elif sample_min == 0 and sample_max > 0:
                    dynamic_range = float('inf')
                else:
                    dynamic_range = 1.0  # All values are the same
                
                # Store statistics
                sample_means.append(sample_mean)
                sample_mean_magnitudes.append(sample_mean_magnitude)
                sample_stds.append(sample_std)
                sample_mins.append(sample_min)
                sample_maxs.append(sample_max)
                sample_dynamic_ranges.append(dynamic_range)
            
            # Convert to numpy arrays for easier analysis
            sample_means = np.array(sample_means)
            sample_mean_magnitudes = np.array(sample_mean_magnitudes)
            sample_stds = np.array(sample_stds)
            sample_mins = np.array(sample_mins)
            sample_maxs = np.array(sample_maxs)
            sample_dynamic_ranges = np.array(sample_dynamic_ranges)
            
            # Print summary statistics
            print(f"\n{dataset_name.upper()} DATASET SUMMARY:")
            print(f"  Total samples: {len(sample_means)}")
            print(f"\n  Sample Means:")
            print(f"    Range: [{sample_means.min():.6e}, {sample_means.max():.6e}]")
            print(f"    Mean of means: {sample_means.mean():.6e}")
            print(f"    Std of means: {sample_means.std():.6e}")
            
            print(f"\n  Sample Mean Magnitudes:")
            print(f"    Range: [{sample_mean_magnitudes.min():.6e}, {sample_mean_magnitudes.max():.6e}]")
            print(f"    Mean of mean magnitudes: {sample_mean_magnitudes.mean():.6e}")
            print(f"    Std of mean magnitudes: {sample_mean_magnitudes.std():.6e}")
            
            print(f"\n  Overall Data Range:")
            print(f"    Global min: {sample_mins.min():.6e}")
            print(f"    Global max: {sample_maxs.max():.6e}")
            print(f"    Global dynamic range: {sample_maxs.max()/sample_mins.min():.2e}")
            
            # Create comprehensive plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'U_CHI {dataset_name.upper()} Dataset - Sample Statistics Analysis\n'
                        f'({len(dataset)} samples)', fontsize=16)
            
            # 1. Histogram of sample means
            axes[0, 0].hist(sample_means, bins=50, alpha=0.7, color='blue', density=True)
            axes[0, 0].set_xlabel('Sample Mean')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('Distribution of Sample Means')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Histogram of sample mean magnitudes
            axes[0, 1].hist(sample_mean_magnitudes, bins=50, alpha=0.7, color='red', density=True)
            axes[0, 1].set_xlabel('Sample Mean Magnitude')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Distribution of Sample Mean Magnitudes')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Scatter plot: mean vs mean magnitude
            axes[0, 2].scatter(sample_means, sample_mean_magnitudes, alpha=0.6, s=10)
            axes[0, 2].set_xlabel('Sample Mean')
            axes[0, 2].set_ylabel('Sample Mean Magnitude')
            axes[0, 2].set_title('Mean vs Mean Magnitude')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Histogram of sample standard deviations
            axes[1, 0].hist(sample_stds, bins=50, alpha=0.7, color='green', density=True)
            axes[1, 0].set_xlabel('Sample Standard Deviation')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Distribution of Sample Std Deviations')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Histogram of dynamic ranges (exclude infinite values)
            finite_dynamic_ranges = sample_dynamic_ranges[np.isfinite(sample_dynamic_ranges)]
            if len(finite_dynamic_ranges) > 0:
                axes[1, 1].hist(finite_dynamic_ranges, bins=50, alpha=0.7, color='orange', density=True)
                axes[1, 1].set_xlabel('Dynamic Range (Max/Min)')
                axes[1, 1].set_ylabel('Density')
                axes[1, 1].set_title('Distribution of Sample Dynamic Ranges')
                axes[1, 1].set_yscale('log')
                axes[1, 1].set_xscale('log')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No finite dynamic ranges', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Dynamic Ranges (No finite values)')
            
            # 6. Sample index vs mean magnitude
            indices = np.arange(len(sample_mean_magnitudes))
            axes[1, 2].plot(indices, sample_mean_magnitudes, 'o-', alpha=0.7, markersize=2)
            axes[1, 2].set_xlabel('Sample Index')
            axes[1, 2].set_ylabel('Mean Magnitude')
            axes[1, 2].set_title('Mean Magnitude vs Sample Index')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = os.path.join(output_dir, f'sample_statistics_{dataset_name}.png')
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved plot: {plot_filename}")
            
            # Save statistics to text file
            stats_filename = os.path.join(output_dir, f'sample_statistics_{dataset_name}.txt')
            with open(stats_filename, 'w') as f:
                f.write(f"U_CHI {dataset_name.upper()} Dataset Sample Statistics\n")
                f.write("="*50 + "\n\n")
                f.write(f"Total samples: {len(sample_means)}\n\n")
                
                f.write("Sample Means:\n")
                f.write(f"  Min: {sample_means.min():.6e}\n")
                f.write(f"  Max: {sample_means.max():.6e}\n")
                f.write(f"  Mean: {sample_means.mean():.6e}\n")
                f.write(f"  Std: {sample_means.std():.6e}\n\n")
                
                f.write("Sample Mean Magnitudes:\n")
                f.write(f"  Min: {sample_mean_magnitudes.min():.6e}\n")
                f.write(f"  Max: {sample_mean_magnitudes.max():.6e}\n")
                f.write(f"  Mean: {sample_mean_magnitudes.mean():.6e}\n")
                f.write(f"  Std: {sample_mean_magnitudes.std():.6e}\n\n")
                
                f.write("Overall Data Range:\n")
                f.write(f"  Global min: {sample_mins.min():.6e}\n")
                f.write(f"  Global max: {sample_maxs.max():.6e}\n")
                f.write(f"  Global dynamic range: {sample_maxs.max()/sample_mins.min():.2e}\n\n")
                
                # Percentiles
                percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                f.write("Sample Mean Magnitude Percentiles:\n")
                for p in percentiles:
                    val = np.percentile(sample_mean_magnitudes, p)
                    f.write(f"  {p:2d}%: {val:.6e}\n")
            
            print(f"  Saved statistics: {stats_filename}")
        
        # Create combined comparison plot
        print(f"\n" + "="*50)
        print("CREATING COMBINED COMPARISON")
        print(f"="*50)
        
        # Get data for both datasets
        train_sample, _ = train_dataset[0]
        val_sample, _ = val_dataset[0]
        
        train_means = []
        val_means = []
        train_mean_mags = []
        val_mean_mags = []
        
        # Collect data for comparison
        for i in range(min(1000, len(train_dataset))):  # Limit for performance
            sample, _ = train_dataset[i]
            data = sample.numpy().squeeze()
            train_means.append(np.mean(data))
            train_mean_mags.append(np.mean(np.abs(data)))
        
        for i in range(min(1000, len(val_dataset))):  # Limit for performance  
            sample, _ = val_dataset[i]
            data = sample.numpy().squeeze()
            val_means.append(np.mean(data))
            val_mean_mags.append(np.mean(np.abs(data)))
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Train vs Validation Dataset Comparison\n(Sample Mean Magnitudes)', fontsize=14)
        
        # Histogram comparison
        axes[0].hist(train_mean_mags, bins=50, alpha=0.7, color='blue', density=True, label='Train')
        axes[0].hist(val_mean_mags, bins=50, alpha=0.7, color='red', density=True, label='Validation')
        axes[0].set_xlabel('Sample Mean Magnitude')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Distribution Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot comparison
        axes[1].boxplot([train_mean_mags, val_mean_mags], labels=['Train', 'Validation'])
        axes[1].set_ylabel('Sample Mean Magnitude')
        axes[1].set_title('Box Plot Comparison')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_filename = os.path.join(output_dir, 'train_vs_val_comparison.png')
        plt.savefig(comparison_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot: {comparison_filename}")
        
        print(f"\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print(f"="*70)
        print(f"Results saved in: {output_dir}/")
        print(f"  - sample_statistics_train.png")
        print(f"  - sample_statistics_val.png")
        print(f"  - train_vs_val_comparison.png")
        print(f"  - sample_statistics_train.txt")
        print(f"  - sample_statistics_val.txt")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_sample_mean_magnitudes() 