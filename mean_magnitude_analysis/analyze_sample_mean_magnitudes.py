#!/usr/bin/env python3
"""
Analyze Mean Magnitudes of U_CHI Dataset Samples
Calculate mean magnitude for each sample and create histogram
Now includes log-scaled analysis using positive-shift method
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from datasets.u_chi_dataset import create_u_chi_datasets

def log_scale_transform(data, epsilon=1e-8):
    """
    Apply positive-shift log-scale transformation to handle high dynamic range
    Using the exact method from visualize_uchi_log_scale.py
    
    Args:
        data: Input data array
        epsilon: Small value to prevent log(0)
    
    Returns:
        log_data: Log-transformed data
        transform_params: Parameters needed for inverse transformation
    """
    # Shift data to positive domain, then log
    data_min = data.min()
    data_shifted = data - data_min + epsilon
    log_data = np.log(data_shifted + epsilon)
    transform_params = {'data_min': data_min, 'epsilon': epsilon} 
    return log_data, transform_params

def analyze_sample_mean_magnitudes():
    """Analyze mean magnitude for each sample in the dataset - both raw and log-scaled"""
    print("="*70)
    print("ANALYZING MEAN MAGNITUDES OF U_CHI DATASET SAMPLES")
    print("Raw Data + Log-Scaled (Positive-Shift Method)")
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
            
            # Collect statistics for each sample - RAW DATA
            raw_sample_means = []
            raw_sample_mean_magnitudes = []
            raw_sample_stds = []
            raw_sample_mins = []
            raw_sample_maxs = []
            raw_sample_dynamic_ranges = []
            
            # Collect statistics for each sample - LOG-SCALED DATA
            log_sample_means = []
            log_sample_mean_magnitudes = []
            log_sample_stds = []
            log_sample_mins = []
            log_sample_maxs = []
            log_sample_dynamic_ranges = []
            
            print("Processing samples (raw + log-scaled analysis)...")
            for i in tqdm(range(len(dataset)), desc=f"Processing {dataset_name}"):
                sample, metadata = dataset[i]
                
                # Remove channel dimension: (1, 7, 7, 7) -> (7, 7, 7)
                raw_data = sample.numpy().squeeze()
                
                # === RAW DATA ANALYSIS ===
                raw_sample_mean = np.mean(raw_data)
                raw_sample_mean_magnitude = np.mean(np.abs(raw_data))
                raw_sample_std = np.std(raw_data)
                raw_sample_min = np.min(raw_data)
                raw_sample_max = np.max(raw_data)
                
                # Calculate dynamic range (avoiding division by zero)
                if raw_sample_min > 0:
                    raw_dynamic_range = raw_sample_max / raw_sample_min
                elif raw_sample_min == 0 and raw_sample_max > 0:
                    raw_dynamic_range = float('inf')
                else:
                    raw_dynamic_range = 1.0  # All values are the same
                
                # Store raw statistics
                raw_sample_means.append(raw_sample_mean)
                raw_sample_mean_magnitudes.append(raw_sample_mean_magnitude)
                raw_sample_stds.append(raw_sample_std)
                raw_sample_mins.append(raw_sample_min)
                raw_sample_maxs.append(raw_sample_max)
                raw_sample_dynamic_ranges.append(raw_dynamic_range)
                
                # === LOG-SCALED DATA ANALYSIS (using positive-shift method) ===
                try:
                    log_data, transform_params = log_scale_transform(raw_data, epsilon=1e-8)
                    
                    log_sample_mean = np.mean(log_data)
                    log_sample_mean_magnitude = np.mean(np.abs(log_data))
                    log_sample_std = np.std(log_data)
                    log_sample_min = np.min(log_data)
                    log_sample_max = np.max(log_data)
                    
                    # Calculate dynamic range for log data
                    if log_sample_min > 0:
                        log_dynamic_range = log_sample_max / log_sample_min
                    elif log_sample_min == 0 and log_sample_max > 0:
                        log_dynamic_range = float('inf')
                    elif log_sample_min < 0:
                        # For log data with negative values, use range span
                        log_dynamic_range = log_sample_max - log_sample_min
                    else:
                        log_dynamic_range = 1.0
                    
                    # Store log statistics
                    log_sample_means.append(log_sample_mean)
                    log_sample_mean_magnitudes.append(log_sample_mean_magnitude)
                    log_sample_stds.append(log_sample_std)
                    log_sample_mins.append(log_sample_min)
                    log_sample_maxs.append(log_sample_max)
                    log_sample_dynamic_ranges.append(log_dynamic_range)
                    
                except Exception as e:
                    print(f"Warning: Could not process log transform for sample {i}: {e}")
                    # Use NaN for failed transformations
                    log_sample_means.append(np.nan)
                    log_sample_mean_magnitudes.append(np.nan)
                    log_sample_stds.append(np.nan)
                    log_sample_mins.append(np.nan)
                    log_sample_maxs.append(np.nan)
                    log_sample_dynamic_ranges.append(np.nan)
            
            # Convert to numpy arrays for easier analysis
            raw_sample_means = np.array(raw_sample_means)
            raw_sample_mean_magnitudes = np.array(raw_sample_mean_magnitudes)
            raw_sample_stds = np.array(raw_sample_stds)
            raw_sample_mins = np.array(raw_sample_mins)
            raw_sample_maxs = np.array(raw_sample_maxs)
            raw_sample_dynamic_ranges = np.array(raw_sample_dynamic_ranges)
            
            log_sample_means = np.array(log_sample_means)
            log_sample_mean_magnitudes = np.array(log_sample_mean_magnitudes)
            log_sample_stds = np.array(log_sample_stds)
            log_sample_mins = np.array(log_sample_mins)
            log_sample_maxs = np.array(log_sample_maxs)
            log_sample_dynamic_ranges = np.array(log_sample_dynamic_ranges)
            
            # Remove NaN values for log statistics
            log_valid_mask = ~np.isnan(log_sample_means)
            log_valid_count = np.sum(log_valid_mask)
            
            # Print summary statistics
            print(f"\n{dataset_name.upper()} DATASET SUMMARY:")
            print(f"  Total samples: {len(raw_sample_means)}")
            print(f"  Successfully log-transformed: {log_valid_count}")
            
            print(f"\n  === RAW DATA STATISTICS ===")
            print(f"  Sample Means:")
            print(f"    Range: [{raw_sample_means.min():.6e}, {raw_sample_means.max():.6e}]")
            print(f"    Mean of means: {raw_sample_means.mean():.6e}")
            print(f"    Std of means: {raw_sample_means.std():.6e}")
            
            print(f"\n  Sample Mean Magnitudes:")
            print(f"    Range: [{raw_sample_mean_magnitudes.min():.6e}, {raw_sample_mean_magnitudes.max():.6e}]")
            print(f"    Mean of mean magnitudes: {raw_sample_mean_magnitudes.mean():.6e}")
            print(f"    Std of mean magnitudes: {raw_sample_mean_magnitudes.std():.6e}")
            
            print(f"\n  Overall Raw Data Range:")
            print(f"    Global min: {raw_sample_mins.min():.6e}")
            print(f"    Global max: {raw_sample_maxs.max():.6e}")
            print(f"    Global dynamic range: {raw_sample_maxs.max()/raw_sample_mins.min():.2e}")
            
            if log_valid_count > 0:
                log_means_valid = log_sample_means[log_valid_mask]
                log_mean_mags_valid = log_sample_mean_magnitudes[log_valid_mask]
                log_mins_valid = log_sample_mins[log_valid_mask]
                log_maxs_valid = log_sample_maxs[log_valid_mask]
                
                print(f"\n  === LOG-SCALED DATA STATISTICS (Positive-Shift Method) ===")
                print(f"  Sample Means:")
                print(f"    Range: [{log_means_valid.min():.6f}, {log_means_valid.max():.6f}]")
                print(f"    Mean of means: {log_means_valid.mean():.6f}")
                print(f"    Std of means: {log_means_valid.std():.6f}")
                
                print(f"\n  Sample Mean Magnitudes:")
                print(f"    Range: [{log_mean_mags_valid.min():.6f}, {log_mean_mags_valid.max():.6f}]")
                print(f"    Mean of mean magnitudes: {log_mean_mags_valid.mean():.6f}")
                print(f"    Std of mean magnitudes: {log_mean_mags_valid.std():.6f}")
                
                print(f"\n  Overall Log-Scaled Data Range:")
                print(f"    Global min: {log_mins_valid.min():.6f}")
                print(f"    Global max: {log_maxs_valid.max():.6f}")
                print(f"    Global range span: {log_maxs_valid.max() - log_mins_valid.min():.6f}")
            else:
                print(f"\n  ❌ No valid log transformations - all samples failed!")
            
            # Create comprehensive plots - RAW DATA
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'U_CHI {dataset_name.upper()} Dataset - RAW DATA Statistics Analysis\n'
                        f'({len(dataset)} samples)', fontsize=16)
            
            # 1. Histogram of sample means
            axes[0, 0].hist(raw_sample_means, bins=50, alpha=0.7, color='blue', density=True)
            axes[0, 0].set_xlabel('Sample Mean')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('Distribution of Sample Means (Raw)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Histogram of sample mean magnitudes
            axes[0, 1].hist(raw_sample_mean_magnitudes, bins=50, alpha=0.7, color='red', density=True)
            axes[0, 1].set_xlabel('Sample Mean Magnitude')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Distribution of Sample Mean Magnitudes (Raw)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Scatter plot: mean vs mean magnitude
            axes[0, 2].scatter(raw_sample_means, raw_sample_mean_magnitudes, alpha=0.6, s=10)
            axes[0, 2].set_xlabel('Sample Mean')
            axes[0, 2].set_ylabel('Sample Mean Magnitude')
            axes[0, 2].set_title('Mean vs Mean Magnitude (Raw)')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Histogram of sample standard deviations
            axes[1, 0].hist(raw_sample_stds, bins=50, alpha=0.7, color='green', density=True)
            axes[1, 0].set_xlabel('Sample Standard Deviation')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Distribution of Sample Std Deviations (Raw)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Histogram of dynamic ranges (exclude infinite values)
            finite_dynamic_ranges = raw_sample_dynamic_ranges[np.isfinite(raw_sample_dynamic_ranges)]
            if len(finite_dynamic_ranges) > 0:
                axes[1, 1].hist(finite_dynamic_ranges, bins=50, alpha=0.7, color='orange', density=True)
                axes[1, 1].set_xlabel('Dynamic Range (Max/Min)')
                axes[1, 1].set_ylabel('Density')
                axes[1, 1].set_title('Distribution of Sample Dynamic Ranges (Raw)')
                axes[1, 1].set_yscale('log')
                axes[1, 1].set_xscale('log')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No finite dynamic ranges', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Dynamic Ranges (No finite values)')
            
            # 6. Sample index vs mean magnitude
            indices = np.arange(len(raw_sample_mean_magnitudes))
            axes[1, 2].plot(indices, raw_sample_mean_magnitudes, 'o-', alpha=0.7, markersize=2)
            axes[1, 2].set_xlabel('Sample Index')
            axes[1, 2].set_ylabel('Mean Magnitude')
            axes[1, 2].set_title('Mean Magnitude vs Sample Index (Raw)')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save raw data plot
            plot_filename_raw = os.path.join(output_dir, f'sample_statistics_{dataset_name}_raw.png')
            plt.savefig(plot_filename_raw, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved raw data plot: {plot_filename_raw}")
            
            # Create LOG-SCALED DATA plots if we have valid data
            if log_valid_count > 0:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle(f'U_CHI {dataset_name.upper()} Dataset - LOG-SCALED DATA (Positive-Shift) Statistics Analysis\n'
                            f'({log_valid_count}/{len(dataset)} valid samples)', fontsize=16)
                
                log_means_valid = log_sample_means[log_valid_mask]
                log_mean_mags_valid = log_sample_mean_magnitudes[log_valid_mask]
                log_stds_valid = log_sample_stds[log_valid_mask]
                log_ranges_valid = log_sample_dynamic_ranges[log_valid_mask]
                
                # 1. Histogram of sample means (log)
                axes[0, 0].hist(log_means_valid, bins=50, alpha=0.7, color='purple', density=True)
                axes[0, 0].set_xlabel('Sample Mean (Log-scaled)')
                axes[0, 0].set_ylabel('Density')
                axes[0, 0].set_title('Distribution of Sample Means (Log)')
                axes[0, 0].grid(True, alpha=0.3)
                
                # 2. Histogram of sample mean magnitudes (log)
                axes[0, 1].hist(log_mean_mags_valid, bins=50, alpha=0.7, color='magenta', density=True)
                axes[0, 1].set_xlabel('Sample Mean Magnitude (Log-scaled)')
                axes[0, 1].set_ylabel('Density')
                axes[0, 1].set_title('Distribution of Sample Mean Magnitudes (Log)')
                axes[0, 1].grid(True, alpha=0.3)
                
                # 3. Scatter plot: mean vs mean magnitude (log)
                axes[0, 2].scatter(log_means_valid, log_mean_mags_valid, alpha=0.6, s=10, color='darkviolet')
                axes[0, 2].set_xlabel('Sample Mean (Log-scaled)')
                axes[0, 2].set_ylabel('Sample Mean Magnitude (Log-scaled)')
                axes[0, 2].set_title('Mean vs Mean Magnitude (Log)')
                axes[0, 2].grid(True, alpha=0.3)
                
                # 4. Histogram of sample standard deviations (log)
                axes[1, 0].hist(log_stds_valid, bins=50, alpha=0.7, color='teal', density=True)
                axes[1, 0].set_xlabel('Sample Standard Deviation (Log-scaled)')
                axes[1, 0].set_ylabel('Density')
                axes[1, 0].set_title('Distribution of Sample Std Deviations (Log)')
                axes[1, 0].grid(True, alpha=0.3)
                
                # 5. Histogram of log dynamic ranges
                finite_log_ranges = log_ranges_valid[np.isfinite(log_ranges_valid)]
                if len(finite_log_ranges) > 0:
                    axes[1, 1].hist(finite_log_ranges, bins=50, alpha=0.7, color='cyan', density=True)
                    axes[1, 1].set_xlabel('Dynamic Range (Log-scaled)')
                    axes[1, 1].set_ylabel('Density')
                    axes[1, 1].set_title('Distribution of Sample Dynamic Ranges (Log)')
                    axes[1, 1].grid(True, alpha=0.3)
                else:
                    axes[1, 1].text(0.5, 0.5, 'No finite dynamic ranges', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Dynamic Ranges (No finite values)')
                
                # 6. Sample index vs mean magnitude (log)
                valid_indices = np.arange(len(dataset))[log_valid_mask]
                axes[1, 2].plot(valid_indices, log_mean_mags_valid, 'o-', alpha=0.7, markersize=2, color='indigo')
                axes[1, 2].set_xlabel('Sample Index')
                axes[1, 2].set_ylabel('Mean Magnitude (Log-scaled)')
                axes[1, 2].set_title('Mean Magnitude vs Sample Index (Log)')
                axes[1, 2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save log-scaled data plot
                plot_filename_log = os.path.join(output_dir, f'sample_statistics_{dataset_name}_log.png')
                plt.savefig(plot_filename_log, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved log-scaled data plot: {plot_filename_log}")
                
                # Create comparison plot (Raw vs Log)
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'U_CHI {dataset_name.upper()} Dataset - Raw vs Log-Scaled (Positive-Shift) Comparison', fontsize=16)
                
                # Mean magnitude comparison
                axes[0, 0].hist(raw_sample_mean_magnitudes, bins=50, alpha=0.7, color='red', density=True, label='Raw')
                axes[0, 0].hist(log_mean_mags_valid, bins=50, alpha=0.7, color='purple', density=True, label='Log-scaled')
                axes[0, 0].set_xlabel('Sample Mean Magnitude')
                axes[0, 0].set_ylabel('Density')
                axes[0, 0].set_title('Mean Magnitude Distribution Comparison')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Standard deviation comparison
                axes[0, 1].hist(raw_sample_stds, bins=50, alpha=0.7, color='green', density=True, label='Raw')
                axes[0, 1].hist(log_stds_valid, bins=50, alpha=0.7, color='teal', density=True, label='Log-scaled')
                axes[0, 1].set_xlabel('Sample Standard Deviation')
                axes[0, 1].set_ylabel('Density')
                axes[0, 1].set_title('Standard Deviation Distribution Comparison')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Sample index comparison
                axes[1, 0].plot(indices, raw_sample_mean_magnitudes, 'o-', alpha=0.5, markersize=1, color='red', label='Raw')
                axes[1, 0].plot(valid_indices, log_mean_mags_valid, 'o-', alpha=0.5, markersize=1, color='purple', label='Log-scaled')
                axes[1, 0].set_xlabel('Sample Index')
                axes[1, 0].set_ylabel('Mean Magnitude')
                axes[1, 0].set_title('Mean Magnitude vs Sample Index Comparison')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Correlation analysis
                # Match valid samples for comparison
                raw_mean_mags_matched = raw_sample_mean_magnitudes[log_valid_mask]
                correlation_coef = np.corrcoef(raw_mean_mags_matched, log_mean_mags_valid)[0, 1]
                
                axes[1, 1].scatter(raw_mean_mags_matched, log_mean_mags_valid, alpha=0.6, s=10, color='darkblue')
                axes[1, 1].set_xlabel('Raw Mean Magnitude')
                axes[1, 1].set_ylabel('Log-scaled Mean Magnitude')
                axes[1, 1].set_title(f'Raw vs Log Correlation\n(r = {correlation_coef:.3f})')
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save comparison plot
                plot_filename_comp = os.path.join(output_dir, f'sample_statistics_{dataset_name}_comparison.png')
                plt.savefig(plot_filename_comp, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved comparison plot: {plot_filename_comp}")
            else:
                print(f"  ❌ Skipping log-scaled plots - no valid transformations")
            
            # Save statistics to text file
            stats_filename = os.path.join(output_dir, f'sample_statistics_{dataset_name}.txt')
            with open(stats_filename, 'w') as f:
                f.write(f"U_CHI {dataset_name.upper()} Dataset Sample Statistics\n")
                f.write("="*70 + "\n\n")
                f.write(f"Total samples: {len(raw_sample_means)}\n")
                f.write(f"Successfully log-transformed: {log_valid_count}\n\n")
                
                f.write("="*50 + "\n")
                f.write("RAW DATA STATISTICS\n")
                f.write("="*50 + "\n\n")
                
                f.write("Sample Means:\n")
                f.write(f"  Min: {raw_sample_means.min():.6e}\n")
                f.write(f"  Max: {raw_sample_means.max():.6e}\n")
                f.write(f"  Mean: {raw_sample_means.mean():.6e}\n")
                f.write(f"  Std: {raw_sample_means.std():.6e}\n\n")
                
                f.write("Sample Mean Magnitudes:\n")
                f.write(f"  Min: {raw_sample_mean_magnitudes.min():.6e}\n")
                f.write(f"  Max: {raw_sample_mean_magnitudes.max():.6e}\n")
                f.write(f"  Mean: {raw_sample_mean_magnitudes.mean():.6e}\n")
                f.write(f"  Std: {raw_sample_mean_magnitudes.std():.6e}\n\n")
                
                f.write("Overall Raw Data Range:\n")
                f.write(f"  Global min: {raw_sample_mins.min():.6e}\n")
                f.write(f"  Global max: {raw_sample_maxs.max():.6e}\n")
                f.write(f"  Global dynamic range: {raw_sample_maxs.max()/raw_sample_mins.min():.2e}\n\n")
                
                # Raw data percentiles
                percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                f.write("Raw Sample Mean Magnitude Percentiles:\n")
                for p in percentiles:
                    val = np.percentile(raw_sample_mean_magnitudes, p)
                    f.write(f"  {p:2d}%: {val:.6e}\n")
                
                # Log-scaled statistics if available
                if log_valid_count > 0:
                    log_means_valid = log_sample_means[log_valid_mask]
                    log_mean_mags_valid = log_sample_mean_magnitudes[log_valid_mask]
                    log_mins_valid = log_sample_mins[log_valid_mask]
                    log_maxs_valid = log_sample_maxs[log_valid_mask]
                    
                    f.write("\n" + "="*50 + "\n")
                    f.write("LOG-SCALED DATA STATISTICS (Positive-Shift Method)\n")
                    f.write("="*50 + "\n\n")
                    
                    f.write("Sample Means (Log-scaled):\n")
                    f.write(f"  Min: {log_means_valid.min():.6f}\n")
                    f.write(f"  Max: {log_means_valid.max():.6f}\n")
                    f.write(f"  Mean: {log_means_valid.mean():.6f}\n")
                    f.write(f"  Std: {log_means_valid.std():.6f}\n\n")
                    
                    f.write("Sample Mean Magnitudes (Log-scaled):\n")
                    f.write(f"  Min: {log_mean_mags_valid.min():.6f}\n")
                    f.write(f"  Max: {log_mean_mags_valid.max():.6f}\n")
                    f.write(f"  Mean: {log_mean_mags_valid.mean():.6f}\n")
                    f.write(f"  Std: {log_mean_mags_valid.std():.6f}\n\n")
                    
                    f.write("Overall Log-Scaled Data Range:\n")
                    f.write(f"  Global min: {log_mins_valid.min():.6f}\n")
                    f.write(f"  Global max: {log_maxs_valid.max():.6f}\n")
                    f.write(f"  Global range span: {log_maxs_valid.max() - log_mins_valid.min():.6f}\n\n")
                    
                    # Log-scaled percentiles
                    f.write("Log-Scaled Sample Mean Magnitude Percentiles:\n")
                    for p in percentiles:
                        val = np.percentile(log_mean_mags_valid, p)
                        f.write(f"  {p:2d}%: {val:.6f}\n")
                    
                    # Transformation quality metrics
                    raw_matched = raw_sample_mean_magnitudes[log_valid_mask]
                    correlation = np.corrcoef(raw_matched, log_mean_mags_valid)[0, 1]
                    
                    f.write("\n" + "="*50 + "\n")
                    f.write("TRANSFORMATION QUALITY METRICS\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Transformation success rate: {100*log_valid_count/len(raw_sample_means):.1f}%\n")
                    f.write(f"Raw vs Log correlation: {correlation:.6f}\n")
                    f.write(f"Dynamic range reduction: {(raw_sample_maxs.max()/raw_sample_mins.min()) / (log_maxs_valid.max() - log_mins_valid.min()):.2e}\n")
                    f.write(f"Positive-shift method: data_shifted = data - data_min + epsilon\n")
                    f.write(f"Log transformation: log_data = log(data_shifted + epsilon)\n")
                else:
                    f.write("\n" + "="*50 + "\n")
                    f.write("LOG-SCALED DATA STATISTICS\n")
                    f.write("="*50 + "\n\n")
                    f.write("❌ No valid log transformations - all samples failed!\n")
            
            print(f"  Saved comprehensive statistics: {stats_filename}")
        
        # Create combined comparison plot
        print(f"\n" + "="*50)
        print("CREATING COMBINED TRAIN/VAL COMPARISON")
        print(f"="*50)
        
        # Get data for both datasets
        train_sample, _ = train_dataset[0]
        val_sample, _ = val_dataset[0]
        
        train_means = []
        val_means = []
        train_mean_mags = []
        val_mean_mags = []
        
        # Collect raw and log data for comparison
        train_raw_mean_mags = []
        val_raw_mean_mags = []
        train_log_mean_mags = []
        val_log_mean_mags = []
        
        # Collect data for comparison
        for i in range(min(1000, len(train_dataset))):  # Limit for performance
            sample, _ = train_dataset[i]
            raw_data = sample.numpy().squeeze()
            train_means.append(np.mean(raw_data))
            train_raw_mean_mags.append(np.mean(np.abs(raw_data)))
            train_mean_mags.append(np.mean(np.abs(raw_data)))  # Keep for backward compatibility
            
            # Try log transformation using positive-shift method
            try:
                log_data, _ = log_scale_transform(raw_data, epsilon=1e-8)
                train_log_mean_mags.append(np.mean(np.abs(log_data)))
            except:
                train_log_mean_mags.append(np.nan)
        
        for i in range(min(1000, len(val_dataset))):  # Limit for performance  
            sample, _ = val_dataset[i]
            raw_data = sample.numpy().squeeze()
            val_means.append(np.mean(raw_data))
            val_raw_mean_mags.append(np.mean(np.abs(raw_data)))
            val_mean_mags.append(np.mean(np.abs(raw_data)))  # Keep for backward compatibility
            
            # Try log transformation using positive-shift method
            try:
                log_data, _ = log_scale_transform(raw_data, epsilon=1e-8)
                val_log_mean_mags.append(np.mean(np.abs(log_data)))
            except:
                val_log_mean_mags.append(np.nan)
        
        # Remove NaN values from log data
        train_log_mean_mags_clean = [x for x in train_log_mean_mags if not np.isnan(x)]
        val_log_mean_mags_clean = [x for x in val_log_mean_mags if not np.isnan(x)]
        
        # Create comprehensive comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Train vs Validation Dataset Comparison\n(Raw and Log-Scaled Analysis using Positive-Shift Method)', fontsize=16)
        
        # RAW DATA COMPARISONS
        # Histogram comparison - Raw
        axes[0, 0].hist(train_raw_mean_mags, bins=50, alpha=0.7, color='blue', density=True, label='Train')
        axes[0, 0].hist(val_raw_mean_mags, bins=50, alpha=0.7, color='red', density=True, label='Validation')
        axes[0, 0].set_xlabel('Sample Mean Magnitude (Raw)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Raw Data Distribution Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot comparison - Raw
        axes[0, 1].boxplot([train_raw_mean_mags, val_raw_mean_mags], labels=['Train', 'Validation'])
        axes[0, 1].set_ylabel('Sample Mean Magnitude (Raw)')
        axes[0, 1].set_title('Raw Data Box Plot Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sample index comparison - Raw
        train_indices = np.arange(len(train_raw_mean_mags))
        val_indices = np.arange(len(val_raw_mean_mags))
        axes[0, 2].plot(train_indices, train_raw_mean_mags, 'o-', alpha=0.5, markersize=1, color='blue', label='Train')
        axes[0, 2].plot(val_indices, val_raw_mean_mags, 'o-', alpha=0.5, markersize=1, color='red', label='Validation')
        axes[0, 2].set_xlabel('Sample Index')
        axes[0, 2].set_ylabel('Mean Magnitude (Raw)')
        axes[0, 2].set_title('Raw Data Index Comparison')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # LOG-SCALED DATA COMPARISONS (if available)
        if len(train_log_mean_mags_clean) > 0 and len(val_log_mean_mags_clean) > 0:
            # Histogram comparison - Log
            axes[1, 0].hist(train_log_mean_mags_clean, bins=50, alpha=0.7, color='purple', density=True, label='Train')
            axes[1, 0].hist(val_log_mean_mags_clean, bins=50, alpha=0.7, color='orange', density=True, label='Validation')
            axes[1, 0].set_xlabel('Sample Mean Magnitude (Log-scaled)')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Log-Scaled Distribution Comparison')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Box plot comparison - Log
            axes[1, 1].boxplot([train_log_mean_mags_clean, val_log_mean_mags_clean], labels=['Train', 'Validation'])
            axes[1, 1].set_ylabel('Sample Mean Magnitude (Log-scaled)')
            axes[1, 1].set_title('Log-Scaled Box Plot Comparison')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Sample index comparison - Log
            train_log_indices = [i for i, x in enumerate(train_log_mean_mags) if not np.isnan(x)]
            val_log_indices = [i for i, x in enumerate(val_log_mean_mags) if not np.isnan(x)]
            axes[1, 2].plot(train_log_indices, train_log_mean_mags_clean, 'o-', alpha=0.5, markersize=1, color='purple', label='Train')
            axes[1, 2].plot(val_log_indices, val_log_mean_mags_clean, 'o-', alpha=0.5, markersize=1, color='orange', label='Validation')
            axes[1, 2].set_xlabel('Sample Index')
            axes[1, 2].set_ylabel('Mean Magnitude (Log-scaled)')
            axes[1, 2].set_title('Log-Scaled Index Comparison')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            # No valid log data
            for j in range(3):
                axes[1, j].text(0.5, 0.5, 'No valid log-scaled data', 
                               ha='center', va='center', transform=axes[1, j].transAxes,
                               fontsize=14, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
                axes[1, j].set_title(f'Log-Scaled Analysis {j+1} (No Data)')
        
        plt.tight_layout()
        
        # Save comprehensive comparison plot
        comparison_filename = os.path.join(output_dir, 'train_vs_val_comprehensive_comparison.png')
        plt.savefig(comparison_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comprehensive comparison plot: {comparison_filename}")
        
        # Keep original simple comparison for backward compatibility
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Train vs Validation Dataset Comparison\n(Sample Mean Magnitudes - Raw Data)', fontsize=14)
        
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
        
        # Save simple comparison plot
        comparison_filename_simple = os.path.join(output_dir, 'train_vs_val_comparison.png')
        plt.savefig(comparison_filename_simple, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved simple comparison plot: {comparison_filename_simple}")
        
        print(f"\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print(f"="*70)
        print(f"Results saved in: {output_dir}/")
        print(f"\n📊 RAW DATA ANALYSIS:")
        print(f"  - sample_statistics_train_raw.png")
        print(f"  - sample_statistics_val_raw.png")
        print(f"\n📊 LOG-SCALED DATA ANALYSIS (Positive-Shift Method):")
        print(f"  - sample_statistics_train_log.png")
        print(f"  - sample_statistics_val_log.png")
        print(f"\n📊 COMPARISON ANALYSIS:")
        print(f"  - sample_statistics_train_comparison.png")
        print(f"  - sample_statistics_val_comparison.png")
        print(f"  - train_vs_val_comprehensive_comparison.png")
        print(f"  - train_vs_val_comparison.png (simple)")
        print(f"\n📈 STATISTICAL SUMMARIES:")
        print(f"  - sample_statistics_train.txt")
        print(f"  - sample_statistics_val.txt")
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"  ✅ Log-scale transformation using positive-shift method")
        print(f"      data_shifted = data - data_min + epsilon")
        print(f"      log_data = log(data_shifted + epsilon)")
        print(f"  ✅ Comprehensive raw vs log-scaled comparison")
        print(f"  ✅ Dynamic range reduction analysis")
        print(f"  ✅ Train/validation dataset comparison")
        print(f"  ✅ Statistical distribution analysis")
        print(f"\n💡 Use these results to:")
        print(f"  - Validate positive-shift preprocessing effectiveness")
        print(f"  - Understand data distribution changes")
        print(f"  - Optimize SWAE 3D training parameters")
        print(f"  - Compare train/val data characteristics")
        print(f"  - Quantify the breakthrough log-scale transformation benefits")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_sample_mean_magnitudes() 