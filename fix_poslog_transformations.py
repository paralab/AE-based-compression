#!/usr/bin/env python
"""
Fix poslog transformation issues for specific variables.
Implements alternative transformations better suited for symmetric and near-zero data.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def poslog_transform(data, epsilon=1e-8):
    """Standard poslog transformation."""
    data_min = data.min()
    if data_min <= 0:
        data_shifted = data - data_min + epsilon
    else:
        data_shifted = data.copy()
    return np.log(data_shifted)

def symmetric_log_transform(data, c=1.0):
    """
    Symmetric log transformation that preserves sign and handles zeros.
    Good for symmetric distributions around zero.
    """
    return np.sign(data) * np.log1p(np.abs(data) / c) * c

def asinh_transform(data, scale=1.0):
    """
    Inverse hyperbolic sine transformation.
    Natural for data spanning positive and negative values.
    """
    return np.arcsinh(data / scale)

def yeo_johnson_transform(data, lmbda=0):
    """
    Yeo-Johnson transformation - extension of Box-Cox for negative values.
    """
    from scipy import stats
    return stats.yeojohnson(data, lmbda=lmbda)[0]

def analyze_and_recommend():
    """Analyze specific variables and recommend transformations."""
    
    print("TRANSFORMATION RECOMMENDATIONS FOR SPECIFIC VARIABLES")
    print("=" * 80)
    
    recommendations = {
        'U_B2': {
            'issues': [
                "Highly skewed (-9.29) with outliers",
                "34.4% negative values",
                "Mean near zero (2e-6) with small std (0.003)",
                "Poslog shift of 0.104 changes relative scale dramatically"
            ],
            'best_transform': 'asinh',
            'reason': "Handles negative values naturally, reduces skewness without extreme shifts",
            'alternative': 'Robust scaling for outlier resistance'
        },
        'U_SYMAT2': {
            'issues': [
                "Perfectly symmetric (50% negative, 50% positive)",
                "Zero mean with no zeros in data",
                "Large range (-1.32 to 1.32)",
                "Poslog destroys natural symmetry"
            ],
            'best_transform': 'symmetric_log',
            'reason': "Preserves symmetry while compressing large values",
            'alternative': 'asinh for similar effect with simpler interpretation'
        },
        'U_GT2': {
            'issues': [
                "Mostly positive (81.65%) but spans zero",
                "Positive skew (2.30)",
                "2.66% exact zeros",
                "Poslog works but shift affects interpretation"
            ],
            'best_transform': 'poslog or asinh',
            'reason': "Data is mostly positive, so poslog acceptable, but asinh avoids shift",
            'alternative': 'Yeo-Johnson for optimal normality'
        },
        'U_SYMGT2': {
            'issues': [
                "Nearly symmetric (48.67% negative)",
                "2.66% zeros",
                "Small scale (std 0.009)",
                "Poslog shift breaks symmetry"
            ],
            'best_transform': 'symmetric_log',
            'reason': "Preserves near-symmetry and handles zeros",
            'alternative': 'Simple standardization may suffice given small range'
        },
        'U_SYMAT4': {
            'issues': [
                "Perfectly symmetric (50-50 split)",
                "No zeros but zero mean",
                "Similar to U_SYMAT2",
                "Poslog destroys symmetry"
            ],
            'best_transform': 'symmetric_log',
            'reason': "Same as U_SYMAT2 - preserves symmetry",
            'alternative': 'asinh'
        },
        'U_SYMAT3': {
            'issues': [
                "Negative skew (-1.81) with negative mean (-0.059)",
                "61.28% negative values",
                "Large range relative to other variables",
                "Poslog shift of 1.66 is very large"
            ],
            'best_transform': 'asinh',
            'reason': "Handles negative skew without large shifts",
            'alternative': 'Yeo-Johnson for normality'
        }
    }
    
    for var_name, info in recommendations.items():
        print(f"\n{var_name}:")
        print("-" * len(var_name))
        print("\nIssues with poslog:")
        for issue in info['issues']:
            print(f"  • {issue}")
        print(f"\nRecommended: {info['best_transform']}")
        print(f"Reason: {info['reason']}")
        print(f"Alternative: {info['alternative']}")
    
    print("\n" + "=" * 80)
    print("IMPLEMENTATION GUIDE")
    print("=" * 80)
    
    print("""
1. For symmetric variables (U_SYMAT2, U_SYMGT2, U_SYMAT4):
   ```python
   transformed = np.sign(data) * np.log1p(np.abs(data))
   ```

2. For skewed variables with negatives (U_B2, U_SYMAT3):
   ```python
   transformed = np.arcsinh(data)  # or np.arcsinh(data/scale)
   ```

3. For mostly positive with some negatives (U_GT2):
   ```python
   # Either poslog (if you must) or preferably:
   transformed = np.arcsinh(data)
   ```

4. To reverse transformations:
   - Symmetric log: `original = np.sign(transformed) * (np.exp(np.abs(transformed)) - 1)`
   - Asinh: `original = np.sinh(transformed)`
   - Poslog: `original = np.exp(transformed) - epsilon + data_min`
""")

def create_visual_comparison():
    """Create a visual comparison of transformations on synthetic data."""
    
    # Create synthetic data mimicking each variable type
    np.random.seed(42)
    n = 10000
    
    # Symmetric around zero (like U_SYMAT2)
    symmetric_data = np.random.laplace(0, 0.05, n)
    
    # Skewed with negatives (like U_B2)
    skewed_data = np.random.exponential(0.003, n) - 0.01
    skewed_data[skewed_data > 0.02] = 0.02  # Cap outliers
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Symmetric data
    data = symmetric_data
    
    ax = axes[0, 0]
    ax.hist(data, bins=50, alpha=0.7, density=True)
    ax.set_title('Original (Symmetric)')
    ax.set_xlabel('Value')
    
    ax = axes[0, 1]
    transformed = poslog_transform(data)
    ax.hist(transformed, bins=50, alpha=0.7, density=True, color='red')
    ax.set_title('Poslog (Breaks Symmetry!)')
    
    ax = axes[0, 2]
    transformed = symmetric_log_transform(data)
    ax.hist(transformed, bins=50, alpha=0.7, density=True, color='green')
    ax.set_title('Symmetric Log (Good!)')
    
    ax = axes[0, 3]
    transformed = asinh_transform(data)
    ax.hist(transformed, bins=50, alpha=0.7, density=True, color='blue')
    ax.set_title('Asinh (Also Good!)')
    
    # Row 2: Skewed data
    data = skewed_data
    
    ax = axes[1, 0]
    ax.hist(data, bins=50, alpha=0.7, density=True)
    ax.set_title('Original (Skewed)')
    ax.set_xlabel('Value')
    
    ax = axes[1, 1]
    transformed = poslog_transform(data)
    ax.hist(transformed, bins=50, alpha=0.7, density=True, color='red')
    ax.set_title('Poslog (Large Shift)')
    
    ax = axes[1, 2]
    transformed = symmetric_log_transform(data)
    ax.hist(transformed, bins=50, alpha=0.7, density=True, color='green')
    ax.set_title('Symmetric Log')
    
    ax = axes[1, 3]
    transformed = asinh_transform(data)
    ax.hist(transformed, bins=50, alpha=0.7, density=True, color='blue')
    ax.set_title('Asinh (Best for Skewed)')
    
    plt.suptitle('Transformation Comparison: Symmetric vs Skewed Data', fontsize=16)
    plt.tight_layout()
    
    os.makedirs('transformation_comparison', exist_ok=True)
    plt.savefig('transformation_comparison/transformation_fixes.png', dpi=150)
    plt.close()
    
    print("\nVisual comparison saved to 'transformation_comparison/transformation_fixes.png'")

def main():
    analyze_and_recommend()
    create_visual_comparison()

if __name__ == "__main__":
    main()