"""Visualization functions for tree canopy data"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from config import COLORMAP_COLORS, COLORMAP_POSITIONS
from analysis import calculate_difference_statistics, analyze_negative_changes

def create_custom_colormap():
    """Create custom diverging colormap with emphasis on negative values"""
    return LinearSegmentedColormap.from_list("custom", list(zip(COLORMAP_POSITIONS, COLORMAP_COLORS)))

def plot_difference_map(ax, diff_data, year1, year2):
    """Plot difference map for a pair of years"""
    # Calculate statistics
    stats, valid_diffs = calculate_difference_statistics(diff_data)
    max_abs_change = max(abs(stats['min_diff']), abs(stats['max_diff']))
    max_abs_change = max(max_abs_change, 2.0)  # Ensure at least ±2% range
    
    # Create plot
    im = ax.imshow(diff_data, 
                  cmap=create_custom_colormap(),
                  vmin=-max_abs_change,
                  vmax=max_abs_change,
                  interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Change in Tree Canopy Cover (%)')
    
    # Add ticks to colorbar
    tick_values = np.linspace(-max_abs_change, max_abs_change, 9)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{val:.1f}%' for val in tick_values])
    
    # Add title with statistics
    ax.set_title(f'Change in Coverage {year1} to {year2}\n'
                 f'Mean Change: {stats["mean_change"]:.1f}%\n'
                 f'Range: {stats["min_diff"]:.1f}% to {stats["max_diff"]:.1f}%\n'
                 f'Significant Decrease (>1%): {stats["neg_changes"]:.1f}% of area\n'
                 f'Significant Increase (>1%): {stats["pos_changes"]:.1f}% of area\n'
                 f'Minor/No Change (±1%): {stats["no_changes"]:.1f}% of area',
                 fontsize=8)
    
    ax.axis('off')
    return stats, valid_diffs

def print_difference_statistics(stats, valid_diffs, year1, year2):
    """Print statistics for difference between years"""
    print(f"\nDifference Map Statistics for {year1}-{year2}:")
    print(f"Min difference: {stats['min_diff']:.1f}%")
    print(f"Max difference: {stats['max_diff']:.1f}%")
    print(f"Mean change: {stats['mean_change']:.1f}%")
    print(f"Areas with significant decrease (>1%): {stats['neg_changes']:.1f}%")
    print(f"Areas with significant increase (>1%): {stats['pos_changes']:.1f}%")
    print(f"Areas with minor/no change (±1%): {stats['no_changes']:.1f}%")
    
    # Print histogram
    hist, bins = np.histogram(valid_diffs, bins=100)
    print("\nDetailed histogram of difference values:")
    for i in range(len(hist)):
        if hist[i] > 0:  # Only print bins with data
            print(f"Bin {bins[i]:.1f} to {bins[i+1]:.1f}: {hist[i]} pixels")
    
    # Print negative changes statistics
    neg_stats = analyze_negative_changes(valid_diffs)
    if neg_stats:
        print("\nNegative changes statistics:")
        print(f"Number of pixels with decrease > 1%: {neg_stats['count']}")
        print(f"Mean decrease: {neg_stats['mean']:.1f}%")
        print(f"Median decrease: {neg_stats['median']:.1f}%")
        print(f"Max decrease: {neg_stats['max_decrease']:.1f}%")

def create_difference_maps(data_by_year, output_dir):
    """Create maps showing the difference in tree canopy between years"""
    if len(data_by_year) < 2:
        return
    
    years = sorted(data_by_year.keys())
    n_comparisons = len(years) - 1
    
    fig, axes = plt.subplots(1, n_comparisons, figsize=(6*n_comparisons, 6))
    if n_comparisons == 1:
        axes = [axes]
    
    for i in range(n_comparisons):
        year1, year2 = years[i], years[i+1]
        data1, _ = data_by_year[year1]
        data2, meta = data_by_year[year2]
        
        # Calculate difference
        diff = calculate_difference(data1, data2)
        
        # Plot and get statistics
        stats, valid_diffs = plot_difference_map(axes[i], diff, year1, year2)
        
        # Print statistics
        print_difference_statistics(stats, valid_diffs, year1, year2)
    
    plt.suptitle('Changes in Tree Canopy Coverage (2011-2021)', y=1.05, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'difference_maps.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_difference(data1, data2):
    """Calculate difference between two years' data"""
    # Handle no-data values (254 and 255)
    nodata_mask1 = (data1 == 254) | (data1 == 255)
    nodata_mask2 = (data2 == 254) | (data2 == 255)
    
    # Create valid masks for each year's data
    valid_mask1 = (data1 >= 0) & (data1 <= 100) & ~nodata_mask1
    valid_mask2 = (data2 >= 0) & (data2 <= 100) & ~nodata_mask2
    
    # Calculate difference only where both years have valid data
    combined_mask = valid_mask1 & valid_mask2
    
    # Initialize with NaN and calculate difference
    diff = np.full_like(data1, np.nan, dtype=float)
    valid_data1 = data1[combined_mask].astype(float)
    valid_data2 = data2[combined_mask].astype(float)
    diff[combined_mask] = valid_data2 - valid_data1
    
    return diff
