"""Analysis functions for tree canopy data"""

import numpy as np
from utils import get_valid_data

def calculate_coverage_classes(valid_data):
    """Calculate coverage class distributions"""
    total = len(valid_data)
    return {
        'no_trees': np.sum(valid_data == 0) / total * 100,
        'low_canopy': np.sum((valid_data > 0) & (valid_data <= 25)) / total * 100,
        'medium_canopy': np.sum((valid_data > 25) & (valid_data <= 50)) / total * 100,
        'high_canopy': np.sum(valid_data > 50) / total * 100
    }

def analyze_tree_canopy(tree_cover, year):
    """Analyze tree canopy coverage statistics for Buffalo"""
    if tree_cover is None:
        return None
    
    valid_data = get_valid_data(tree_cover)
    coverage_classes = calculate_coverage_classes(valid_data)
    
    stats = {
        'year': year,
        'mean_coverage': np.mean(valid_data),
        'median_coverage': np.median(valid_data),
        'min_coverage': np.min(valid_data),
        'max_coverage': np.max(valid_data),
        'std_coverage': np.std(valid_data),
        'total_pixels': len(valid_data),
        'pixels_with_trees': np.sum(valid_data > 0),
        'percent_area_with_trees': (np.sum(valid_data > 0) / len(valid_data)) * 100,
        'area_high_canopy': coverage_classes['high_canopy'],
        **coverage_classes,
        'valid_data': valid_data
    }
    
    print_statistics(stats)
    return stats

def print_statistics(stats):
    """Print tree canopy statistics"""
    print(f"\nBuffalo Tree Canopy Statistics for {stats['year']}:")
    print(f"Mean tree canopy cover: {stats['mean_coverage']:.1f}%")
    print(f"Median tree canopy cover: {stats['median_coverage']:.1f}%")
    print(f"Maximum canopy cover: {stats['max_coverage']:.1f}%")
    print(f"Standard deviation: {stats['std_coverage']:.1f}%")
    print(f"Percent of area with trees: {stats['percent_area_with_trees']:.1f}%")
    print(f"Percent of area with high canopy (>50%): {stats['area_high_canopy']:.1f}%")

def calculate_difference_statistics(diff_data):
    """Calculate statistics for difference between two years"""
    valid_diffs = diff_data[~np.isnan(diff_data)]
    total_valid = len(valid_diffs)
    
    stats = {
        'min_diff': np.nanmin(valid_diffs),
        'max_diff': np.nanmax(valid_diffs),
        'mean_change': np.nanmean(valid_diffs),
        'neg_changes': np.sum(valid_diffs < -1.0) / total_valid * 100,
        'pos_changes': np.sum(valid_diffs > 1.0) / total_valid * 100,
    }
    stats['no_changes'] = 100 - stats['neg_changes'] - stats['pos_changes']
    
    return stats, valid_diffs

def analyze_negative_changes(valid_diffs):
    """Analyze negative changes in tree canopy"""
    neg_diffs = valid_diffs[valid_diffs < -1.0]
    if len(neg_diffs) > 0:
        return {
            'count': len(neg_diffs),
            'mean': np.mean(neg_diffs),
            'median': np.median(neg_diffs),
            'max_decrease': np.min(neg_diffs)
        }
    return None
