"""Visualization functions for neighborhood-level tree canopy analysis"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

def create_neighborhood_time_series(stats_df, output_dir):
    """Create time series plots for each neighborhood"""
    # Create output subdirectory
    neighborhood_dir = os.path.join(output_dir, 'neighborhood_analysis')
    os.makedirs(neighborhood_dir, exist_ok=True)
    
    # Set up the style
    plt.style.use('seaborn')
    
    # Plot for each neighborhood
    for neighborhood in stats_df['neighborhood'].unique():
        neighborhood_data = stats_df[stats_df['neighborhood'] == neighborhood]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
        
        # 1. Time series of mean coverage
        ax1.plot(neighborhood_data['year'], neighborhood_data['mean_coverage'], 
                marker='o', linewidth=2, markersize=8)
        ax1.set_title(f'Tree Canopy Coverage Trend: {neighborhood}', pad=20)
        ax1.set_ylabel('Mean Coverage (%)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(neighborhood_data['year'], neighborhood_data['mean_coverage']):
            ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        # 2. Bar chart of year-over-year changes
        years = neighborhood_data['year'].tolist()
        changes = []
        for i in range(len(years)-1):
            change = (neighborhood_data.iloc[i+1]['mean_coverage'] - 
                     neighborhood_data.iloc[i]['mean_coverage'])
            changes.append(change)
        
        colors = ['red' if x < 0 else 'green' for x in changes]
        ax2.bar(range(len(changes)), changes, color=colors)
        ax2.set_xticks(range(len(changes)))
        ax2.set_xticklabels([f'{years[i]}-{years[i+1]}' for i in range(len(changes))])
        ax2.set_ylabel('Change in Coverage (%)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(changes):
            ax2.text(i, v + (0.1 if v >= 0 else -0.1),
                    f'{v:+.1f}%',
                    ha='center',
                    va='bottom' if v >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(neighborhood_dir, f'{neighborhood.replace(" ", "_")}_trend.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

def create_comparative_visualization(stats_df, output_dir):
    """Create comparative visualizations across neighborhoods"""
    # Calculate total change for each neighborhood
    neighborhood_changes = []
    for neighborhood in stats_df['neighborhood'].unique():
        neighborhood_data = stats_df[stats_df['neighborhood'] == neighborhood]
        first_year = neighborhood_data.iloc[0]['mean_coverage']
        last_year = neighborhood_data.iloc[-1]['mean_coverage']
        total_change = last_year - first_year
        neighborhood_changes.append({
            'neighborhood': neighborhood,
            'total_change': total_change,
            'initial_coverage': first_year,
            'final_coverage': last_year
        })
    
    changes_df = pd.DataFrame(neighborhood_changes)
    
    # Sort by total change
    changes_df = changes_df.sort_values('total_change')
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Create diverging colormap
    colors = ['#67000d', '#fc9272', '#fee0d2', '#fff', '#e5f5e0', '#74c476', '#238b45']
    n_colors = 256
    palette = LinearSegmentedColormap.from_list('custom', colors, N=n_colors)
    
    # Create bars
    bars = plt.barh(changes_df['neighborhood'], changes_df['total_change'])
    
    # Color bars based on value
    for bar in bars:
        if bar.get_width() < 0:
            bar.set_color('#fc9272')
        else:
            bar.set_color('#74c476')
    
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.2)
    plt.title('Total Change in Tree Canopy Coverage (2011-2021)', pad=20)
    plt.xlabel('Change in Coverage (%)')
    
    # Add value labels
    for i, v in enumerate(changes_df['total_change']):
        plt.text(v + (0.1 if v >= 0 else -0.1), i,
                f'{v:+.1f}%',
                va='center',
                ha='left' if v >= 0 else 'right')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neighborhood_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def generate_neighborhood_report(stats_df, output_dir):
    """Generate a comprehensive report of neighborhood-level changes"""
    report = "BUFFALO TREE CANOPY ANALYSIS BY NEIGHBORHOOD (2011-2021)\n"
    report += "=" * 80 + "\n\n"
    
    # Calculate city-wide statistics
    city_stats = {
        'initial_mean': stats_df[stats_df['year'] == 2011]['mean_coverage'].mean(),
        'final_mean': stats_df[stats_df['year'] == 2021]['mean_coverage'].mean(),
        'total_change': (stats_df[stats_df['year'] == 2021]['mean_coverage'].mean() -
                        stats_df[stats_df['year'] == 2011]['mean_coverage'].mean())
    }
    
    report += "CITY-WIDE SUMMARY\n"
    report += "-" * 20 + "\n"
    report += f"Mean Coverage 2011: {city_stats['initial_mean']:.1f}%\n"
    report += f"Mean Coverage 2021: {city_stats['final_mean']:.1f}%\n"
    report += f"Overall Change: {city_stats['total_change']:+.1f}%\n\n"
    
    # Analyze each neighborhood
    report += "NEIGHBORHOOD ANALYSIS\n"
    report += "-" * 20 + "\n\n"
    
    for neighborhood in sorted(stats_df['neighborhood'].unique()):
        neighborhood_data = stats_df[stats_df['year'] == 2021]
        neighborhood_data = neighborhood_data[neighborhood_data['neighborhood'] == neighborhood]
        
        if len(neighborhood_data) > 0:
            data = neighborhood_data.iloc[0]
            initial_data = stats_df[(stats_df['year'] == 2011) & 
                                  (stats_df['neighborhood'] == neighborhood)].iloc[0]
            
            total_change = data['mean_coverage'] - initial_data['mean_coverage']
            
            report += f"{neighborhood}\n"
            report += f"{'=' * len(neighborhood)}\n"
            report += f"Current Coverage (2021): {data['mean_coverage']:.1f}%\n"
            report += f"Change since 2011: {total_change:+.1f}%\n"
            report += f"Area with Trees: {data['percent_area_with_trees']:.1f}%\n"
            report += f"High Canopy Area (>50%): {data['area_high_canopy']:.1f}%\n\n"
    
    # Save report
    with open(os.path.join(output_dir, 'neighborhood_analysis_report.txt'), 'w') as f:
        f.write(report)

def main():
    """Main function to create neighborhood visualizations"""
    # Create output directory
    output_dir = "output/neighborhood_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process data
    # Note: This would need to be integrated with the main data processing pipeline
    
    # Create visualizations
    # Note: These functions would be called with the processed data
    
    print(f"Analysis complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
