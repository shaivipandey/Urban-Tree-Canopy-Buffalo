"""Generate comprehensive visualizations for tree canopy analysis"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def create_neighborhood_rankings(df, output_dir):
    """Create rankings visualization for neighborhoods"""
    # Get 2021 data
    df_2021 = df[df['year'] == 2021].sort_values('mean_coverage', ascending=True)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Create horizontal bar chart
    bars = plt.barh(df_2021['neighborhood'], df_2021['mean_coverage'])
    
    # Color bars based on coverage
    norm = plt.Normalize(df_2021['mean_coverage'].min(), df_2021['mean_coverage'].max())
    colors = plt.cm.YlGn(norm(df_2021['mean_coverage']))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.title('Tree Canopy Coverage by Neighborhood (2021)', pad=20)
    plt.xlabel('Mean Coverage (%)')
    
    # Add value labels
    for i, v in enumerate(df_2021['mean_coverage']):
        plt.text(v + 0.1, i, f'{v:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neighborhood_rankings_2021.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_change_analysis(df, output_dir):
    """Create visualization of changes from 2011 to 2021"""
    # Calculate changes
    changes = []
    for neighborhood in df['neighborhood'].unique():
        data_2011 = df[(df['year'] == 2011) & (df['neighborhood'] == neighborhood)].iloc[0]
        data_2021 = df[(df['year'] == 2021) & (df['neighborhood'] == neighborhood)].iloc[0]
        change = data_2021['mean_coverage'] - data_2011['mean_coverage']
        changes.append({
            'neighborhood': neighborhood,
            'change': change,
            'initial_coverage': data_2011['mean_coverage'],
            'final_coverage': data_2021['mean_coverage']
        })
    
    changes_df = pd.DataFrame(changes).sort_values('change')
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Create bars
    bars = plt.barh(changes_df['neighborhood'], changes_df['change'])
    
    # Color bars based on change
    for bar in bars:
        if bar.get_width() < 0:
            bar.set_color('#d73027')  # Red for decrease
        else:
            bar.set_color('#1a9850')  # Green for increase
    
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.2)
    plt.title('Change in Tree Canopy Coverage (2011-2021)', pad=20)
    plt.xlabel('Change in Coverage (%)')
    
    # Add value labels
    for i, v in enumerate(changes_df['change']):
        plt.text(v + (0.1 if v >= 0 else -0.1), i,
                f'{v:+.1f}%',
                va='center',
                ha='left' if v >= 0 else 'right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neighborhood_changes_2011_2021.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_time_series_heatmap(df, output_dir):
    """Create heatmap showing coverage over time for all neighborhoods"""
    # Pivot data for heatmap
    pivot_df = df.pivot(index='neighborhood', columns='year', values='mean_coverage')
    
    # Create figure
    plt.figure(figsize=(15, 12))
    
    # Create heatmap
    sns.heatmap(pivot_df, cmap='YlGn', center=pivot_df.mean().mean(),
                annot=True, fmt='.1f', cbar_kws={'label': 'Tree Canopy Coverage (%)'})
    
    plt.title('Tree Canopy Coverage by Neighborhood Over Time', pad=20)
    plt.xlabel('Year')
    plt.ylabel('Neighborhood')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coverage_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_density_analysis(df, output_dir):
    """Create visualization of tree density patterns"""
    # Get 2021 data
    df_2021 = df[df['year'] == 2021].sort_values('trees_per_acre', ascending=True)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Create horizontal bar chart
    bars = plt.barh(df_2021['neighborhood'], df_2021['trees_per_acre'])
    
    # Color bars based on density
    norm = plt.Normalize(df_2021['trees_per_acre'].min(), df_2021['trees_per_acre'].max())
    colors = plt.cm.YlGn(norm(df_2021['trees_per_acre']))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.title('Tree Density by Neighborhood (2021)', pad=20)
    plt.xlabel('Trees per Acre')
    
    # Add value labels
    for i, v in enumerate(df_2021['trees_per_acre']):
        plt.text(v + 0.01, i, f'{v:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tree_density_2021.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_report(df, output_dir):
    """Create text report summarizing key findings"""
    report = "BUFFALO TREE CANOPY ANALYSIS SUMMARY (2011-2021)\n"
    report += "=" * 50 + "\n\n"
    
    # Overall city changes
    city_2011 = df[df['year'] == 2011]['mean_coverage'].mean()
    city_2021 = df[df['year'] == 2021]['mean_coverage'].mean()
    total_change = city_2021 - city_2011
    
    report += "CITY-WIDE CHANGES\n"
    report += "-" * 20 + "\n"
    report += f"2011 Average Coverage: {city_2011:.1f}%\n"
    report += f"2021 Average Coverage: {city_2021:.1f}%\n"
    report += f"Total Change: {total_change:+.1f}%\n\n"
    
    # Best and worst performing neighborhoods
    changes = []
    for neighborhood in df['neighborhood'].unique():
        data_2011 = df[(df['year'] == 2011) & (df['neighborhood'] == neighborhood)].iloc[0]
        data_2021 = df[(df['year'] == 2021) & (df['neighborhood'] == neighborhood)].iloc[0]
        change = data_2021['mean_coverage'] - data_2011['mean_coverage']
        changes.append({
            'neighborhood': neighborhood,
            'change': change,
            'coverage_2021': data_2021['mean_coverage']
        })
    
    changes_df = pd.DataFrame(changes)
    
    report += "TOP 5 NEIGHBORHOODS BY CURRENT COVERAGE (2021)\n"
    report += "-" * 45 + "\n"
    for _, row in changes_df.nlargest(5, 'coverage_2021').iterrows():
        report += f"{row['neighborhood']}: {row['coverage_2021']:.1f}%\n"
    report += "\n"
    
    report += "TOP 5 NEIGHBORHOODS BY COVERAGE LOSS\n"
    report += "-" * 35 + "\n"
    for _, row in changes_df.nsmallest(5, 'change').iterrows():
        report += f"{row['neighborhood']}: {row['change']:+.1f}%\n"
    
    # Save report
    with open(os.path.join(output_dir, 'visualization_summary.txt'), 'w') as f:
        f.write(report)

def main():
    """Main function to generate all visualizations"""
    # Create output directory
    output_dir = "output/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    df = pd.read_csv("output/neighborhood_tree_canopy.csv")
    
    # Create visualizations
    create_neighborhood_rankings(df, output_dir)
    create_change_analysis(df, output_dir)
    create_time_series_heatmap(df, output_dir)
    create_density_analysis(df, output_dir)
    create_summary_report(df, output_dir)
    
    print(f"Visualizations saved to {output_dir}/")

if __name__ == "__main__":
    main()
