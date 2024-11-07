import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.mask import mask
from shapely.geometry import box, Polygon
import contextily as ctx
from pyproj import CRS, Transformer
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime

def get_buffalo_bounds():
    """
    Get Buffalo's bounds in the NLCD coordinate system (Albers Equal Area)
    Buffalo approximate bounds:
    - Longitude: -78.9377 to -78.8127
    - Latitude: 42.8448 to 42.9563
    """
    # Create transformer from WGS84 to NLCD's Albers projection
    source_crs = CRS.from_epsg(4326)  # WGS84
    target_crs = CRS.from_string('+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs')
    
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    
    # Buffalo coordinates
    buffalo_coords = {
        'west': -78.9377,
        'east': -78.8127,
        'north': 42.9563,
        'south': 42.8448
    }
    
    # Transform coordinates
    west, south = transformer.transform(buffalo_coords['west'], buffalo_coords['south'])
    east, north = transformer.transform(buffalo_coords['east'], buffalo_coords['north'])
    
    return box(west, south, east, north)

def load_and_clip_tree_canopy(tiff_path):
    """
    Load NLCD tree canopy data and clip to Buffalo area
    """
    buffalo_geom = get_buffalo_bounds()
    
    with rasterio.open(tiff_path) as src:
        try:
            out_image, out_transform = mask(src, [buffalo_geom], crop=True)
            out_meta = src.meta.copy()
            
            # Update metadata
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            return out_image[0], out_meta
            
        except ValueError as e:
            print(f"Error loading {tiff_path}: {e}")
            return None, None

def analyze_tree_canopy(tree_cover, year):
    """
    Analyze tree canopy coverage statistics for Buffalo
    """
    if tree_cover is None:
        return None
    
    # Create mask for valid data (0-100, excluding 254 and 255)
    valid_mask = (tree_cover >= 0) & (tree_cover <= 100)
    valid_data = tree_cover[valid_mask]
    
    # Calculate coverage class distributions
    no_trees = np.sum(valid_data == 0) / len(valid_data) * 100
    low_canopy = np.sum((valid_data > 0) & (valid_data <= 25)) / len(valid_data) * 100
    medium_canopy = np.sum((valid_data > 25) & (valid_data <= 50)) / len(valid_data) * 100
    high_canopy = np.sum(valid_data > 50) / len(valid_data) * 100
    
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
        'area_high_canopy': high_canopy,
        'no_trees': no_trees,
        'low_canopy': low_canopy,
        'medium_canopy': medium_canopy,
        'valid_data': valid_data
    }
    
    print(f"\nBuffalo Tree Canopy Statistics for {year}:")
    print(f"Mean tree canopy cover: {stats['mean_coverage']:.1f}%")
    print(f"Median tree canopy cover: {stats['median_coverage']:.1f}%")
    print(f"Maximum canopy cover: {stats['max_coverage']:.1f}%")
    print(f"Standard deviation: {stats['std_coverage']:.1f}%")
    print(f"Percent of area with trees: {stats['percent_area_with_trees']:.1f}%")
    print(f"Percent of area with high canopy (>50%): {stats['area_high_canopy']:.1f}%")
    
    return stats

def create_difference_maps(data_by_year, output_dir):
    """
    Create maps showing the difference in tree canopy between years
    """
    if len(data_by_year) < 2:
        return
    
    years = sorted(data_by_year.keys())
    n_comparisons = len(years) - 1
    
    fig, axes = plt.subplots(1, n_comparisons, figsize=(6*n_comparisons, 6))
    if n_comparisons == 1:
        axes = [axes]
    
    # Create custom diverging colormap (red to white to green)
    colors = ['darkred', 'red', 'white', 'green', 'darkgreen']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    for i in range(n_comparisons):
        year1, year2 = years[i], years[i+1]
        data1, _ = data_by_year[year1]
        data2, meta = data_by_year[year2]
        
        # Calculate difference
        valid_mask = (data1 >= 0) & (data1 <= 100) & (data2 >= 0) & (data2 <= 100)
        diff = np.where(valid_mask, data2 - data1, np.nan)
        
        # Plot difference
        im = axes[i].imshow(diff, 
                          cmap=cmap,
                          vmin=-20,  # Adjust range based on your data
                          vmax=20)
        
        plt.colorbar(im, ax=axes[i], label='Change in Tree Canopy Cover (%)')
        axes[i].set_title(f'Change in Coverage\n{year1} to {year2}')
        axes[i].axis('off')
    
    plt.suptitle('Changes in Tree Canopy Coverage (2011-2021)', y=1.05, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'difference_maps.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_enhanced_visualizations(stats_list, data_by_year, output_dir):
    """
    Create enhanced visualizations showing patterns and trends
    """
    if not stats_list:
        return
    
    # Convert stats list to DataFrame and ensure proper sorting
    df = pd.DataFrame(stats_list).sort_values('year')
    years = df['year'].tolist()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Violin plot of distributions
    ax1 = fig.add_subplot(gs[0, 0])
    violin_data = []
    violin_labels = []
    for year, (data, _) in sorted(data_by_year.items()):
        valid_mask = (data >= 0) & (data <= 100)
        valid_data = data[valid_mask]
        violin_data.append(valid_data)
        violin_labels.append(str(year))
    
    violin_parts = ax1.violinplot(violin_data, showmeans=True)
    ax1.set_xticks(range(1, len(violin_labels) + 1))
    ax1.set_xticklabels(violin_labels)
    ax1.set_title('Distribution of Tree Canopy Coverage')
    ax1.set_ylabel('Percentage Cover')
    
    # Customize violin plot colors
    for pc in violin_parts['bodies']:
        pc.set_facecolor('green')
        pc.set_alpha(0.3)
    violin_parts['cmeans'].set_color('darkgreen')
    
    # 2. Stacked bar chart of coverage classes
    ax2 = fig.add_subplot(gs[0, 1])
    coverage_classes = np.array([
        df['no_trees'].tolist(),
        df['low_canopy'].tolist(),
        df['medium_canopy'].tolist(),
        df['area_high_canopy'].tolist()
    ])
    
    bottom = np.zeros(len(years))
    colors = ['lightgray', 'lightgreen', 'green', 'darkgreen']
    labels = ['No Trees (0%)', 'Low (1-25%)', 'Medium (26-50%)', 'High (>50%)']
    
    for i, coverage in enumerate(coverage_classes):
        ax2.bar(years, coverage, bottom=bottom, label=labels[i], color=colors[i])
        bottom += coverage
    
    ax2.set_title('Distribution of Canopy Coverage Classes')
    ax2.set_ylabel('Percentage of Area')
    ax2.legend()
    
    # 3. Trend lines with confidence intervals
    ax3 = fig.add_subplot(gs[1, 0])
    years_array = np.array(years)
    coverage_array = np.array(df['mean_coverage'])
    
    # Calculate trend line
    z = np.polyfit(years_array, coverage_array, 1)
    p = np.poly1d(z)
    
    # Calculate confidence intervals
    n = len(years_array)
    m = np.mean(years_array)
    se = np.sqrt(np.sum((coverage_array - p(years_array)) ** 2) / (n-2) / np.sum((years_array - m) ** 2))
    
    ax3.plot(years_array, coverage_array, 'o-', label='Observed', color='green', linewidth=2)
    ax3.plot(years_array, p(years_array), '--', label='Trend', color='red', linewidth=2)
    ax3.fill_between(years_array, 
                     p(years_array) - 2*se, 
                     p(years_array) + 2*se, 
                     alpha=0.2, 
                     color='gray',
                     label='95% Confidence Interval')
    
    ax3.set_title('Tree Canopy Coverage Trend')
    ax3.set_ylabel('Mean Coverage (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Year-over-year changes
    ax4 = fig.add_subplot(gs[1, 1])
    changes = []
    periods = []
    for i in range(len(years)-1):
        change = df['mean_coverage'].iloc[i+1] - df['mean_coverage'].iloc[i]
        changes.append(change)
        periods.append(f'{years[i]}-{years[i+1]}')
    
    colors = ['red' if x < 0 else 'green' for x in changes]
    ax4.bar(periods, changes, color=colors)
    ax4.set_title('Year-over-Year Changes in Tree Canopy Coverage')
    ax4.set_ylabel('Change in Mean Coverage (%)')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on the bars
    for i, v in enumerate(changes):
        ax4.text(i, v + (0.1 if v >= 0 else -0.1),
                f'{v:.1f}%',
                ha='center',
                va='bottom' if v >= 0 else 'top')
    
    plt.suptitle('Buffalo Tree Canopy Analysis (2011-2021)', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trend_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save statistics to CSV
    df.to_csv(os.path.join(output_dir, 'tree_canopy_statistics.csv'), index=False)

def generate_report(stats_list, output_dir):
    """
    Generate a comprehensive report of the analysis
    """
    if not stats_list:
        return
    
    df = pd.DataFrame(stats_list).sort_values('year')
    start_year = df.iloc[0]['year']
    end_year = df.iloc[-1]['year']
    
    report = f"""Buffalo Tree Canopy Analysis Report ({start_year}-{end_year})
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

SUMMARY OF FINDINGS
------------------
"""
    
    # Calculate total changes
    total_change_mean = df.iloc[-1]['mean_coverage'] - df.iloc[0]['mean_coverage']
    total_change_area = df.iloc[-1]['percent_area_with_trees'] - df.iloc[0]['percent_area_with_trees']
    
    # Calculate annual rates of change
    years_diff = df.iloc[-1]['year'] - df.iloc[0]['year']
    annual_rate_mean = total_change_mean / years_diff
    annual_rate_area = total_change_area / years_diff
    
    report += f"""
Overall Changes:
- Total Change in Mean Coverage: {total_change_mean:.1f}%
- Total Change in Area with Trees: {total_change_area:.1f}%
- Average Annual Change in Mean Coverage: {annual_rate_mean:.2f}% per year
- Average Annual Change in Area with Trees: {annual_rate_area:.2f}% per year

Year-by-Year Statistics:
-----------------------"""
    
    for _, row in df.iterrows():
        report += f"""
{row['year']}:
- Mean tree canopy cover: {row['mean_coverage']:.1f}%
- Median tree canopy cover: {row['median_coverage']:.1f}%
- Maximum canopy cover: {row['max_coverage']:.1f}%
- Standard deviation: {row['std_coverage']:.1f}%
- Percent of area with trees: {row['percent_area_with_trees']:.1f}%
- Percent of area with high canopy (>50%): {row['area_high_canopy']:.1f}%"""
    
    report += "\n\nYear-over-Year Changes:\n----------------------"
    for i in range(len(df)-1):
        year1 = df.iloc[i]['year']
        year2 = df.iloc[i+1]['year']
        change = df.iloc[i+1]['mean_coverage'] - df.iloc[i]['mean_coverage']
        report += f"\n{year1} to {year2}: {change:+.1f}%"
    
    # Save report
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
        f.write(report)

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paths for all years
    data_paths = {
        2011: "data/nlcd_tcc_CONUS_2011_v2021-4/nlcd_tcc_conus_2011_v2021-4.tif",
        2012: "data/nlcd_tcc_CONUS_2012_v2021-4/nlcd_tcc_conus_2012_v2021-4.tif",
        2013: "data/nlcd_tcc_CONUS_2013_v2021-4/nlcd_tcc_conus_2013_v2021-4.tif",
        2014: "data/nlcd_tcc_CONUS_2014_v2021-4/nlcd_tcc_conus_2014_v2021-4.tif",
        2015: "data/nlcd_tcc_CONUS_2015_v2021-4/nlcd_tcc_conus_2015_v2021-4.tif",
        2016: "data/nlcd_tcc_CONUS_2016_v2021-4/nlcd_tcc_conus_2016_v2021-4.tif",
        2017: "data/nlcd_tcc_CONUS_2017_v2021-4/nlcd_tcc_conus_2017_v2021-4.tif",
        2018: "data/nlcd_tcc_CONUS_2018_v2021-4/nlcd_tcc_conus_2018_v2021-4.tif",
        2019: "data/nlcd_tcc_CONUS_2019_v2021-4/nlcd_tcc_conus_2019_v2021-4.tif",
        2020: "data/nlcd_tcc_CONUS_2020_v2021-4/nlcd_tcc_conus_2020_v2021-4.tif",
        2021: "data/nlcd_tcc_CONUS_2021_v2021-4/nlcd_tcc_conus_2021_v2021-4.tif"
    }
    
    # Process each year
    data_by_year = {}
    stats_list = []
    
    print("Loading and processing Buffalo tree canopy data for multiple years...")
    
    for year, path in data_paths.items():
        print(f"\nProcessing year {year}...")
        tree_cover, meta = load_and_clip_tree_canopy(path)
        
        if tree_cover is not None:
            data_by_year[year] = (tree_cover, meta)
            stats = analyze_tree_canopy(tree_cover, year)
            if stats:
                stats_list.append(stats)
    
    if data_by_year:
        print("\nGenerating visualizations and analysis...")
        create_difference_maps(data_by_year, output_dir)
        create_enhanced_visualizations(stats_list, data_by_year, output_dir)
        generate_report(stats_list, output_dir)
        print(f"\nAnalysis complete. Results saved to {output_dir}/")
    else:
        print("No data was successfully loaded.")
