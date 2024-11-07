import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
from shapely.wkt import loads
from pyproj import CRS
import os
from libpysal.weights import Queen
from esda.moran import Moran
from esda.getisord import G_Local
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import contextily as ctx

def load_neighborhoods(csv_path):
    """
    Load Buffalo neighborhoods from CSV and convert to GeoDataFrame
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Convert WKT geometry strings to shapely geometries
    df['geometry'] = df['Geometry'].apply(loads)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.set_crs(epsg=4326, inplace=True)
    
    # Transform to the NLCD projection
    target_crs = CRS.from_string('+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs')
    gdf = gdf.to_crs(target_crs)
    
    return gdf

def analyze_neighborhood_canopy(neighborhood_geom, tiff_path):
    """
    Analyze tree canopy for a specific neighborhood
    """
    with rasterio.open(tiff_path) as src:
        try:
            # Mask the raster data to the neighborhood boundary
            out_image, _ = mask(src, [neighborhood_geom], crop=True)
            data = out_image[0]
            
            # Create mask for valid data (0-100)
            valid_mask = (data >= 0) & (data <= 100)
            valid_data = data[valid_mask]
            
            if len(valid_data) == 0:
                return None
            
            # Calculate statistics
            stats = {
                'mean_coverage': np.mean(valid_data),
                'median_coverage': np.median(valid_data),
                'std_coverage': np.std(valid_data),
                'percent_area_with_trees': (np.sum(valid_data > 0) / len(valid_data)) * 100,
                'area_high_canopy': np.sum(valid_data > 50) / len(valid_data) * 100
            }
            
            return stats
            
        except Exception as e:
            print(f"Error processing neighborhood: {e}")
            return None

def calculate_spatial_statistics(gdf, year_data):
    """
    Calculate spatial statistics (Moran's I and Getis-Ord Gi*)
    """
    # Create spatial weights matrix using Queen contiguity
    weights = Queen.from_dataframe(gdf, use_index=True)
    weights.transform = 'r'  # Row-standardize weights
    
    # Calculate Moran's I
    moran = Moran(year_data, weights)
    
    # Calculate Local Getis-Ord Gi*
    g_local = G_Local(year_data, weights)
    
    return moran, g_local

def plot_moran_scatterplot(moran, year, output_dir):
    """
    Create Moran's I scatterplot
    """
    plt.figure(figsize=(10, 10))
    
    # Calculate standardized variables
    zx = (moran.y - np.mean(moran.y)) / np.std(moran.y)
    zy = (moran.w.sparse.dot(moran.y) - np.mean(moran.y)) / np.std(moran.y)
    
    # Plot scatter points
    plt.scatter(zx, zy, c='b', alpha=0.6)
    
    # Calculate and plot regression line
    slope = np.polyfit(zx, zy, 1)[0]
    line_x = np.array([min(zx), max(zx)])
    plt.plot(line_x, slope * line_x, 'r', alpha=0.9)
    
    # Add quadrant lines
    plt.axvline(x=0, color='k', alpha=0.5)
    plt.axhline(y=0, color='k', alpha=0.5)
    
    plt.title(f"Moran's I Scatterplot ({year})\nI={moran.I:.3f}, p={moran.p_sim:.3f}")
    plt.xlabel('Standardized Tree Coverage')
    plt.ylabel('Spatial Lag of Standardized Tree Coverage')
    
    # Add quadrant labels
    plt.text(max(zx)*0.7, max(zy)*0.7, 'High-High\n(Clusters)', ha='center', va='center')
    plt.text(min(zx)*0.7, max(zy)*0.7, 'Low-High\n(Spatial Outliers)', ha='center', va='center')
    plt.text(max(zx)*0.7, min(zy)*0.7, 'High-Low\n(Spatial Outliers)', ha='center', va='center')
    plt.text(min(zx)*0.7, min(zy)*0.7, 'Low-Low\n(Clusters)', ha='center', va='center')
    
    plt.savefig(os.path.join(output_dir, f'moran_scatter_{year}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_hotspot_map(gdf, g_local, year, output_dir):
    """
    Create hotspot analysis map
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Calculate significance levels
    sig_levels = np.array([0.001, 0.01, 0.05])
    
    # Create custom colormap for hot and cold spots
    colors = ['darkblue', 'blue', 'lightblue', 'white', 'white',
              'salmon', 'red', 'darkred']
    n_bins = 8
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    # Classify areas based on significance and direction
    classification = np.zeros(len(gdf))
    for i, p in enumerate(g_local.p_sim):
        if p <= 0.001:
            classification[i] = 3 if g_local.Zs[i] > 0 else -3
        elif p <= 0.01:
            classification[i] = 2 if g_local.Zs[i] > 0 else -2
        elif p <= 0.05:
            classification[i] = 1 if g_local.Zs[i] > 0 else -1
        else:
            classification[i] = 0
    
    # Create custom legend labels
    legend_labels = {
        -3: 'Cold Spot (99.9% confidence)',
        -2: 'Cold Spot (99% confidence)',
        -1: 'Cold Spot (95% confidence)',
        0: 'Not Significant',
        1: 'Hot Spot (95% confidence)',
        2: 'Hot Spot (99% confidence)',
        3: 'Hot Spot (99.9% confidence)'
    }
    
    # Plot map
    gdf.plot(column=classification, 
             ax=ax,
             cmap=cmap,
             legend=True,
             legend_kwds={'label': 'Tree Canopy Clustering'})
    
    # Add neighborhood labels
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        ax.annotate(row['Neighborhood Name'],
                   xy=(centroid.x, centroid.y),
                   ha='center',
                   va='center',
                   fontsize=8)
    
    plt.title(f'Tree Canopy Coverage Hot Spots ({year})')
    
    # Save map
    plt.savefig(os.path.join(output_dir, f'hotspot_map_{year}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(morans_results, output_dir):
    """
    Generate a report of the spatial analysis results
    """
    report = "BUFFALO TREE CANOPY SPATIAL ANALYSIS REPORT\n"
    report += "=" * 50 + "\n\n"
    
    report += "MORAN'S I STATISTICS BY YEAR\n"
    report += "-" * 30 + "\n\n"
    
    # Sort results by year
    years = sorted(morans_results.keys())
    
    for year in years:
        moran = morans_results[year]
        report += f"{year}:\n"
        report += f"  Moran's I: {moran.I:.3f}\n"
        report += f"  p-value: {moran.p_sim:.3f}\n"
        report += f"  z-score: {moran.z_sim:.3f}\n"
        
        # Add interpretation
        if moran.p_sim < 0.05:
            if moran.I > 0:
                pattern = "significant clustering"
            else:
                pattern = "significant dispersion"
        else:
            pattern = "random spatial pattern"
        report += f"  Interpretation: {pattern}\n\n"
    
    # Add detailed interpretation
    report += "\nINTERPRETATION GUIDE\n"
    report += "-" * 20 + "\n\n"
    
    report += "Moran's I Interpretation:\n"
    report += "- Values range from -1 (perfect dispersion) to +1 (perfect clustering)\n"
    report += "- Values near 0 indicate random spatial pattern\n"
    report += "- Positive values indicate similar values cluster together\n"
    report += "- Negative values indicate dissimilar values cluster together\n\n"
    
    report += "Statistical Significance:\n"
    report += "- p-value < 0.05 indicates statistically significant spatial pattern\n"
    report += "- z-score > 1.96 or < -1.96 indicates significant clustering at 95% confidence\n\n"
    
    report += "Hot Spot Analysis Interpretation:\n"
    report += "- Hot Spots: Areas where high tree coverage clusters together\n"
    report += "- Cold Spots: Areas where low tree coverage clusters together\n"
    report += "- Confidence levels indicate strength of clustering:\n"
    report += "  * 99.9%: Very strong evidence of clustering\n"
    report += "  * 99%: Strong evidence of clustering\n"
    report += "  * 95%: Moderate evidence of clustering\n\n"
    
    report += "Implications:\n"
    report += "- Clusters of high tree coverage (hot spots) may indicate:\n"
    report += "  * Well-established neighborhoods with mature trees\n"
    report += "  * Areas with successful tree planting programs\n"
    report += "  * Parks and green spaces\n"
    report += "- Clusters of low tree coverage (cold spots) may indicate:\n"
    report += "  * Areas that need targeted tree planting efforts\n"
    report += "  * Areas with recent development or urban density\n"
    report += "  * Industrial or commercial zones\n"
    
    # Save report
    with open(os.path.join(output_dir, 'spatial_analysis_report.txt'), 'w') as f:
        f.write(report)

def main():
    # Create output directory
    output_dir = "output/Geospatial Analysis and Spatial Statistics"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load neighborhoods
    print("Loading neighborhood data...")
    neighborhoods = load_neighborhoods("data/Neighborhoods_20241102.csv")
    
    # Define analysis years
    analysis_years = [2011, 2016, 2021]  # Analyze start, middle, and end years
    
    # Store Moran's I results
    morans_results = {}
    
    # Process each year
    for year in analysis_years:
        print(f"\nAnalyzing spatial patterns for {year}...")
        
        # Get tree canopy data for each neighborhood
        coverage_data = []
        for idx, row in neighborhoods.iterrows():
            tiff_path = f"data/nlcd_tcc_CONUS_{year}_v2021-4/nlcd_tcc_conus_{year}_v2021-4.tif"
            stats = analyze_neighborhood_canopy(row.geometry, tiff_path)
            if stats:
                coverage_data.append(stats['mean_coverage'])
            else:
                coverage_data.append(np.nan)
        
        # Add coverage data to GeoDataFrame
        neighborhoods[f'coverage_{year}'] = coverage_data
        
        # Calculate spatial statistics
        moran, g_local = calculate_spatial_statistics(
            neighborhoods, 
            neighborhoods[f'coverage_{year}']
        )
        
        # Store results
        morans_results[year] = moran
        
        # Create visualizations
        print("Creating visualizations...")
        plot_moran_scatterplot(moran, year, output_dir)
        plot_hotspot_map(neighborhoods, g_local, year, output_dir)
    
    # Generate report
    print("\nGenerating spatial analysis report...")
    generate_report(morans_results, output_dir)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
