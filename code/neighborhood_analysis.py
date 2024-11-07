import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
from shapely.wkt import loads
from pyproj import CRS
import os

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
                'area_high_canopy': np.sum(valid_data > 50) / len(valid_data) * 100,
                'total_pixels': len(valid_data),
                'pixels_with_trees': np.sum(valid_data > 0),
                'estimated_trees': np.sum(valid_data > 0) * 0.09  # Approximate number of trees based on pixel size
            }
            
            return stats
            
        except Exception as e:
            print(f"Error processing neighborhood: {e}")
            return None

def calculate_changes(df):
    """
    Calculate year-over-year changes and add summary statistics
    """
    # Sort by neighborhood and year
    df = df.sort_values(['neighborhood', 'year'])
    
    # Calculate year-over-year changes
    df['change_from_previous'] = df.groupby('neighborhood')['mean_coverage'].diff()
    df['change_from_2011'] = df.groupby('neighborhood')['mean_coverage'].transform(
        lambda x: x - x.iloc[0]
    )
    
    # Calculate trees per acre
    df['trees_per_acre'] = df['estimated_trees'] / df['acres']
    
    # Calculate summary statistics by neighborhood
    summary = df.groupby('neighborhood').agg({
        'acres': 'first',
        'mean_coverage': ['first', 'last', 'mean'],
        'trees_per_acre': ['first', 'last', 'mean'],
        'change_from_2011': 'last'
    }).round(2)
    
    summary.columns = [
        'acres',
        'coverage_2011',
        'coverage_2021',
        'coverage_mean',
        'density_2011',
        'density_2021',
        'density_mean',
        'total_change'
    ]
    
    return df, summary

def save_results(df, summary, output_dir):
    """
    Save results with proper formatting
    """
    # Format column headers for better readability
    df.columns = [
        'Neighborhood',
        'Year',
        'Acres',
        'Mean Coverage (%)',
        'Median Coverage (%)',
        'Std Coverage',
        'Area with Trees (%)',
        'High Canopy Area (%)',
        'Est. Tree Count',
        'Trees per Acre',
        'Change from Prev Year (%)',
        'Change from 2011 (%)'
    ]
    
    summary.columns = [
        'Total Acres',
        'Coverage 2011 (%)',
        'Coverage 2021 (%)',
        'Mean Coverage (%)',
        'Density 2011 (trees/acre)',
        'Density 2021 (trees/acre)',
        'Mean Density (trees/acre)',
        'Total Change 2011-2021 (%)'
    ]
    
    # Save detailed results
    output_path = os.path.join(output_dir, 'neighborhood_tree_canopy.csv')
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path}")
    
    # Save summary results
    summary_path = os.path.join(output_dir, 'neighborhood_summary.csv')
    summary.to_csv(summary_path)
    print(f"Summary results saved to {summary_path}")

def print_key_findings(df, summary):
    """
    Print key findings in a formatted way
    """
    print("\nKEY FINDINGS: BUFFALO TREE CANOPY ANALYSIS (2011-2021)")
    print("=" * 80)
    print(f"Total neighborhoods analyzed: {len(summary)}")
    
    # Overall city statistics
    city_2011 = df[df['Year'] == 2011]['Mean Coverage (%)'].mean()
    city_2021 = df[df['Year'] == 2021]['Mean Coverage (%)'].mean()
    city_change = city_2021 - city_2011
    
    print(f"\nCITY-WIDE STATISTICS:")
    print(f"  Average tree coverage 2011: {city_2011:.1f}%")
    print(f"  Average tree coverage 2021: {city_2021:.1f}%")
    print(f"  Overall change: {city_change:+.1f}%")
    
    # Coverage rankings 2021
    coverage_2021 = df[df['Year'] == 2021].sort_values('Mean Coverage (%)', ascending=False)
    
    print("\nTOP 5 NEIGHBORHOODS BY TREE COVERAGE (2021):")
    for _, row in coverage_2021.head().iterrows():
        print(f"  {row['Neighborhood']:25} {row['Mean Coverage (%)']:5.1f}%")
    
    print("\nBOTTOM 5 NEIGHBORHOODS BY TREE COVERAGE (2021):")
    for _, row in coverage_2021.tail().iterrows():
        print(f"  {row['Neighborhood']:25} {row['Mean Coverage (%)']:5.1f}%")
    
    # Biggest changes
    changes = summary.sort_values('Total Change 2011-2021 (%)', ascending=False)
    
    print("\nTOP 5 NEIGHBORHOODS BY COVERAGE IMPROVEMENT:")
    for idx, row in changes.head().iterrows():
        print(f"  {idx:25} {row['Total Change 2011-2021 (%):']:+5.1f}%")
    
    print("\nTOP 5 NEIGHBORHOODS BY COVERAGE DECLINE:")
    for idx, row in changes.tail().iterrows():
        print(f"  {idx:25} {row['Total Change 2011-2021 (%):']:+5.1f}%")
    
    # Density statistics
    density_2021 = df[df['Year'] == 2021].sort_values('Trees per Acre', ascending=False)
    
    print("\nTOP 5 NEIGHBORHOODS BY TREE DENSITY (2021):")
    for _, row in density_2021.head().iterrows():
        print(f"  {row['Neighborhood']:25} {row['Trees per Acre']:5.1f} trees/acre")

def main():
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load neighborhoods
    neighborhoods = load_neighborhoods("data/Neighborhoods_20241102.csv")
    
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
    
    # Process each neighborhood for each year
    results = []
    
    print("Analyzing tree canopy by neighborhood...")
    total_neighborhoods = len(neighborhoods)
    
    for idx, row in neighborhoods.iterrows():
        print(f"\nProcessing neighborhood {idx + 1}/{total_neighborhoods}: {row['Neighborhood Name']}")
        
        for year, path in data_paths.items():
            print(f"  Year {year}...", end='', flush=True)
            
            stats = analyze_neighborhood_canopy(row.geometry, path)
            if stats:
                stats['neighborhood'] = row['Neighborhood Name']
                stats['year'] = year
                stats['acres'] = row['CalcAcres']
                results.append(stats)
                print(" done")
            else:
                print(" failed")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate changes and summary statistics
    df, summary = calculate_changes(df)
    
    # Save results with proper formatting
    save_results(df, summary, output_dir)
    
    # Print key findings
    print_key_findings(df, summary)

if __name__ == "__main__":
    main()
