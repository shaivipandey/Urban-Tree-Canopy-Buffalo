"""Main script for tree canopy analysis"""

import os
from config import DATA_PATHS, OUTPUT_DIR
from utils import load_and_clip_tree_canopy
from analysis import analyze_tree_canopy
from visualization import create_difference_maps
from neighborhood_analysis import (load_neighborhoods, analyze_neighborhood_canopy,
                                calculate_changes, save_results, print_key_findings)
from neighborhood_visualization import (create_neighborhood_time_series,
                                    create_comparative_visualization,
                                    generate_neighborhood_report)
from spatial_analysis import calculate_spatial_statistics

def process_city_analysis():
    """Process city-wide tree canopy analysis"""
    print("\nProcessing city-wide analysis...")
    
    # Process each year
    data_by_year = {}
    stats_list = []
    
    for year, path in DATA_PATHS.items():
        print(f"\nProcessing year {year}...")
        tree_cover, meta = load_and_clip_tree_canopy(path)
        
        if tree_cover is not None:
            data_by_year[year] = (tree_cover, meta)
            stats = analyze_tree_canopy(tree_cover, year)
            if stats:
                stats_list.append(stats)
    
    if data_by_year:
        print("\nGenerating city-wide visualizations...")
        create_difference_maps(data_by_year, OUTPUT_DIR)
    
    return data_by_year, stats_list

def process_neighborhood_analysis():
    """Process neighborhood-level analysis"""
    print("\nProcessing neighborhood analysis...")
    
    # Load neighborhoods
    neighborhoods = load_neighborhoods("data/Neighborhoods_20241102.csv")
    
    # Process each neighborhood for each year
    results = []
    total_neighborhoods = len(neighborhoods)
    
    for idx, row in neighborhoods.iterrows():
        print(f"\nProcessing neighborhood {idx + 1}/{total_neighborhoods}: {row['Neighborhood Name']}")
        
        for year, path in DATA_PATHS.items():
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
    
    # Calculate changes and create visualizations
    if results:
        print("\nGenerating neighborhood visualizations...")
        df = pd.DataFrame(results)
        df, summary = calculate_changes(df)
        
        # Save results
        save_results(df, summary, OUTPUT_DIR)
        
        # Create visualizations
        create_neighborhood_time_series(df, OUTPUT_DIR)
        create_comparative_visualization(df, OUTPUT_DIR)
        generate_neighborhood_report(df, OUTPUT_DIR)
        
        # Print findings
        print_key_findings(df, summary)
        
        return df, summary
    
    return None, None

def main():
    """Main function to run all analyses"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run city-wide analysis
    data_by_year, city_stats = process_city_analysis()
    
    # Run neighborhood analysis
    neighborhood_df, neighborhood_summary = process_neighborhood_analysis()
    
    if data_by_year and neighborhood_df is not None:
        print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}/")
        print("\nKey directories:")
        print(f"- City-wide analysis: {OUTPUT_DIR}/")
        print(f"- Neighborhood analysis: {OUTPUT_DIR}/neighborhood_analysis/")
        print(f"- Spatial statistics: {OUTPUT_DIR}/spatial_statistics/")
    else:
        print("\nError: Some analyses failed to complete.")

if __name__ == "__main__":
    main()
