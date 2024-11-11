"""Main script for tree canopy analysis"""

import os
from config import DATA_PATHS, OUTPUT_DIR
from utils import load_and_clip_tree_canopy
from analysis import analyze_tree_canopy
from visualization import create_difference_maps

def main():
    """Main function to run the tree canopy analysis"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each year
    data_by_year = {}
    stats_list = []
    
    print("Loading and processing Buffalo tree canopy data for multiple years...")
    
    for year, path in DATA_PATHS.items():
        print(f"\nProcessing year {year}...")
        tree_cover, meta = load_and_clip_tree_canopy(path)
        
        if tree_cover is not None:
            data_by_year[year] = (tree_cover, meta)
            stats = analyze_tree_canopy(tree_cover, year)
            if stats:
                stats_list.append(stats)
    
    if data_by_year:
        print("\nGenerating visualizations and analysis...")
        create_difference_maps(data_by_year, OUTPUT_DIR)
        print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}/")
    else:
        print("No data was successfully loaded.")

if __name__ == "__main__":
    main()
