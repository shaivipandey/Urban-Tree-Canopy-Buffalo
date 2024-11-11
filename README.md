<<<<<<< HEAD
# Urban-Tree-Canopy-Buffalo
This project analyzes the tree canopy coverage in Buffalo, NY, over the years 2011 to 2021. It explores changes in urban tree canopy, identifying trends, patterns, and neighborhood-specific insights. The findings can potentially be linked to impacts on air quality and urban planning.

## Project Files
1. spatial_analysis.py
Purpose: Analyzes spatial patterns in tree canopy coverage across Buffalo.
Functionality: Identifies areas with high and low tree coverage (hot and cold spots) and checks if canopy patterns are random or clustered.
Output: Generates maps and scatter plots to visualize canopy distribution.

2. neighborhood_analysis.py
Purpose: Examines tree canopy coverage within specific neighborhoods.
Functionality: Summarizes canopy coverage trends for each neighborhood, highlighting significant changes over time.
Output: Provides data summaries for neighborhood-level insights.

3. main.py
Purpose: Main script to run the entire analysis.
Functionality: Integrates results from spatial_analysis.py and neighborhood_analysis.py to produce final outputs.
Output: Generates a comprehensive report and combined visualizations.

## Output
Hot Spot Maps: Visual maps highlighting neighborhoods with high (hot spots) and low (cold spots) tree canopy coverage. These maps show where Buffalo has significant greenery and where it lacks tree coverage.

Yearly Change Maps: Year-over-year visual comparisons that show how tree canopy coverage has changed across the city, indicating areas with noticeable decline or growth.

Moran’s I Scatter Plots: These plots provide a simple way to understand the spatial distribution of tree coverage—whether neighborhoods with high or low canopy are clustered together or randomly distributed.

Trend and Distribution Charts: Graphs showing the overall decline in Buffalo’s tree canopy coverage over time, including year-over-year changes and breakdowns by canopy class.
=======
# Buffalo Tree Canopy Analysis

This project analyzes tree canopy coverage in Buffalo, NY using NLCD (National Land Cover Database) data. It includes tools for analyzing changes in tree coverage over time, neighborhood-level analysis, and spatial statistics.

## Project Structure

```
.
├── __init__.py          # Package initialization
├── config.py            # Configuration settings and constants
├── utils.py             # Core utility functions
├── analysis.py          # Analysis functions
├── visualization.py     # Visualization functions
├── main.py             # Main script
├── spatial_analysis.py  # Spatial statistics analysis
└── neighborhood_analysis.py  # Neighborhood-level analysis
```

## Installation

1. Create conda environment:
```bash
conda create -n tree-canopy -c conda-forge python=3.10 rasterio numpy pandas matplotlib geopandas contextily pyproj seaborn -y
```

2. Activate environment:
```bash
conda activate tree-canopy
```

## Usage

### Basic Analysis
Run the main analysis script:
```bash
python main.py
```

### Neighborhood Analysis
Analyze tree canopy by neighborhood:
```bash
python neighborhood_analysis.py
```

### Spatial Statistics
Run spatial statistics analysis:
```bash
python spatial_analysis.py
```

## Data Requirements

Place your data files in the following structure:
```
data/
├── Neighborhoods_20241102.csv
└── nlcd_tcc_CONUS_YYYY_v2021-4/
    └── nlcd_tcc_conus_YYYY_v2021-4.tif
```
Where YYYY represents years (2011, 2016, 2021)

## Output

Results are saved in the `output/` directory:
- Difference maps showing changes between years
- Statistical reports
- Visualization plots
- Neighborhood-level analysis
- Spatial statistics results

## Features

1. **Basic Analysis**
   - Tree canopy coverage statistics
   - Change detection between years
   - Visualization of changes

2. **Neighborhood Analysis**
   - Per-neighborhood statistics
   - Tree density calculations
   - Temporal change analysis

3. **Spatial Statistics**
   - Moran's I analysis
   - Hot spot analysis
   - Spatial clustering visualization

## Dependencies

- rasterio: Raster data handling
- numpy: Numerical computations
- pandas: Data manipulation
- matplotlib: Plotting
- geopandas: Geospatial data handling
- contextily: Base maps
- pyproj: Coordinate transformations
- seaborn: Statistical visualizations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License
>>>>>>> 2848e9a (Initial commit: Buffalo Tree Canopy Analysis project)
