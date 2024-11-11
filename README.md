
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
