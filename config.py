"""Configuration settings and constants"""

# Buffalo coordinates
BUFFALO_COORDS = {
    'west': -78.9377,
    'east': -78.8127,
    'north': 42.9563,
    'south': 42.8448
}

# NLCD projection string
NLCD_PROJ = '+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'

# Data paths
DATA_PATHS = {
    2011: "data/nlcd_tcc_CONUS_2011_v2021-4/nlcd_tcc_conus_2011_v2021-4.tif",
    2016: "data/nlcd_tcc_CONUS_2016_v2021-4/nlcd_tcc_conus_2016_v2021-4.tif",
    2021: "data/nlcd_tcc_CONUS_2021_v2021-4/nlcd_tcc_conus_2021_v2021-4.tif"
}

# Output directory
OUTPUT_DIR = "output"

# Visualization settings
COLORMAP_COLORS = [
    '#67000d',  # Very dark red
    '#a50f15',  # Dark red
    '#ef3b2c',  # Red
    '#fc9272',  # Light red
    '#fee0d2',  # Very light red
    '#ffffff',  # White
    '#e5f5e0',  # Very light green
    '#74c476',  # Light green
    '#238b45'   # Green
]

COLORMAP_POSITIONS = [0.0, 0.15, 0.25, 0.35, 0.45, 0.5, 0.65, 0.8, 1.0]
