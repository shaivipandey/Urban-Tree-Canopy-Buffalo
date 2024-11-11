"""Core utility functions for data loading and preprocessing"""

import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from pyproj import CRS, Transformer
from config import BUFFALO_COORDS, NLCD_PROJ

def get_buffalo_bounds():
    """Get Buffalo's bounds in the NLCD coordinate system (Albers Equal Area)"""
    # Create transformer from WGS84 to NLCD's Albers projection
    source_crs = CRS.from_epsg(4326)  # WGS84
    target_crs = CRS.from_string(NLCD_PROJ)
    
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    
    # Transform coordinates
    west, south = transformer.transform(BUFFALO_COORDS['west'], BUFFALO_COORDS['south'])
    east, north = transformer.transform(BUFFALO_COORDS['east'], BUFFALO_COORDS['north'])
    
    return box(west, south, east, north)

def load_and_clip_tree_canopy(tiff_path):
    """Load NLCD tree canopy data and clip to Buffalo area"""
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

def validate_data(data):
    """Create mask for valid data (0-100, excluding 254 and 255)"""
    return (data >= 0) & (data <= 100)

def get_valid_data(data):
    """Extract valid data values"""
    valid_mask = validate_data(data)
    return data[valid_mask]
