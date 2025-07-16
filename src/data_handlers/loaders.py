"""
Data loading functions for Street Level Change Detection.

This module provides functions for loading panorama data from various sources,
including CSV files, GeoJSON files, and databases.
"""

import os
import pandas as pd
import geopandas as gpd
from shapely import wkt
from typing import Optional, Union, Dict, Any

from src.core.panorama import PanoramaCollection


def load_panorama_data(file_path: str) -> gpd.GeoDataFrame:
    """
    Load panorama data from a file and convert to GeoDataFrame.
    
    This function attempts to load data from various file formats and
    convert it to a GeoDataFrame with proper geometry column.
    
    Parameters
    ----------
    file_path : str
        Path to the file containing panorama data
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with panorama data
        
    Raises
    ------
    ValueError
        If the file cannot be loaded or converted to a GeoDataFrame
    """
    print(f"Loading panorama data from {file_path}")
    
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    try:
        # Try to load as GeoDataFrame first (for GeoJSON, Shapefile, etc.)
        if ext in ['.geojson', '.shp']:
            gdf = gpd.read_file(file_path)
            print(f"Loaded {len(gdf)} panoramas as GeoDataFrame")
            return gdf
        
        # For CSV files
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} panoramas as DataFrame")
            
            # Check if 'geometry' column exists
            if 'geometry' in df.columns:
                # Try to convert WKT strings to geometry objects
                try:
                    geometries = df['geometry'].apply(wkt.loads)
                    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
                    return gdf
                except Exception as e:
                    print(f"Error converting geometry column: {e}")
            
            # If no geometry column or conversion failed, try using lat/lon columns
            if 'lat' in df.columns and 'lon' in df.columns:
                from shapely.geometry import Point
                geometries = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
                gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
                return gdf
            
            raise ValueError("Could not find geometry or lat/lon columns in the data")
        
        # For other file types, try pandas read functions
        else:
            df = pd.read_csv(file_path)  # Default to CSV
            
            # Try to convert to GeoDataFrame
            if 'lat' in df.columns and 'lon' in df.columns:
                from shapely.geometry import Point
                geometries = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
                gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
                return gdf
            
            raise ValueError(f"Unsupported file format: {ext}")
            
    except Exception as e:
        print(f"Error loading panorama data: {e}")
        raise


def load_panorama_collection(file_path: str) -> PanoramaCollection:
    """
    Load panorama data from a file and convert to a PanoramaCollection.
    
    Parameters
    ----------
    file_path : str
        Path to the file containing panorama data
        
    Returns
    -------
    PanoramaCollection
        Collection of panoramas
    """
    # Load as GeoDataFrame first
    gdf = load_panorama_data(file_path)
    
    # Convert to PanoramaCollection
    return PanoramaCollection.from_geodataframe(gdf)


def load_from_csv(file_path: str) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Load data from a CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file
        
    Returns
    -------
    Union[pd.DataFrame, gpd.GeoDataFrame]
        DataFrame or GeoDataFrame with loaded data
    """
    print(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    if "geometry" in data.columns:
        data["geometry"] = data["geometry"].apply(wkt.loads)
        return gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:4326")
    return data
