"""
Data exporting functions for Street Level Change Detection.

This module provides functions for exporting panorama data to various formats,
including CSV files, GeoJSON files, and databases.
"""

import os
import pandas as pd
import geopandas as gpd
from typing import Optional, Union, Dict, Any, List

from src.core.panorama import PanoramaCollection


def ensure_directory_exists(file_path: str) -> None:
    """
    Ensure the directory for a file exists.
    
    Parameters
    ----------
    file_path : str
        Path to the file
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def export_panorama_data(
    data: Union[pd.DataFrame, gpd.GeoDataFrame, PanoramaCollection],
    file_path: str,
    format: Optional[str] = None
) -> str:
    """
    Export panorama data to a file.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, gpd.GeoDataFrame, PanoramaCollection]
        Data to export
    file_path : str
        Path to save the file
    format : Optional[str], default=None
        Format to export (csv, geojson, etc.). If None, inferred from file extension.
        
    Returns
    -------
    str
        Path to the exported file
    """
    # Ensure directory exists
    ensure_directory_exists(file_path)
    
    # Convert PanoramaCollection to GeoDataFrame if needed
    if isinstance(data, PanoramaCollection):
        data = data.to_geodataframe()
    
    # Determine format from file extension if not specified
    if format is None:
        _, ext = os.path.splitext(file_path)
        format = ext.lower().lstrip('.')
    
    # Export based on format
    if format in ['csv']:
        data.to_csv(file_path, index=False)
    elif format in ['geojson', 'json']:
        # Ensure we have a GeoDataFrame
        if isinstance(data, pd.DataFrame) and not isinstance(data, gpd.GeoDataFrame):
            if 'lat' in data.columns and 'lon' in data.columns:
                from shapely.geometry import Point
                geometries = [Point(lon, lat) for lon, lat in zip(data['lon'], data['lat'])]
                data = gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:4326")
            else:
                raise ValueError("Cannot convert to GeoJSON without geometry or lat/lon columns")
        
        # Export to GeoJSON
        data.to_file(file_path, driver='GeoJSON')
    else:
        raise ValueError(f"Unsupported export format: {format}")
    
    print(f"Exported data to {file_path}")
    return file_path

def export_to_csv(data: Union[pd.DataFrame, gpd.GeoDataFrame, PanoramaCollection], file_path: str) -> str:
    """
    Export data to a CSV file.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, gpd.GeoDataFrame, PanoramaCollection]
        Data to export
    file_path : str
        Path to save the CSV file
        
    Returns
    -------
    str
        Path to the exported file
    """
    if isinstance(data, PanoramaCollection):
        export_panorama_data(data, file_path, format='csv')
    else:
        data.to_csv(file_path, index=False)
    return file_path


def export_to_geojson(data: Union[gpd.GeoDataFrame, PanoramaCollection], file_path: str) -> str:
    """
    Export data to a GeoJSON file.
    
    Parameters
    ----------
    data : Union[gpd.GeoDataFrame, PanoramaCollection]
        Data to export
    file_path : str
        Path to save the GeoJSON file
        
    Returns
    -------
    str
        Path to the exported file
    """
    return export_panorama_data(data, file_path, format='geojson')