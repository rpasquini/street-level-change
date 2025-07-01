"""
Point unification module for grouping panorama points based on spatial proximity.

This module provides two methods for unifying points:
1. H3-based hexagon grouping
2. DBSCAN clustering with haversine distance

Both methods return the original dataframe with an additional location_id column
that can be used to identify points that are spatially close to each other.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import h3
from sklearn.cluster import DBSCAN
from typing import Union, Optional


def _ensure_geodataframe(df: Union[pd.DataFrame, gpd.GeoDataFrame], 
                         geometry_col: str = 'geometry') -> gpd.GeoDataFrame:
    """
    Ensure the input is a GeoDataFrame with proper geometry column in WGS84.
    
    Parameters
    ----------
    df : Union[pd.DataFrame, gpd.GeoDataFrame]
        Input dataframe containing point geometries or lat/lon columns
    geometry_col : str, default 'geometry'
        Name of the column containing geometries
        
    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with properly formatted geometry column in WGS84 (EPSG:4326)
        
    Raises
    ------
    ValueError
        If the geometry column doesn't exist or contains invalid geometries,
        and lat/lon columns are not available as an alternative
    TypeError
        If the input geometries are not compatible with Point objects
    """
    # Check if input is already a GeoDataFrame
    if isinstance(df, gpd.GeoDataFrame) and df.geometry.name == geometry_col:
        # Ensure CRS is set to WGS84
        if df.crs is None:
            df.set_crs(epsg=4326, inplace=True)
        elif df.crs != "EPSG:4326":
            df = df.to_crs(epsg=4326)
        return df
    
    # If it's a DataFrame, check for lat/lon columns first
    if isinstance(df, pd.DataFrame) and 'lat' in df.columns and 'lon' in df.columns:
        # Create Point geometries from lat/lon columns
        from shapely.geometry import Point
        geometries = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
        return gdf
    
    # If no lat/lon columns, check for geometry column
    if geometry_col not in df.columns:
        raise ValueError(f"Geometry column '{geometry_col}' not found in dataframe and no lat/lon columns available")
    
    # Process geometry column - handle both Point objects and WKT strings
    geometries = []
    for geom in df[geometry_col]:
        if isinstance(geom, Point):
            geometries.append(geom)
        elif isinstance(geom, str):
            try:
                geometries.append(wkt.loads(geom))
            except Exception as e:
                raise ValueError(f"Failed to parse WKT string: {e}")
        else:
            raise TypeError(f"Geometry must be a Point object or WKT string, got {type(geom)}")
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df.drop(columns=[geometry_col]), 
                           geometry=geometries, 
                           crs="EPSG:4326")
    
    # Validate all geometries are points
    if not all(isinstance(geom, Point) for geom in gdf.geometry):
        raise TypeError("All geometries must be Point objects")
    
    return gdf


def h3_unification(df: Union[pd.DataFrame, gpd.GeoDataFrame], 
                   resolution: int = 14,
                   geometry_col: str = 'geometry') -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Group points by H3 hexagon using a specified resolution.
    
    Parameters
    ----------
    df : Union[pd.DataFrame, gpd.GeoDataFrame]
        Input dataframe containing point geometries
    resolution : int, default 14
        H3 resolution level (0-15), higher values create smaller hexagons
    geometry_col : str, default 'geometry'
        Name of the column containing geometries
        
    Returns
    -------
    Union[pd.DataFrame, gpd.GeoDataFrame]
        Original dataframe with an additional 'location_id' column containing H3 indices
        
    Notes
    -----
    H3 resolution levels and approximate hexagon edge lengths:
    - Resolution 9: ~174 meters
    - Resolution 10: ~65 meters
    - Resolution 11: ~25 meters
    - Resolution 12: ~9 meters
    - Resolution 13: ~3.5 meters
    - Resolution 14: ~1.3 meters
    - Resolution 15: ~0.5 meters
    """
    # Input validation
    if not isinstance(resolution, int) or resolution < 0 or resolution > 15:
        raise ValueError("Resolution must be an integer between 0 and 15")
    
    # Ensure we have a proper GeoDataFrame
    gdf = _ensure_geodataframe(df, geometry_col)
    
    # Get lat/lon coordinates from geometry
    lats = gdf.geometry.y
    lons = gdf.geometry.x
    
    # Generate H3 indices for each point
    h3_indices = [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(lats, lons)]
    
    # Add location_id column to the original dataframe
    result = df.copy()
    result['location_id'] = h3_indices
    
    return result


def dbscan_unification(df: Union[pd.DataFrame, gpd.GeoDataFrame], 
                       eps_meters: float = 10, 
                       min_samples: int = 1,
                       geometry_col: str = 'geometry') -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Cluster points using DBSCAN with haversine distance.
    
    Parameters
    ----------
    df : Union[pd.DataFrame, gpd.GeoDataFrame]
        Input dataframe containing point geometries
    eps_meters : float, default 10
        Maximum distance (in meters) between two points for them to be considered neighbors
    min_samples : int, default 1
        Minimum number of points required to form a dense region
    geometry_col : str, default 'geometry'
        Name of the column containing geometries
        
    Returns
    -------
    Union[pd.DataFrame, gpd.GeoDataFrame]
        Original dataframe with an additional 'location_id' column containing cluster IDs
        
    Notes
    -----
    - Points assigned to cluster -1 are considered noise points by DBSCAN
    - Using min_samples=1 ensures every point gets assigned to a cluster
    - The haversine distance is used to account for Earth's curvature
    """
    # Input validation
    if eps_meters <= 0:
        raise ValueError("eps_meters must be positive")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    
    # Ensure we have a proper GeoDataFrame
    gdf = _ensure_geodataframe(df, geometry_col)
    
    # Extract coordinates and convert to radians for haversine distance
    coords = np.radians(np.vstack([gdf.geometry.y, gdf.geometry.x]).T)
    
    # Convert eps from meters to radians (approximate conversion)
    earth_radius_meters = 6371000  # Earth radius in meters
    eps_radians = eps_meters / earth_radius_meters
    
    # Define haversine distance metric for DBSCAN
    def haversine_distance(x, y):
        # Haversine formula
        lat1, lon1 = x
        lat2, lon2 = y
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return c
    
    # Run DBSCAN clustering
    dbscan = DBSCAN(eps=eps_radians, min_samples=min_samples, metric=haversine_distance)
    clusters = dbscan.fit_predict(coords)
    
    # Add location_id column to the original dataframe
    result = df.copy()
    result['location_id'] = clusters
    
    return result
