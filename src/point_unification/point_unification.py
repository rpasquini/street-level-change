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


def bbox_unification(df: Union[pd.DataFrame, gpd.GeoDataFrame], 
                    epsilon: float = 0.0001,
                    output_dir: str = 'data/point_unification') -> tuple:
    """
    Group panorama points using bounding box matching and graph-based connected components.
    
    Parameters
    ----------
    df : Union[pd.DataFrame, gpd.GeoDataFrame]
        Input dataframe containing panorama points with columns: pano_id, lat, lon
        (or geometry column with Point objects)
    epsilon : float, default 0.0001
        Size of the bounding box in degrees (approximately 10 meters)
    output_dir : str, default 'data/point_unification'
        Directory to save output CSV files
        
    Returns
    -------
    tuple
        (clusters_df, centroids_df) containing the cluster assignments and centroids
        
    Notes
    -----
    - Uses bounding box matching to find nearby points
    - Builds a graph where each point is a node and nearby points are connected by edges
    - Uses networkx connected components to identify clusters
    - Calculates centroids as the mean lat/lon of points in each cluster
    - Saves results to CSV files
    """
    import os
    import networkx as nx
    import pandas as pd
    from collections import defaultdict
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure we have lat/lon columns
    if isinstance(df, gpd.GeoDataFrame) and 'geometry' in df.columns:
        if 'lat' not in df.columns or 'lon' not in df.columns:
            df['lat'] = df.geometry.y
            df['lon'] = df.geometry.x
    elif 'lat' not in df.columns or 'lon' not in df.columns:
        raise ValueError("Input dataframe must have 'lat' and 'lon' columns or a geometry column")
    
    if 'pano_id' not in df.columns:
        raise ValueError("Input dataframe must have a 'pano_id' column")
    
    # Create a spatial index using a dictionary of grid cells
    grid_index = defaultdict(list)
    
    # Assign each point to grid cells
    for idx, row in df.iterrows():
        lat, lon = row['lat'], row['lon']
        # Get grid cell coordinates (floor division by epsilon)
        cell_lat = int(lat / epsilon)
        cell_lon = int(lon / epsilon)
        
        # Add point to its cell and neighboring cells for efficient lookup
        for i in range(-1, 2):  # -1, 0, 1
            for j in range(-1, 2):  # -1, 0, 1
                grid_index[(cell_lat + i, cell_lon + j)].append(idx)
    
    # Build graph of connected points
    G = nx.Graph()
    
    # Add all points as nodes
    for idx in df.index:
        G.add_node(idx)
    
    # Find nearby points and add edges
    for idx, row in df.iterrows():
        lat, lon = row['lat'], row['lon']
        cell_lat = int(lat / epsilon)
        cell_lon = int(lon / epsilon)
        
        # Get potential neighbors from the grid cells
        potential_neighbors = set()
        for i in range(-1, 2):  # -1, 0, 1
            for j in range(-1, 2):  # -1, 0, 1
                potential_neighbors.update(grid_index[(cell_lat + i, cell_lon + j)])
        
        # Check actual distance and add edges
        for neighbor_idx in potential_neighbors:
            if idx != neighbor_idx:  # Don't compare with self
                neighbor = df.iloc[neighbor_idx]
                # Check if within bounding box
                if (abs(lat - neighbor['lat']) <= epsilon and 
                    abs(lon - neighbor['lon']) <= epsilon):
                    G.add_edge(idx, neighbor_idx)
    
    # Find connected components (clusters)
    clusters = list(nx.connected_components(G))
    
    # Assign cluster IDs
    cluster_mapping = {}
    for cluster_id, cluster in enumerate(clusters):
        for node in cluster:
            cluster_mapping[node] = cluster_id
    
    # Create clusters dataframe
    clusters_df = pd.DataFrame({
        'pano_id': df['pano_id'],
        'cluster_id': [cluster_mapping.get(idx, -1) for idx in df.index]
    })
    
    # Calculate centroids
    centroids = []
    for cluster_id in range(len(clusters)):
        # Get indices of points in this cluster
        cluster_points = [idx for idx, cid in cluster_mapping.items() if cid == cluster_id]
        # Calculate mean lat/lon
        mean_lat = df.loc[cluster_points, 'lat'].mean()
        mean_lon = df.loc[cluster_points, 'lon'].mean()
        centroids.append({
            'cluster_id': cluster_id,
            'latitude': mean_lat,
            'longitude': mean_lon
        })
    
    centroids_df = pd.DataFrame(centroids)
    
    # Save to CSV
    clusters_path = os.path.join(output_dir, 'bbox_clusters.csv')
    centroids_path = os.path.join(output_dir, 'bbox_cluster_centroids.csv')
    
    clusters_df.to_csv(clusters_path, index=False)
    centroids_df.to_csv(centroids_path, index=False)
    
    print(f"Found {len(clusters)} clusters")
    print(f"Saved cluster assignments to {clusters_path}")
    print(f"Saved cluster centroids to {centroids_path}")
    
    return clusters_df, centroids_df


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
