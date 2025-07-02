"""
Point unification algorithms for Street Level Change Detection.

This module provides algorithms for unifying panorama points using
different clustering methods, including H3, DBSCAN, and bounding box.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from shapely.geometry import Point, Polygon
import h3
from sklearn.cluster import DBSCAN
from tqdm import tqdm

import networkx as nx
from sklearn.neighbors import BallTree
import numpy as np


from src.core.panorama import PanoramaCollection


def unify_points_h3(
    gdf: Union[gpd.GeoDataFrame, PanoramaCollection],
    resolution: int = 10,
    id_column: str = 'location_id'
) -> gpd.GeoDataFrame:
    """
    Unify points using H3 hexagonal grid.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with points to unify
    resolution : int, default=10
        H3 resolution (0-15, where 15 is highest resolution)
    id_column : str, default='location_id'
        Name of the column to store H3 indices
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with unified points
    """
    # Make a copy to avoid modifying the original
    if isinstance(gdf, PanoramaCollection):
        gdf = gdf.to_geodataframe()
    result = gdf.copy()
    
    # Ensure the GeoDataFrame has a proper CRS
    if result.crs is None:
        result.set_crs(epsg=4326, inplace=True)
    elif result.crs != "EPSG:4326":
        result = result.to_crs(epsg=4326)
    
    # Calculate H3 indices
    result[id_column] = result.apply(
        lambda row: h3.latlng_to_cell(row.geometry.y, row.geometry.x, resolution),
        axis=1
    )
    
    return result


def unify_points_dbscan(
    gdf: Union[gpd.GeoDataFrame, PanoramaCollection],
    eps: float = 0.0001,
    min_samples: int = 1,
    id_column: str = 'location_id'
) -> gpd.GeoDataFrame:
    """
    Unify points using DBSCAN clustering.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with points to unify
    eps : float, default=0.0001
        Maximum distance between points in a cluster
    min_samples : int, default=1
        Minimum number of points to form a cluster
    id_column : str, default='location_id'
        Name of the column to store cluster IDs
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with unified points
    """
    if isinstance(gdf, PanoramaCollection):
        gdf = gdf.to_geodataframe()
    
    # Make a copy to avoid modifying the original
    result = gdf.copy()
    
    # Ensure the GeoDataFrame has a proper CRS
    if result.crs is None:
        result.set_crs(epsg=4326, inplace=True)
    elif result.crs != "EPSG:4326":
        result = result.to_crs(epsg=4326)
    
    # Extract coordinates for clustering
    coords = np.array([(p.x, p.y) for p in result.geometry])
    
    # Apply DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    
    # Add cluster labels to the GeoDataFrame
    result[id_column] = db.labels_
    
    return result


def unify_points_bounding_box(
    gdf: Union[gpd.GeoDataFrame, 'PanoramaCollection'],
    box_size: float = 0.0001
) -> gpd.GeoDataFrame:
    """
    Unify points using bounding box grid.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with points to unify. Must have POINT geometries.
    box_size : float, default=0.0001
        Size of the bounding box grid cells (in degrees).
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with an additional `cluster_id` column.
    """
    if hasattr(gdf, "to_geodataframe"):  # support PanoramaCollection-style wrapper
        gdf = gdf.to_geodataframe()

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError("Input must be a GeoDataFrame or PanoramaCollection")

    if not gdf.geometry.iloc[0].geom_type == "Point":
        raise ValueError("Geometry column must contain POINT geometries")

    gdf = gdf.copy()
    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x
    gdf = gdf.reset_index(drop=True)

    # Sort for efficient join
    gdf_sorted = gdf.sort_values(["lat", "lon"]).reset_index(drop=True)

    matches = []

    # Brute force: only check small sliding window in sorted coordinates
    for i, row in gdf_sorted.iterrows():
        lat_min = row["lat"] - box_size
        lat_max = row["lat"] + box_size
        lon_min = row["lon"] - box_size
        lon_max = row["lon"] + box_size

        subset = gdf_sorted[
            (gdf_sorted["lat"] >= lat_min) & (gdf_sorted["lat"] <= lat_max) &
            (gdf_sorted["lon"] >= lon_min) & (gdf_sorted["lon"] <= lon_max)
        ]

        for j in subset.index:
            if i < j:
                matches.append((i, j))

    # Build graph from matched index pairs
    G = nx.Graph()
    G.add_nodes_from(gdf.index)
    G.add_edges_from(matches)

    # Connected components = clusters
    index_to_cluster = {}
    for cluster_id, component in enumerate(nx.connected_components(G)):
        for idx in component:
            index_to_cluster[idx] = cluster_id

    gdf["cluster_id"] = gdf.index.map(index_to_cluster)

    return gdf

def unify_panorama_collection(
    panoramas: PanoramaCollection,
    method: str = 'h3',
    **kwargs
) -> PanoramaCollection:
    """
    Unify points in a PanoramaCollection.
    
    Parameters
    ----------
    panoramas : PanoramaCollection
        Collection of panoramas to unify
    method : str, default='h3'
        Unification method ('h3', 'dbscan', or 'bounding_box')
    **kwargs
        Additional parameters for the unification method
        
    Returns
    -------
    PanoramaCollection
        Collection with unified panoramas
    """
    # Convert to GeoDataFrame
    gdf = panoramas.to_geodataframe()
    
    # Apply unification method
    if method == 'h3':
        resolution = kwargs.get('resolution', 10)
        result_gdf = unify_points_h3(gdf, resolution=resolution)
    elif method == 'dbscan':
        eps = kwargs.get('eps', 0.0001)
        min_samples = kwargs.get('min_samples', 1)
        result_gdf = unify_points_dbscan(gdf, eps=eps, min_samples=min_samples)
    elif method == 'bounding_box':
        box_size = kwargs.get('box_size', 0.0001)
        result_gdf = unify_points_bounding_box(gdf, box_size=box_size)
    else:
        raise ValueError(f"Unknown unification method: {method}")
    
    # Convert back to PanoramaCollection
    result_collection = PanoramaCollection.from_geodataframe(result_gdf)
    
    return result_collection
