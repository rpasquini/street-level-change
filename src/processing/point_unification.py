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
    gdf: Union[gpd.GeoDataFrame, PanoramaCollection],
    box_size: float = 0.0001,
    id_column: str = 'location_id'
) -> gpd.GeoDataFrame:
    """
    Unify points using bounding box grid.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with points to unify
    box_size : float, default=0.0001
        Size of the bounding box grid cells
    id_column : str, default='location_id'
        Name of the column to store grid cell IDs
        
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
    
    # Calculate grid cell IDs
    result[id_column] = result.apply(
        lambda row: f"{int(row.geometry.x / box_size)}_{int(row.geometry.y / box_size)}",
        axis=1
    )
    
    return result


def analyze_h3_results(
    gdf: gpd.GeoDataFrame,
    id_column: str = 'location_id'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyze H3 unification results.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with H3 unification results
    id_column : str, default='location_id'
        Name of the column containing H3 indices
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        DataFrame with H3 counts and dictionary with statistics
    """
    # Calculate counts per H3 cell
    h3_counts = gdf[id_column].value_counts().reset_index()
    h3_counts.columns = ['h3_index', 'point_count']
    
    # Calculate statistics
    stats = {
        'total_points': len(gdf),
        'unique_h3_cells': len(h3_counts),
        'max_points_per_cell': h3_counts['point_count'].max(),
        'min_points_per_cell': h3_counts['point_count'].min(),
        'avg_points_per_cell': h3_counts['point_count'].mean(),
        'median_points_per_cell': h3_counts['point_count'].median()
    }
    
    return h3_counts, stats


def analyze_dbscan_results(
    gdf: gpd.GeoDataFrame,
    id_column: str = 'location_id'
) -> Tuple[pd.DataFrame, Dict[str, Any], gpd.GeoDataFrame]:
    """
    Analyze DBSCAN unification results.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with DBSCAN unification results
    id_column : str, default='location_id'
        Name of the column containing cluster IDs
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any], gpd.GeoDataFrame]
        DataFrame with cluster counts, dictionary with statistics,
        and GeoDataFrame with cluster centers
    """
    # Calculate counts per cluster
    cluster_counts = gdf[id_column].value_counts().reset_index()
    cluster_counts.columns = ['cluster_id', 'point_count']
    
    # Calculate statistics
    stats = {
        'total_points': len(gdf),
        'unique_clusters': len(cluster_counts),
        'noise_points': len(gdf[gdf[id_column] == -1]),
        'max_points_per_cluster': cluster_counts['point_count'].max(),
        'min_points_per_cluster': cluster_counts['point_count'].min(),
        'avg_points_per_cluster': cluster_counts['point_count'].mean(),
        'median_points_per_cluster': cluster_counts['point_count'].median()
    }
    
    # Calculate cluster centers
    cluster_centers = []
    
    for cluster_id in sorted(gdf[id_column].unique()):
        if cluster_id == -1:
            continue
            
        # Get points in this cluster
        cluster_points = gdf[gdf[id_column] == cluster_id]
        
        # Calculate centroid
        center_lon = cluster_points.geometry.x.mean()
        center_lat = cluster_points.geometry.y.mean()
        
        cluster_centers.append({
            'cluster_id': cluster_id,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'point_count': len(cluster_points)
        })
    
    # Create DataFrame with cluster centers
    if cluster_centers:
        cluster_centers_df = pd.DataFrame(cluster_centers)
        
        # Create geometry column for the centers
        geometries = [Point(row['center_lon'], row['center_lat']) 
                     for _, row in cluster_centers_df.iterrows()]
        cluster_centers_gdf = gpd.GeoDataFrame(
            cluster_centers_df, 
            geometry=geometries,
            crs="EPSG:4326"
        )
    else:
        # Create empty GeoDataFrame if no clusters
        cluster_centers_gdf = gpd.GeoDataFrame(
            columns=['cluster_id', 'center_lat', 'center_lon', 'point_count'],
            geometry=[],
            crs="EPSG:4326"
        )
    
    return cluster_counts, stats, cluster_centers_gdf


def compare_unification_methods(
    gdf: gpd.GeoDataFrame,
    h3_resolution: int = 10,
    dbscan_eps: float = 0.0001,
    dbscan_min_samples: int = 1,
    box_size: float = 0.0001
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different point unification methods.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with points to unify
    h3_resolution : int, default=10
        H3 resolution (0-15, where 15 is highest resolution)
    dbscan_eps : float, default=0.0001
        Maximum distance between points in a DBSCAN cluster
    dbscan_min_samples : int, default=1
        Minimum number of points to form a DBSCAN cluster
    box_size : float, default=0.0001
        Size of the bounding box grid cells
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary with comparison results
    """
    # Apply each unification method
    h3_results = unify_points_h3(gdf, resolution=h3_resolution)
    dbscan_results = unify_points_dbscan(gdf, eps=dbscan_eps, min_samples=dbscan_min_samples)
    bbox_results = unify_points_bounding_box(gdf, box_size=box_size)
    
    # Analyze results
    h3_counts, h3_stats = analyze_h3_results(h3_results)
    dbscan_counts, dbscan_stats, dbscan_centers = analyze_dbscan_results(dbscan_results)
    
    # Calculate counts for bounding box
    bbox_counts = bbox_results['location_id'].value_counts().reset_index()
    bbox_counts.columns = ['bbox_id', 'point_count']
    
    bbox_stats = {
        'total_points': len(bbox_results),
        'unique_boxes': len(bbox_counts),
        'max_points_per_box': bbox_counts['point_count'].max(),
        'min_points_per_box': bbox_counts['point_count'].min(),
        'avg_points_per_box': bbox_counts['point_count'].mean(),
        'median_points_per_box': bbox_counts['point_count'].median()
    }
    
    # Compile comparison results
    comparison = {
        'h3': {
            'results': h3_results,
            'counts': h3_counts,
            'stats': h3_stats
        },
        'dbscan': {
            'results': dbscan_results,
            'counts': dbscan_counts,
            'stats': dbscan_stats,
            'centers': dbscan_centers
        },
        'bounding_box': {
            'results': bbox_results,
            'counts': bbox_counts,
            'stats': bbox_stats
        }
    }
    
    return comparison


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
