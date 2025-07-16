"""
Point unification algorithms for Street Level Change Detection.

This module provides algorithms for unifying panorama points using
different clustering methods, including H3, DBSCAN, and bounding box.
"""

import geopandas as gpd
import pandas as pd
from typing import Union
from sklearn.cluster import DBSCAN
import numpy as np
from shapely.geometry import Point
from tqdm import tqdm
from src.core.panorama import PanoramaCollection
from src.core.geo_utils import haversine_distance

def unify_points(
    gdf: Union[gpd.GeoDataFrame, 'PanoramaCollection'],
    eps: float = 5,  # in meters
    min_samples: int = 1,
    projected_crs: str = "EPSG:3857"  # can be set to UTM if needed
) -> gpd.GeoDataFrame:
    """
    Unify points using DBSCAN clustering with distance in meters.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame or PanoramaCollection
        GeoDataFrame with points to unify
    eps : float, default=5
        Maximum distance between points in a cluster, in meters
    min_samples : int, default=1
        Minimum number of points to form a cluster
    projected_crs : str, default="EPSG:3857"
        Projected CRS used to measure distance in meters
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with unified points (cluster_id column)
    """
    if isinstance(gdf, PanoramaCollection):
        gdf = gdf.to_geodataframe()

    # Ensure CRS is set to EPSG:4326 before projecting
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    
    # Project to a metric CRS (for distance-based clustering)
    gdf_proj = gdf.to_crs(projected_crs)

    # Extract coordinates in meters
    coords = np.array([(p.x, p.y) for p in gdf_proj.geometry])

    # Apply DBSCAN in projected space (meters)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    gdf["cluster_id"] = db.labels_

    return gdf

def compute_cluster_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Aggregates points in each cluster and computes the centroid.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with Point geometries and a 'cluster_id' column.
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with one row per cluster containing:
        - cluster_id
        - point_count: number of points in the cluster
        - geometry: the centroid (as Point) of all points in the cluster
    """
    if 'cluster_id' not in gdf.columns:
        raise ValueError("GeoDataFrame must contain a 'cluster_id' column.")
    if gdf.geometry.geom_type.unique().tolist() != ['Point']:
        raise ValueError("All geometries must be Points.")

    # Compute centroids by group
    grouped = gdf.groupby("cluster_id")

    centroids = grouped.geometry.apply(lambda geoms: geoms.union_all().centroid)
    counts = grouped.size()

    # Combine into a GeoDataFrame
    result = gpd.GeoDataFrame({
        'cluster_id': centroids.index,
        'point_count': counts.values,
        'geometry': centroids.values
    }, geometry='geometry', crs=gdf.crs)

    return result


def evaluate_compactness(
    gdf: gpd.GeoDataFrame,
    cluster_col: str = "cluster_id"
) -> pd.DataFrame:
    """
    Evaluate clustering compactness by computing the average and maximum
    haversine distance from each point to its cluster centroid.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with Point geometries and a cluster column.
    cluster_col : str, default="cluster_id"
        Name of the column identifying cluster membership.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per cluster:
        - cluster_id
        - point_count
        - avg_distance (meters)
        - max_distance (meters)
    """
    results = []

    for cluster_id, group in gdf.groupby(cluster_col):
        if len(group) <= 1:
            continue  # skip singletons or noise

        lats = group.geometry.y.values
        lons = group.geometry.x.values

        # Compute centroid of cluster
        centroid_lat = lats.mean()
        centroid_lon = lons.mean()

        # Compute distances from each point to centroid
        distances = [
            haversine_distance(lat, lon, centroid_lat, centroid_lon)
            for lat, lon in zip(lats, lons)
        ]

        results.append({
            "cluster_id": cluster_id,
            "point_count": len(group),
            "avg_distance": sum(distances) / len(distances),
            "max_distance": max(distances)
        })

    return pd.DataFrame(results)

def _process_point_silhouette(point_data):
    """
    Process silhouette score for a single point (used for parallel processing).
    
    Parameters
    ----------
    point_data : tuple
        Tuple containing (index, lat, lon, cluster_id, centroids_dict, cluster_col)
        
    Returns
    -------
    dict
        Dictionary with silhouette score results for this point
    """
    idx, lat, lon, cluster_id, centroids, cluster_col = point_data
    
    # a: distance to own cluster centroid
    own_centroid = centroids[cluster_id]
    a = haversine_distance(lat, lon, own_centroid.y, own_centroid.x)
    
    # b: distance to nearest other cluster centroid
    b = float("inf")
    for other_id, centroid in centroids.items():
        if other_id == cluster_id:
            continue
        d = haversine_distance(lat, lon, centroid.y, centroid.x)
        if d < b:
            b = d
    
    # Calculate silhouette score
    s = (b - a) / max(a, b) if max(a, b) > 0 else 0
    
    return {
        "index": idx,
        cluster_col: cluster_id,
        "a_distance": a,
        "b_distance": b,
        "silhouette_score": s
    }


def spatial_silhouette_score(
    gdf: gpd.GeoDataFrame,
    cluster_col: str = "cluster_id",
    max_workers: int = None
) -> pd.DataFrame:
    """
    Computes a spatial silhouette-like score for each point and cluster using parallel processing.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with Point geometries and a 'cluster_id' column.
    cluster_col : str, default="cluster_id"
        Column name that contains the cluster ID.
    max_workers : int, default=None
        Maximum number of worker processes to use for parallel processing.
        If None, it will use the number of processors on the machine.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - pano_id (or index)
        - cluster_id
        - a (intra-cluster distance to own centroid)
        - b (nearest other-cluster centroid distance)
        - silhouette_score
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    if 'geometry' not in gdf.columns:
        raise ValueError("GeoDataFrame must contain 'geometry' column with Point geometries.")
    if cluster_col not in gdf.columns:
        raise ValueError(f"GeoDataFrame must contain '{cluster_col}' column.")

    gdf = gdf.copy()

    # Compute centroids of each cluster
    centroids = gdf.groupby(cluster_col).geometry.apply(
        lambda geoms: Point(geoms.x.mean(), geoms.y.mean())
    ).to_dict()
    
    # Prepare data for parallel processing
    point_data = [
        (idx, row.geometry.y, row.geometry.x, row[cluster_col], centroids, cluster_col)
        for idx, row in gdf.iterrows()
    ]
    
    results = []
    
    # Process points in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_point_silhouette, data) for data in point_data]
        
        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing silhouette scores"):
            results.append(future.result())
    
    return pd.DataFrame(results).set_index("index")