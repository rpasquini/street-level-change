"""
Point unification algorithms for Street Level Change Detection.

This module provides algorithms for unifying panorama points using
different clustering methods, including H3, DBSCAN, and bounding box.
"""

import geopandas as gpd
from typing import Union
from sklearn.cluster import DBSCAN
import numpy as np

from src.core.panorama import PanoramaCollection
from src.core.geo_utils import haversine_distance

def unify_points(
    gdf: Union[gpd.GeoDataFrame, PanoramaCollection],
    eps: float = 0.000045, # 5 meters at the equator
    min_samples: int = 1,
) -> gpd.GeoDataFrame:
    """
    Unify points using DBSCAN clustering.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with points to unify
    eps : float, default=0.000045
        Maximum distance between points in a cluster
    min_samples : int, default=1
        Minimum number of points to form a cluster
        
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
    result['cluster_id'] = db.labels_
    
    return result

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
    def haversine_distance(lat1, lon1, lat2, lon2):
        from math import radians, sin, cos, sqrt, atan2
        R = 6371000  # meters
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

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
