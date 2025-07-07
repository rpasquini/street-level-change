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

def spatial_silhouette_score(
    gdf: gpd.GeoDataFrame,
    cluster_col: str = "cluster_id"
) -> pd.DataFrame:
    """
    Computes a spatial silhouette-like score for each point and cluster.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with Point geometries and a 'cluster_id' column.
    cluster_col : str, default="cluster_id"
        Column name that contains the cluster ID.

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
    if 'geometry' not in gdf.columns:
        raise ValueError("GeoDataFrame must contain 'geometry' column with Point geometries.")
    if cluster_col not in gdf.columns:
        raise ValueError(f"GeoDataFrame must contain '{cluster_col}' column.")

    gdf = gdf.copy()

    # Compute centroids of each cluster
    centroids = gdf.groupby(cluster_col).geometry.apply(
        lambda geoms: Point(geoms.x.mean(), geoms.y.mean())
    ).to_dict()

    results = []

    for idx, row in gdf.iterrows():
        cluster_id = row[cluster_col]
        lat = row.geometry.y
        lon = row.geometry.x

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

        s = (b - a) / max(a, b) if max(a, b) > 0 else 0

        results.append({
            "index": idx,
            cluster_col: cluster_id,
            "a_distance": a,
            "b_distance": b,
            "silhouette_score": s
        })

    return pd.DataFrame(results).set_index("index")