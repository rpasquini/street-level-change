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
    Unifies panorama points using DBSCAN clustering with distance in meters.
    
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
    np.random.seed(42)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    gdf["cluster_id"] = db.labels_
    np.random.seed(None)

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


def evaluate_dbscan_clusters(clusters_gdf, points_gdf, disable_tqdm=False):
    """
    Evaluate DBSCAN clustering results with simple, interpretable metrics.
    
    Parameters
    ----------
    clusters_gdf : GeoDataFrame
        DataFrame with cluster centroids. Must have columns:
        - 'cluster_id'
        - 'geometry' (Point)
        
    points_gdf : GeoDataFrame
        DataFrame with points assigned to clusters. Must have columns:
        - 'cluster_id' (DBSCAN labels, -1 = noise)
        - 'geometry' (Point)
        
    Returns
    -------
    metrics : dict
        Dictionary with evaluation metrics.
    """

    # Transform to EPSG:3857 to get metrics in meters
    clusters_gdf = clusters_gdf.to_crs(3857)
    points_gdf = points_gdf.to_crs(3857)
    
    # Exclude noise for cluster-based calculations
    clustered_points = points_gdf[points_gdf["cluster_id"] != -1]
    valid_clusters = clusters_gdf[clusters_gdf["cluster_id"] != -1]
    
    # Number of clusters
    n_clusters = valid_clusters["cluster_id"].nunique()
    
    # Noise ratio
    noise_ratio = (points_gdf["cluster_id"] == -1).mean()
    
    # Cluster size distribution
    cluster_sizes = clustered_points.groupby("cluster_id").size()
    avg_cluster_size = cluster_sizes.mean()
    cluster_size_stats = cluster_sizes.describe().to_dict()
    
    # Within-cluster average distance
    within_distances = []
    for cid, group in tqdm(clustered_points.groupby("cluster_id"), total=n_clusters, disable=disable_tqdm):
        centroid = valid_clusters.loc[valid_clusters["cluster_id"] == cid, "geometry"].values[0]
        dists = group["geometry"].apply(lambda g: g.distance(centroid)).values
        within_distances.append(np.mean(dists))
    avg_within_distance = np.mean(within_distances) if within_distances else np.nan
    
    # Between-cluster distances (pairwise between centroids)
    between_distances = []
    centroids = valid_clusters["geometry"].values
    for i in tqdm(range(len(centroids)), total=len(centroids), disable=disable_tqdm):
        for j in range(i+1, len(centroids)):
            between_distances.append(centroids[i].distance(centroids[j]))
    avg_between_distance = np.mean(between_distances) if between_distances else np.nan
    
    # Separation / Cohesion ratio
    sep_coh_ratio = (
        avg_between_distance / avg_within_distance
        if avg_within_distance and not np.isnan(avg_within_distance) else np.nan
    )
    
    return {
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "avg_cluster_size": avg_cluster_size,
        "cluster_size_stats": cluster_size_stats,
        "avg_within_distance": avg_within_distance,
        "avg_between_distance": avg_between_distance,
        "sep_coh_ratio": sep_coh_ratio
    }


def run_dbscan_evaluations(points_gdf, eps_values, min_samples_values, disable_tqdm=True):
    """
    Run DBSCAN for a grid of eps and min_samples values,
    evaluate results with evaluate_dbscan_clusters,
    and return a DataFrame with all metrics.
    
    Parameters
    ----------
    points_gdf : GeoDataFrame
        Input points with geometry (no cluster_id column yet).
    eps_values : list
        List of eps values to try.
    min_samples_values : list
        List of min_samples values to try.
    metric : str
        Distance metric for DBSCAN (default = 'euclidean').
    
    Returns
    -------
    results_df : DataFrame
        Evaluation results for each (eps, min_samples).
    """
    
    results = []

    for eps in tqdm(eps_values):
        for ms in tqdm(min_samples_values):
            
            dbscan_results, centroids = gen_clusters(points_gdf, eps=eps, min_samples=ms)
            
            # Evaluate
            metrics = evaluate_dbscan_clusters(centroids, points_gdf, disable_tqdm=disable_tqdm)
            metrics["eps"] = eps
            metrics["min_samples"] = ms
            
            results.append(metrics)
    
    return pd.DataFrame(results)
