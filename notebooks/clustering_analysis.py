#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys

# Add src to path to import our modules
sys.path.append("..")

from src.core.point_unification import (
    unify_points,
    compute_cluster_centroids,
)

from tqdm import tqdm

import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
from shapely.ops import nearest_points


def gen_clusters(panoramas, eps, min_samples=1):
    print(
        f"Applying DBSCAN clustering with eps={eps}, min_samples={min_samples}"
    )
    # Apply DBSCAN clustering
    dbscan_results = unify_points(panoramas, eps=eps, min_samples=min_samples)

    # Compute centroids
    # print("Computing cluster centroids")
    centroids = compute_cluster_centroids(dbscan_results)
    return dbscan_results, centroids


# def evaluate_dbscan_clusters(clusters_gdf, points_gdf, disable_tqdm=False):
#     """
#     Evaluate DBSCAN clustering results with simple, interpretable metrics.
#
#     Parameters
#     ----------
#     clusters_gdf : GeoDataFrame
#         DataFrame with cluster centroids. Must have columns:
#         - 'cluster_id'
#         - 'geometry' (Point)
#
#     points_gdf : GeoDataFrame
#         DataFrame with points assigned to clusters. Must have columns:
#         - 'cluster_id' (DBSCAN labels, -1 = noise)
#         - 'geometry' (Point)
#
#     Returns
#     -------
#     metrics : dict
#         Dictionary with evaluation metrics.
#     """
#
#     # Transform to EPSG:3857 to get metrics in meters
#     clusters_gdf = clusters_gdf.to_crs(3857)
#     points_gdf = points_gdf.to_crs(3857)
#
#     # Exclude noise for cluster-based calculations
#     clustered_points = points_gdf[points_gdf["cluster_id"] != -1]
#     valid_clusters = clusters_gdf[clusters_gdf["cluster_id"] != -1]
#
#     # Number of clusters
#     n_clusters = valid_clusters["cluster_id"].nunique()
#
#     # Noise ratio
#     noise_ratio = (points_gdf["cluster_id"] == -1).mean()
#
#     # Cluster size distribution
#     cluster_sizes = clustered_points.groupby("cluster_id").size()
#     avg_cluster_size = cluster_sizes.mean()
#     cluster_size_stats = cluster_sizes.describe().to_dict()
#
#     # Within-cluster average distance
#     within_distances = []
#     for cid, group in tqdm(
#         clustered_points.groupby("cluster_id"),
#         total=n_clusters,
#         disable=disable_tqdm,
#     ):
#         centroid = valid_clusters.loc[
#             valid_clusters["cluster_id"] == cid, "geometry"
#         ].values[0]
#         dists = group["geometry"].apply(lambda g: g.distance(centroid)).values
#         within_distances.append(np.mean(dists))
#     avg_within_distance = (
#         np.mean(within_distances) if within_distances else np.nan
#     )
#
#     # Between-cluster distances (pairwise between centroids)
#     between_distances = []
#     centroids = valid_clusters["geometry"].values
#     for i in tqdm(
#         range(len(centroids)), total=len(centroids), disable=disable_tqdm
#     ):
#         for j in range(i + 1, len(centroids)):
#             between_distances.append(centroids[i].distance(centroids[j]))
#     avg_between_distance = (
#         np.mean(between_distances) if between_distances else np.nan
#     )
#
#     # Separation / Cohesion ratio
#     sep_coh_ratio = (
#         avg_between_distance / avg_within_distance
#         if avg_within_distance and not np.isnan(avg_within_distance)
#         else np.nan
#     )
#
#     return {
#         "n_clusters": n_clusters,
#         "noise_ratio": noise_ratio,
#         "avg_cluster_size": avg_cluster_size,
#         "cluster_size_stats": cluster_size_stats,
#         "avg_within_distance": avg_within_distance,
#         "avg_between_distance": avg_between_distance,
#         "sep_coh_ratio": sep_coh_ratio,
#     }
#
#
def run_dbscan(points_gdf, eps_values, min_samples_values, disable_tqdm=True):
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

            dbscan_results, centroids = gen_clusters(
                points_gdf, eps=eps, min_samples=ms
            )

            # Evaluate
            metrics = evaluate_dbscan_clusters(
                centroids, points_gdf, disable_tqdm=disable_tqdm
            )
            print()
            print(f"EPS: {eps}MS: {ms}")
            print(metrics)
            print()
            metrics["eps"] = eps
            metrics["min_samples"] = ms

            results.append(metrics)

    return pd.DataFrame(results)


import numpy as np
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import MultiPoint


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

    # Cluster-level measures
    within_distances = []
    max_distances = []
    std_distances = []
    convex_hull_areas = []
    densities = []
    eccentricities = []

    for cid, group in tqdm(
        clustered_points.groupby("cluster_id"),
        total=n_clusters,
        disable=disable_tqdm,
    ):
        centroid = valid_clusters.loc[
            valid_clusters["cluster_id"] == cid, "geometry"
        ].values[0]
        dists = group["geometry"].apply(lambda g: g.distance(centroid)).values

        # Basic distance measures
        within_distances.append(np.mean(dists))
        max_distances.append(np.max(dists))
        std_distances.append(np.std(dists))

        # Convex hull compactness
        if len(group) > 2:  # at least 3 points to form a polygon
            hull = MultiPoint(group.geometry.values).convex_hull
            area = hull.area
            convex_hull_areas.append(area)
            densities.append(len(group) / area if area > 0 else np.nan)

            # Eccentricity = ratio longest_axis / shortest_axis
            min_rect = hull.minimum_rotated_rectangle
            coords = list(min_rect.exterior.coords)
            edge_lengths = [
                np.linalg.norm(np.array(coords[i]) - np.array(coords[i + 1]))
                for i in range(len(coords) - 1)
            ]
            major = max(edge_lengths)
            minor = min(edge_lengths) if min(edge_lengths) > 0 else np.nan
            eccentricities.append(
                major / minor if minor and minor > 0 else np.nan
            )
        else:
            convex_hull_areas.append(np.nan)
            densities.append(np.nan)
            eccentricities.append(np.nan)

    # Aggregate cluster-level measures
    avg_within_distance = (
        np.nanmean(within_distances) if within_distances else np.nan
    )
    avg_max_distance = np.nanmean(max_distances) if max_distances else np.nan
    avg_std_distance = np.nanmean(std_distances) if std_distances else np.nan
    avg_convex_area = (
        np.nanmean(convex_hull_areas) if convex_hull_areas else np.nan
    )
    avg_density = np.nanmean(densities) if densities else np.nan
    avg_eccentricity = np.nanmean(eccentricities) if eccentricities else np.nan

    # Between-cluster distances (pairwise between centroids)
    between_distances = []
    centroids = valid_clusters["geometry"].values
    for i in tqdm(
        range(len(centroids)), total=len(centroids), disable=disable_tqdm
    ):
        for j in range(i + 1, len(centroids)):
            between_distances.append(centroids[i].distance(centroids[j]))
    avg_between_distance = (
        np.mean(between_distances) if between_distances else np.nan
    )

    # Separation / Cohesion ratio
    sep_coh_ratio = (
        avg_between_distance / avg_within_distance
        if avg_within_distance and not np.isnan(avg_within_distance)
        else np.nan
    )

    return {
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "avg_cluster_size": avg_cluster_size,
        "cluster_size_stats": cluster_size_stats,
        "avg_within_distance": avg_within_distance,
        "avg_between_distance": avg_between_distance,
        "sep_coh_ratio": sep_coh_ratio,
        # Extra measures
        "avg_max_distance": avg_max_distance,
        "avg_std_distance": avg_std_distance,
        "avg_convex_area": avg_convex_area,
        "avg_density": avg_density,
        "avg_eccentricity": avg_eccentricity,
    }


def df_to_gdf(df):
    gdf = df.copy()
    gdf.loc[:, "geometry"] = gdf.geometry.apply(wkt.loads)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=4326)
    return gdf


# In[6]:


extent = wkt.loads(
    "POLYGON((-58.58198 -34.582224, -58.515231 -34.582224, -58.515231 -34.635357, -58.58198 -34.635357, -58.58198 -34.582224))"
)


# In[7]:


panos = df_to_gdf(pd.read_csv("../data/tresdefebrero/panos_enriched.csv"))

rbp = df_to_gdf(pd.read_csv("../data/tresdefebrero/renabap_intersected.csv"))

panos = gpd.sjoin_nearest(
    panos.to_crs(3857),
    rbp[["id_renabap", "geometry"]].to_crs(3857),
    how="left",
    distance_col="distance",
).to_crs(4326)
panos = panos.rename(columns={"id_renabap": "closest_barrio"}).drop(
    columns=["index_right"]
)


# In[8]:


panos = panos[panos.intersects(extent)]

eps_range = [5, 2.5, 1]  # in meters
min_samples_range = [2, 3, 4]

results_df = run_dbscan(
    panos, eps_range, min_samples_range, disable_tqdm=False
)

print(results_df.head())


# In[ ]:


results_df.to_csv("results.csv", index=None)


# In[ ]:


# In[ ]:
