"""
Pipeline components for Street Level Change Detection.

This module provides individual processing components that can be used
in workflows for processing street-level imagery data.
"""

import os
import geopandas as gpd
import pandas as pd
from typing import Tuple, Optional
from tqdm import tqdm

from src.data_handlers.exporters import export_to_csv
from src.processing.point_unification import (
    evaluate_compactness,
    spatial_silhouette_score,
    unify_points,
    compute_cluster_centroids,
)
from src.core.geo_utils import create_point_grid_from_gdf
from src.api.streetview import get_panoramas_for_points, download_panorama_image
from src.core.geo_utils import buffer_region
from src.data_handlers.loaders import load_from_csv, load_panorama_data
from src.core.geo_utils import find_region
from src.visualization.static_plotting import plot_date_distribution


def process_region(region_osm: str, buffer_dist: int, data_dir: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoSeries, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Process a region by finding its boundaries and intersecting with RENABAP data.
    
    Parameters
    ----------
    region_osm : str
        OSM region name to process
    buffer_dist : int
        Buffer distance in meters
    data_dir : str
        Directory to save output files
        
    Returns
    -------
    Tuple[gpd.GeoDataFrame, gpd.GeoSeries, gpd.GeoDataFrame, gpd.GeoDataFrame]
        Region GeoDataFrame, mask, intersected RENABAP data, and buffered RENABAP data
    """
    # Polígonos del RENABAP
    # https://datos.gob.ar/dataset/habitat-registro-nacional-barrios-populares
    renabap = gpd.read_file(
        "https://archivo.habitat.gob.ar/dataset/ssisu/renabap-datos-barrios-geojson"
    )

    region_gdf_path = os.path.join(data_dir, "region_gdf.csv")
    renabap_intersected_path = os.path.join(
        data_dir, "renabap_intersected.csv"
    )
    renabap_buffered_path = os.path.join(data_dir, "renabap_buffered.csv")

    if not os.path.exists(region_gdf_path):
        region_gdf = find_region(region_osm)
        mask = region_gdf.union_all()
        export_to_csv(region_gdf, region_gdf_path)
    else:
        region_gdf = load_from_csv(region_gdf_path)
        mask = region_gdf.union_all()

    if not os.path.exists(renabap_intersected_path):
        # Save renabap intersected with region_gdf
        renabap_intersected = renabap[renabap.intersects(mask)]
        export_to_csv(renabap_intersected, renabap_intersected_path)
    else:
        renabap_intersected = load_from_csv(renabap_intersected_path)

    if not os.path.exists(renabap_buffered_path):
        # Save intersected renabap buffers
        renabap_buffered = buffer_region(renabap_intersected, buffer_dist)
        renabap_buffered = renabap_buffered.overlay(
            renabap_intersected, how="difference"
        )
        export_to_csv(renabap_buffered, renabap_buffered_path)
    else:
        renabap_buffered = load_from_csv(renabap_buffered_path)

    return region_gdf, mask, renabap_intersected, renabap_buffered


def process_panos(
    renabap_buffered: gpd.GeoDataFrame, dist_points_grid: int, data_dir: str
) -> gpd.GeoDataFrame:
    """
    Process panoramas by creating a point grid and fetching panorama data.
    
    Parameters
    ----------
    renabap_buffered : gpd.GeoDataFrame
        Buffered RENABAP data
    dist_points_grid : int
        Distance between points in the grid
    data_dir : str
        Directory to save output files
        
    Returns
    -------
    gpd.GeoDataFrame
        Panorama data as a GeoDataFrame
    """
    panos_path = os.path.join(data_dir, "panos.csv")
    if not os.path.exists(panos_path):
        points_gdf = create_point_grid_from_gdf(
            renabap_buffered, dist_points_grid
        )

        panoramas = get_panoramas_for_points(points_gdf, verbose=True)

        panoramas = panoramas.clean(renabap_buffered)

        export_to_csv(panoramas, panos_path)
        print(f"Panoramas saved to {panos_path}")
    else:
        panoramas = load_panorama_data(panos_path)

    return panoramas


def process_dbscan(
    panoramas: gpd.GeoDataFrame,
    dbscan_eps: float,
    dbscan_min_samples: int,
    data_dir: str,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Process panorama data using DBSCAN clustering.
    
    Parameters
    ----------
    panoramas : gpd.GeoDataFrame
        Panorama data
    dbscan_eps : float
        DBSCAN epsilon parameter
    dbscan_min_samples : int
        DBSCAN minimum samples parameter
    data_dir : str
        Directory to save output files
        
    Returns
    -------
    Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        DBSCAN results and centroids
    """
    dbscan_output_path = os.path.join(data_dir, "dbscan_results.csv")
    dbscan_centroids_path = os.path.join(data_dir, "dbscan_centroids.csv")

    if not os.path.exists(dbscan_output_path):
        # Apply DBSCAN unification
        print(
            f"\nApplying DBSCAN unification with eps={dbscan_eps}, min_samples={dbscan_min_samples}"
        )
        dbscan_results = unify_points(
            panoramas, eps=dbscan_eps, min_samples=dbscan_min_samples
        )

        # Save results using data_handlers exporters
        export_to_csv(dbscan_results, dbscan_output_path)
        print(f"DBSCAN results saved to {dbscan_output_path}")
    else:
        dbscan_results = load_from_csv(dbscan_output_path)

    if not os.path.exists(dbscan_centroids_path):
        # Compute centroids
        print("\nComputing centroids for DBSCAN results")
        centroids = compute_cluster_centroids(dbscan_results)

        # Save centroids
        export_to_csv(centroids, dbscan_centroids_path)
        print(f"Centroids saved to {dbscan_centroids_path}")
    else:
        centroids = load_from_csv(dbscan_centroids_path)

    return dbscan_results, centroids


def join_barrios(
    panos: gpd.GeoDataFrame,
    renabap_intersected: gpd.GeoDataFrame,
    data_dir: str,
) -> gpd.GeoDataFrame:
    """
    Join panorama data with RENABAP barrio data.
    
    Parameters
    ----------
    panos : gpd.GeoDataFrame
        Panorama data
    renabap_intersected : gpd.GeoDataFrame
        Intersected RENABAP data
    data_dir : str
        Directory to save output files
        
    Returns
    -------
    gpd.GeoDataFrame
        Joined data
    """
    joined_path = os.path.join(data_dir, "joined.csv")
    if not os.path.exists(joined_path):
        # Spatial join with renabap
        joined = gpd.sjoin(
            panos, renabap_intersected, how="left", predicate="within"
        )
        # Save joined data
        export_to_csv(joined, joined_path)
    else:
        joined = load_from_csv(joined_path)

    return joined


def evaluate_clustering(
    gdf: gpd.GeoDataFrame,
    start: int,
    end: int,
    step: int,
    data_dir: str,
) -> pd.DataFrame:
    """
    Evaluate clustering with different parameters.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame
    start : int
        Start value for epsilon
    end : int
        End value for epsilon
    step : int
        Step size for epsilon
    data_dir : str
        Directory to save output files
        
    Returns
    -------
    pd.DataFrame
        Evaluation results
    """
    print(f"\nEvaluating clustering with eps from {start} to {end} with step {step}")
    output = []
    for eps in range(start, end + 1, step):
        dbscan_results = unify_points(gdf, eps=eps)
        compactness = evaluate_compactness(dbscan_results)
        compactness['eps'] = str(eps)

        output.append(compactness)
    
    output = pd.concat(output)
    output.to_csv(os.path.join(data_dir, "clustering_evaluation.csv"))

    return output 


def evaluate_clustering_full(
    gdf: gpd.GeoDataFrame,
    start: int,
    end: int,
    step: int,
    data_dir: str,
) -> pd.DataFrame:
    """
    Perform full clustering evaluation with silhouette scores.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame
    start : int
        Start value for epsilon
    end : int
        End value for epsilon
    step : int
        Step size for epsilon
    data_dir : str
        Directory to save output files
        
    Returns
    -------
    pd.DataFrame
        Full evaluation results
    """
    print(f"\nEvaluating clustering with eps from {start} to {end} with step {step}")
    from tqdm import tqdm
    output = []
    for eps in tqdm(range(start, end + 1, step), total=(end - start) // step):
        dbscan_results = unify_points(gdf, eps=eps)
        scores = spatial_silhouette_score(dbscan_results)
        scores['eps'] = str(eps)

        compactness = evaluate_compactness(dbscan_results)
        compactness['eps'] = str(eps)

        final = compactness.set_index(
            ["cluster_id", "eps"]
        ).join(scores.set_index(["cluster_id", "eps"]))
        output.append(final)
    
    output = pd.concat(output)
    output.to_csv(os.path.join(data_dir, "clustering_evaluation.csv"))

    return output


def calculate_coverage_area(
    centroid_buffer: int,
    centroids: gpd.GeoDataFrame,
    renabap_intersected: gpd.GeoDataFrame,
    renabap_buffered: gpd.GeoDataFrame,
    data_dir: str,
) -> pd.DataFrame:
    """
    Calculate coverage area metrics.
    
    Parameters
    ----------
    centroid_buffer : int
        Buffer distance for centroids
    centroids : gpd.GeoDataFrame
        Centroids data
    renabap_intersected : gpd.GeoDataFrame
        Intersected RENABAP data
    renabap_buffered : gpd.GeoDataFrame
        Buffered RENABAP data
    data_dir : str
        Directory to save output files
        
    Returns
    -------
    pd.DataFrame
        Coverage area metrics
    """
    coverage_path = os.path.join(data_dir, "coverage_areas.csv")

    def calculate_area_per_region(
        regions: gpd.GeoDataFrame, buffered_centroids: gpd.GeoDataFrame
    ):
        """
        Calculate area coverage per region.
        
        Parameters
        ----------
        regions : gpd.GeoDataFrame
            Region data
        buffered_centroids : gpd.GeoDataFrame
            Buffered centroids
            
        Returns
        -------
        gpd.GeoDataFrame
            Regions with coverage area metrics
        """
        regions = regions.to_crs(3857)
        buffered_centroids = buffered_centroids.to_crs(3857)
        regions["original_region_area"] = regions.area
        regions = regions.clip(buffered_centroids.to_crs(3857).union_all())
        regions["coverage_area"] = regions.area / regions["original_region_area"]
        return regions.to_crs(4326)

    if not os.path.exists(coverage_path):
        centroids["geometry"] = (
            centroids.to_crs(3857).buffer(centroid_buffer).to_crs(4326)
        )

        barrio_coverage = calculate_area_per_region(
            renabap_intersected, centroids
        )
        buffered_barrio_coverage = calculate_area_per_region(
            renabap_buffered, centroids
        ).rename({"coverage_area": "buffered_coverage_area"}, axis=1)

        coverage = (
            barrio_coverage.set_index("id_renabap").coverage_area.to_frame()
            .join(buffered_barrio_coverage.set_index("id_renabap").buffered_coverage_area.to_frame())
            .reset_index()
        )
        coverage.to_csv(coverage_path, index=False)
    else:
        coverage = load_from_csv(coverage_path)

    return coverage
