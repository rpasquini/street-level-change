"""
Pipeline workflows for Street Level Change Detection.

This module provides workflow orchestration functions that combine
multiple pipeline components to process street-level imagery data.
"""

import os
from typing import Tuple, Optional
import geopandas as gpd
import pandas as pd

from .components import (
    process_region,
    process_panos,
    process_dbscan,
    process_barrios,
    calculate_coverage_area,
    process_heading_fov,
    enrich_panorama_database_from_centroids
)
from src.visualization.static_plotting import plot_date_distribution
from src.api.streetview import download_panorama_image


def run_region(region_slug: str, region_osm: str) -> None:
    """
    Run the complete region processing workflow.
    
    This function orchestrates the entire process of analyzing a region,
    from loading data to calculating coverage metrics.
    
    Parameters
    ----------
    region_slug : str
        Slug identifier for the region (used for directory naming)
    region_osm : str
        OSM region name to process
        
    Returns
    -------
    None
    """
    output_dir = os.path.join("./data", region_slug)
    os.makedirs(output_dir, exist_ok=True)

    # Polygons buffer distance in meters
    buffer_dist = 500
    # Distance between points to point-gridding polygon buffers
    dist_points_grid = 50

    # DBSCAN parameters
    dbscan_eps = 2.5
    dbscan_min_samples = 1

    # Centroid buffer distance in meters
    centroid_buffer = 5

    # Process region and get necessary GeoDataFrames
    region_gdf, mask, renabap_intersected, renabap_buffered = process_region(
        region_osm, buffer_dist, output_dir
    )
    
    # Process panoramas
    regions = pd.concat([renabap_intersected, renabap_buffered])
    panoramas = process_panos(regions, dist_points_grid, output_dir)

    # Visualize date distribution
    # plot_date_distribution(panoramas, output_dir=output_dir)

    # Download a sample panorama image
    # download_panorama_image("pWcvnuI0aGwGObCdcy2avg", output_path=os.path.join(output_dir, "panorama.jpg"))

    # Process DBSCAN clustering
    dbscan_results, centroids = process_dbscan(
        panoramas, dbscan_eps, dbscan_min_samples, output_dir
    )
    
    # Enrich panorama database using DBSCAN centroids
    enriched_panoramas = enrich_panorama_database_from_centroids(
        centroids=centroids,
        renabap_buffered=renabap_buffered,
        data_dir=output_dir,
        max_workers=10,
        verbose=True
    )
    
    # Re-run DBSCAN on enriched panoramas to get final centroids
    print("\nRe-running DBSCAN on enriched panorama database...")
    enriched_dbscan_results, enriched_centroids = process_dbscan(
        enriched_panoramas, 
        eps=dbscan_eps, 
        min_samples=dbscan_min_samples, 
        data_dir=output_dir,
        output_prefix="enriched_"
    )
    
    # Save a comparison of original vs enriched centroids
    print("\nComparing original vs enriched DBSCAN results:")
    print(f"Original clusters: {len(centroids)}")
    print(f"Enriched clusters: {len(enriched_centroids)}")
    print(f"Difference: {len(enriched_centroids) - len(centroids)} clusters")

    centroids = enriched_centroids  # Use enriched centroids for downstream tasks
    dbscan_results = enriched_dbscan_results  # Use enriched results for downstream tasks
    

    # Optional: Evaluate clustering
    # clustering_eval = evaluate_clustering_full(panoramas, 5, 10, 1, output_dir)

    # Join with barrios data
    joined = process_barrios(dbscan_results, renabap_intersected, barrio_buffer_dist=5, data_dir=output_dir)

    # Calculate coverage area metrics
    coverage = calculate_coverage_area(
        polygons=renabap_intersected,
        capture_points=centroids,
        buffer_dist=15,
        data_dir=output_dir,
        buffer_polygons=5
    )
    
    # Process heading and FOV
    heading_fov = process_heading_fov(
        panos=joined,
        control_points=centroids,
        data_dir=output_dir,
        max_distance=100,
        max_fov=120
    )
    
    print(f"Region processing completed for {region_slug}")
    return
