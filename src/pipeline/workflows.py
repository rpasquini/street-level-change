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
    get_panos,
    prepare_region,
    process_dbscan,
    enrich_barrios,
    calculate_coverage_area,
    calculate_heading_fov,
    get_metadata_dates
)

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
    first_dbscan_eps = 5
    final_dbscan_eps = 2.5
    dbscan_min_samples = 2

    # Centroid buffer distance in meters
    centroid_buffer = 5

    # Process region and get necessary GeoDataFrames
    regions, renabap_intersected, renabap_buffered = prepare_region(
        region_osm, buffer_dist, output_dir
    )
    
    # Process panoramas
    panoramas = get_panos(regions, dist_points_grid, output_dir, dbscan_eps=first_dbscan_eps)

    # Re-run DBSCAN on enriched panoramas to get final centroids
    dbscan_results, centroids = process_dbscan(
        panoramas, 
        eps=final_dbscan_eps, 
        min_samples=dbscan_min_samples, 
        data_dir=output_dir,
        output_prefix="enriched_"
    )
    
    # Join with barrios data
    panoramas = enrich_barrios(dbscan_results, renabap_intersected, barrio_buffer_dist=5, data_dir=output_dir)

    # Calculate coverage area metrics
    coverage = calculate_coverage_area(
        polygons=renabap_intersected,
        capture_points=centroids,
        buffer_dist=15,
        data_dir=output_dir,
        buffer_polygons=5
    )
    
    # Process heading and FOV
    heading_fov = calculate_heading_fov(
        panos=panoramas,
        control_points=centroids,
        data_dir=output_dir,
        max_distance=100,
        max_fov=120
    )
    
    api_key = os.getenv("GOOGLE_STREET_VIEW_API_KEY")
    panos_w_dates = get_metadata_dates(
        panoramas,
        api_key,
        data_dir=output_dir
    )
    
    print(f"Region processing completed for {region_slug}")
    return
