#!/usr/bin/env python
"""
Script to download Google Street View panoramas for specified regions.

This script finds regions, creates point grids, and downloads panorama data
for specified areas, saving results to CSV files.
"""

import os
import geopandas as gpd
from src.core.geo_utils import find_region, buffer_region, create_point_grid_from_gdf
from src.api.streetview import get_panoramas_for_points
from src.data_handlers.exporters import export_to_csv



if __name__ == "__main__":
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "panos")
    os.makedirs(output_dir, exist_ok=True)
    
    # Polygons buffer distance in meters
    buffer_dist = 500
    # Distance between points to point-gridding polygon buffers
    dist_points_grid = 50

    # Polígonos del RENABAP
    # https://datos.gob.ar/dataset/habitat-registro-nacional-barrios-populares
    renabap = gpd.read_file(
        "https://archivo.habitat.gob.ar/dataset/ssisu/renabap-datos-barrios-geojson"
    )

    regions = [
        "Partido de La Plata, Buenos Aires, Argentina",
        "Partido de Tres de Febrero, Buenos Aires, Argentina",
        "Partido de San Isidro, Buenos Aires, Argentina"
    ]
    region_gdf = find_region(regions)
    mask = region_gdf.union_all()

    
    region_gdf_path = os.path.join(output_dir, "region_gdf.csv")
    export_to_csv(region_gdf, region_gdf_path)

    # Save renabap intersected with region_gdf
    renabap_intersected = renabap[renabap.intersects(mask)]
    renabap_intersected_path = os.path.join(output_dir, "renabap_intersected.csv")
    export_to_csv(renabap_intersected, renabap_intersected_path)

    # Save intersected renabap buffers
    renabap_buffered = buffer_region(renabap_intersected, buffer_dist)
    renabap_buffered_path = os.path.join(output_dir, "renabap_buffered.csv")
    export_to_csv(renabap_buffered, renabap_buffered_path)

    points_gdf = create_point_grid_from_gdf(renabap_buffered, dist_points_grid)
    
    panoramas = get_panoramas_for_points(
        points_gdf, 
        max_workers=max_workers,
        verbose=True
    )
    
    panos_path = os.path.join(output_dir, "panos.csv")
    
    export_to_csv(panoramas, panos_path)
        
    print(f"Panoramas saved to {panos_path}")
