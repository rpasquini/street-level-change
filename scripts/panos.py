#!/usr/bin/env python
"""
Script to download Google Street View panoramas for specified regions.

This script finds regions, creates point grids, and downloads panorama data
for specified areas, saving results to CSV files.
"""

import os
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

# Import from refactored modules
from src.core.geo_utils import find_region, create_point_grid
from src.api.streetview import get_panoramas_for_points
from src.core.panorama import Panorama, PanoramaCollection


def load_mask(wkt_text):
    """
    Load a mask from a WKT string.
    
    Parameters
    ----------
    wkt_text : str
        WKT string representing a geometry
        
    Returns
    -------
    shapely.geometry.BaseGeometry
        Geometry object
    """
    from shapely import wkt

    mask = wkt.loads(wkt_text)
    return mask


def buffer_region_for_osm(gdf, buffer_dist, mask=None):
    """
    Buffer a region for OSM queries, avoiding overlapping buffers.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with geometries to buffer
    buffer_dist : float
        Buffer distance in meters
    mask : shapely.geometry.BaseGeometry, optional
        Mask to filter geometries before buffering
        
    Returns
    -------
    gpd.GeoDataFrame
        Buffered geometries
    """
    buffered = gdf.copy()
    if mask:
        buffered = buffered[buffered.intersects(mask)]
    buffered["geometry"] = (
        buffered.geometry.to_crs(3857).buffer(buffer_dist).to_crs(4326)
    )
    # We have to avoid overlapping buffers before calling OSM
    buffered = gpd.GeoDataFrame(
        [buffered.union_all()], columns=["geometry"], crs=4326
    )
    buffered = buffered.overlay(buffered, how="union")
    return buffered


def loop_polys_get_panos(
    gdf, buffer_dist, dist_points, mask=None, max_workers=10
):
    """
    Generate grid points within buffered polygons and fetch panoramas for each point.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with polygons
    buffer_dist : float
        Buffer distance in meters
    dist_points : float
        Distance between grid points in meters
    mask : shapely.geometry.BaseGeometry, optional
        Mask to filter polygons
    max_workers : int, default=10
        Maximum number of concurrent workers
        
    Returns
    -------
    pd.DataFrame or PanoramaCollection
        Panorama data
    """
    gdf_buffered = gdf.copy()

    # Apply optional mask
    if mask is not None:
        gdf_buffered = gdf_buffered[gdf_buffered.intersects(mask)]

    # Buffer in meters (EPSG:3857), then back to WGS84
    gdf_buffered["geometry"] = (
        gdf_buffered.to_crs(3857).buffer(buffer_dist).to_crs(4326)
    )

    # Union and explode to get disjoint polygons
    gdf_buffered = (
        gpd.GeoDataFrame([gdf_buffered.union_all()], columns=["geometry"])
        .reset_index(drop=True)
        .explode(index_parts=False)
    )

    # Step 1: Generate all grid points
    all_points = []
    for geom in tqdm(gdf_buffered.geometry, desc="Generating Grid Points"):
        points = create_point_grid(geom, dist_points)
        all_points.append(points)

    points_gdf = pd.concat(all_points).reset_index(drop=True)
    
    # Step 2: Get panoramas for all points using the new API module
    panoramas = get_panoramas_for_points(
        points_gdf, 
        max_workers=max_workers,
        verbose=True
    )
    
    return panoramas


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
        # "Partido de La Plata, Buenos Aires, Argentina",
        "Partido de Tres de Febrero, Buenos Aires, Argentina",
        # "Partido de San Isidro, Buenos Aires, Argentina",
    ]
    region_gdf = find_region(regions)
    mask = region_gdf.union_all()

    # Save region_gdf to CSV in data/demo using data_handlers module
    from src.data_handlers.exporters import export_to_csv
    
    region_gdf_path = os.path.join(output_dir, "region_gdf.csv")
    export_to_csv(region_gdf, region_gdf_path)
    print(f"Region GDF saved to {region_gdf_path}")

    # Save renabap intersected with region_gdf
    renabap_intersected = renabap[renabap.intersects(mask)]
    renabap_intersected_path = os.path.join(output_dir, "renabap_intersected.csv")
    export_to_csv(renabap_intersected, renabap_intersected_path)
    print(f"Renabap intersected saved to {renabap_intersected_path}")

    # Save intersected renabap buffers
    renabap_buffered = renabap_intersected.copy()
    renabap_buffered["geometry"] = (
        renabap_buffered.to_crs(3857).buffer(buffer_dist).to_crs(4326)
    )
    renabap_buffered_path = os.path.join(output_dir, "renabap_buffered.csv")
    export_to_csv(renabap_buffered, renabap_buffered_path)
    print(f"Renabap buffered saved to {renabap_buffered_path}")

    # Get panoramas using the refactored function
    panoramas = loop_polys_get_panos(
        renabap,
        buffer_dist=buffer_dist,
        dist_points=dist_points_grid,
        mask=mask,
    )
    
    # Save panoramas to CSV in data/demo
    panos_path = os.path.join(output_dir, "panos.csv")
    
    # If panoramas is a PanoramaCollection, convert to DataFrame before saving
    if isinstance(panoramas, PanoramaCollection):
        panoramas_df = panoramas.to_dataframe()
        export_to_csv(panoramas_df, panos_path)
    else:
        export_to_csv(panoramas, panos_path)
        
    print(f"Panoramas saved to {panos_path}")
