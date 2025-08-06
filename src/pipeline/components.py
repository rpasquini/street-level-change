"""
Pipeline components for Street Level Change Detection.

This module provides individual processing components that can be used
in workflows for processing street-level imagery data.
"""

import os
import geopandas as gpd
import pandas as pd
from typing import Tuple, Union
from tqdm import tqdm

from src.data_handlers.exporters import export_to_csv
from src.core.point_unification import (
    evaluate_compactness,
    spatial_silhouette_score,
    unify_points,
    compute_cluster_centroids,
)

from src.core.heading_fov import get_angles
from src.core.geo_utils import create_point_grid_from_gdf
from src.api.streetview import get_panoramas_for_points
from src.core.geo_utils import buffer_region
from src.data_handlers.loaders import load_from_csv, load_panorama_data
from src.core.geo_utils import find_region, get_roads_from_gdf, get_roads_from_polygon


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
        "https://archivo.infraestructura.gob.ar/dataset/ssisu/renabap-datos-barrios-geojson"
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
        renabap_buffered = buffer_region(renabap_intersected, buffer_dist, exclude_original=True)
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


def process_barrios(
    panos: gpd.GeoDataFrame,
    renabap_intersected: gpd.GeoDataFrame,
    barrio_buffer_dist: int,
    data_dir: str,
) -> gpd.GeoDataFrame:
    """
    Join panorama data with RENABAP barrio data.
    Creates two dummies for each panorama:
        - inside: 1 if the panorama is inside the barrio, 0 otherwise
        - close: 1 if the panorama is inside a small buffered barrio, 0 otherwise
    
    Parameters
    ----------
    panos : gpd.GeoDataFrame
        Panorama data
    renabap_intersected : gpd.GeoDataFrame
        Intersected RENABAP data
    barrio_buffer_dist : int
        Buffer distance for barrios
    data_dir : str
        Directory to save output files
        
    Returns
    -------
    gpd.GeoDataFrame
        Joined data
    """
    joined_path = os.path.join(data_dir, "joined.csv")
    if not os.path.exists(joined_path):
        renabap_buffered = buffer_region(renabap_intersected, barrio_buffer_dist, exclude_original=True)
        # Mark pano points as inside or inside_buffered
        panos['inside'] = panos.within(renabap_intersected.union_all()).astype(int)
        panos['close'] = panos.within(renabap_buffered.union_all()).astype(int)
        panos = gpd.sjoin_nearest(
            panos.to_crs(3857), renabap_intersected[['id_renabap','geometry']].to_crs(3857),
            how="left", 
            distance_col="distance"
        ).to_crs(4326)

        panos["inside_close"] = panos['inside'] + panos["close"]

        panos = panos.rename(
            columns={"id_renabap": "closest_barrio"}).drop(columns=["index_right"]
            )

        # Save joined data
        export_to_csv(panos, joined_path)
    else:
        panos = load_from_csv(joined_path)

    return panos


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
    polygons: gpd.GeoDataFrame,
    capture_points: gpd.GeoDataFrame,
    buffer_dist: int,
    data_dir: str,
    buffer_polygons: Union[int, None] = None,
    check:bool = False
    ) -> pd.DataFrame:
    """
    Get Google Street View images coverage for a dataset of polygons, 
    where coverage is defined as the percentage of meters roads inside a polygon
    'observed' (15m buffer)

    Parameters
    ----------
    polygons : gpd.GeoDataFrame
        Input GeoDataFrame
    capture_points : gpd.GeoDataFrame
        Capture points GeoDataFrame
    data_dir : str
        Directory to save output files
    buffer_polygons : Union[int, None] = None
        Buffer distance for polygons
    check : bool = False
        Whether to export the data for plotting
        
    Returns
    -------
    pd.DataFrame
        Coverage area metrics
    """

    coverage_per_barrio_path = os.path.join(data_dir, "coverage_per_barrio.csv")
    if not os.path.exists(coverage_per_barrio_path):
        coverage_per_barrio = []
        if buffer_polygons is not None:
            polygons = buffer_region(polygons, buffer_dist=buffer_polygons)

        capture_points = buffer_region(capture_points, buffer_dist=buffer_dist)

        # For visual checks on kepler.gl
        if check:
            capture_points.to_csv(os.path.join(data_dir, "capture_points_buffered.csv"))
            roads = get_roads_from_gdf(polygons)
            roads.to_csv(os.path.join(data_dir, "roads.csv"))

        for _, row in polygons.iterrows():
            polygon = row["geometry"]
            intersecting_capture_points = capture_points[
                capture_points.intersects(polygon)
            ]
            union = pd.concat([
                intersecting_capture_points, 
                polygon
            ]).reset_index(drop=True).union_all()
            
            roads = get_roads_from_polygon(
                union
            )
            total = roads.to_crs(3857)["roadlength"].sum()

            result = roads.clip(capture_points.union_all()).to_crs(3857)
            result["roadlength"] = result.geometry.length
            partial = result.roadlength.sum()

            coverage_per_barrio.append({
                "id_renabap": row["id_renabap"],
                "total": total,
                "partial": partial,
                "coverage": round(partial / total, 3),
                "geometry": row["geometry"].wkt
            })
        
        coverage_per_barrio = pd.DataFrame(coverage_per_barrio)
        coverage_per_barrio.to_csv(coverage_per_barrio_path)
    else:
        coverage_per_barrio = load_from_csv(coverage_per_barrio_path)
    return coverage_per_barrio
        

def process_heading_fov(
    panos: gpd.GeoDataFrame,
    control_points: gpd.GeoDataFrame,
    data_dir: str,
    max_distance: int = 10,
    max_fov: int = 120,
) -> pd.DataFrame:
    """
    Process heading and FOV for panoramas.
    
    Parameters
    ----------
    panos : gpd.GeoDataFrame
        Panorama data
    control_points : gpd.GeoDataFrame
        Control points data
    data_dir : str
        Directory to save output files
        
    Returns
    -------
    gpd.GeoDataFrame
        Panorama data with heading and FOV
    """

    # We need to reproject before getting angles due to distance
    panos = panos.to_crs(3857)
    control_points = control_points.to_crs(3857)
    
    output_path = os.path.join(data_dir, "heading_fov.csv")
    if not os.path.exists(output_path):
        output = []
        for _, row in control_points.iterrows():
            cpid = row['cluster_id']
            cpgeom = row['geometry']
            related = panos[panos['cluster_id'] == cpid].copy()
            for _, panorow in related.iterrows():
                pano_id = panorow['pano_id']
                pano_geom = panorow['geometry']
                angles = get_angles(pano_geom, cpgeom, max_distance, max_fov)
                for angle in angles:
                    output.append({
                        "pano_id": pano_id,
                        "cluster_id": cpid,
                        "direction": angle[0],
                        "heading": angle[1],
                        "fov": angle[2],
                        "view_id": pano_id + "_" + angle[0]
                    })
        
        output = pd.DataFrame(output)
        output.to_csv(output_path)
    else:
        output = load_from_csv(output_path)

    return output