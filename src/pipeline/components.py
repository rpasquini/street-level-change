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
import shapely
from shapely import union, buffer

from src.data_handlers.exporters import export_to_csv
from src.core.point_unification import (
    evaluate_compactness,
    spatial_silhouette_score,
    unify_points,
    compute_cluster_centroids,
)

from src.core.panorama import PanoramaCollection
from src.core.heading_fov import get_angles
from src.core.geo_utils import create_point_grid_from_gdf
from src.api.streetview import get_panoramas_for_points
from src.core.geo_utils import buffer_region
from src.data_handlers.loaders import load_from_csv, load_panorama_data
from src.core.geo_utils import (
    find_region,
    get_roads_from_polygon,
)


def process_region(
    region_osm: str, buffer_dist: int, data_dir: str
) -> Tuple[
    gpd.GeoDataFrame, gpd.GeoSeries, gpd.GeoDataFrame, gpd.GeoDataFrame
]:
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
        renabap_buffered = buffer_region(
            renabap_intersected, buffer_dist, exclude_original=True
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
        print(panoramas.head())

        export_to_csv(panoramas, panos_path)
        print(f"Panoramas saved to {panos_path}")
    else:
        panoramas = load_panorama_data(panos_path)

    return panoramas


def enrich_panorama_database_from_centroids(
    centroids: gpd.GeoDataFrame, renabap_buffered: gpd.GeoDataFrame, data_dir: str, max_workers: int = 10, verbose: bool = True
) -> gpd.GeoDataFrame:
    """
    Enrich panorama database by querying at each DBSCAN centroid location.
    
    This function takes DBSCAN centroids as input and queries the Google Street View API
    at each centroid location to find all available panoramas, including historical ones.
    This approach helps capture temporal sequences of panoramas at the same location that
    might be missed by the initial grid-based approach.

    Parameters
    ----------
    centroids : gpd.GeoDataFrame
        GeoDataFrame containing DBSCAN centroid points
    renabap_buffered : gpd.GeoDataFrame
        Buffered RENABAP data
    data_dir : str
        Directory to save output files
    max_workers : int, default=10
        Maximum number of parallel workers for API queries
    verbose : bool, default=True
        Whether to display progress information

    Returns
    -------
    gpd.GeoDataFrame
        Enriched panorama data as a GeoDataFrame
    """
    
    # Path for enriched panoramas
    enriched_panos_path = os.path.join(data_dir, "panos_enriched.csv")
    
    # Check if enriched panoramas already exist
    if os.path.exists(enriched_panos_path):
        print(f"Loading existing enriched panoramas from {enriched_panos_path}")
        return load_panorama_data(enriched_panos_path)
    
    # Load original panoramas for comparison later
    original_panos_path = os.path.join(data_dir, "panos.csv")
    original_panoramas = None
    if os.path.exists(original_panos_path):
        original_panoramas = load_panorama_data(original_panos_path)
        print(f"Loaded {len(original_panoramas)} original panoramas for comparison")
    
    # Query panoramas at each centroid location
    print(f"Querying panoramas at {len(centroids)} centroid locations...")
    enriched_panoramas = get_panoramas_for_points(
        centroids, max_workers=max_workers, verbose=verbose
    )
    enriched_panoramas = enriched_panoramas.clean(renabap_buffered)
    
    combined_panoramas = []
    # Combine with original panoramas if available
    if original_panoramas is not None:
        # Create a new collection with all panoramas
        
        # Add original panoramas
        for _, panorama in original_panoramas.iterrows():
            combined_panoramas.append(panorama)
        
        existing_ids = [panorama.pano_id for _, panorama in original_panoramas.iterrows()]
        # Add new panoramas from centroids
        for _, panorama in enriched_panoramas.iterrows():
            if panorama.pano_id not in existing_ids:
                combined_panoramas.append(panorama)
        
        # Report statistics
        original_count = len(original_panoramas)
        combined_count = len(combined_panoramas)
        new_count = combined_count - original_count
        
        print(f"Original panorama count: {original_count}")
        print(f"New panoramas found: {new_count}")
        print(f"Total combined panoramas: {combined_count}")
        
        # Use the combined collection
        enriched_panoramas = combined_panoramas
    
    combined_panoramas = PanoramaCollection(combined_panoramas)
    # Save the enriched panoramas
    export_to_csv(combined_panoramas.to_dataframe(), enriched_panos_path)
    print(f"Enriched panoramas saved to {enriched_panos_path}")
    
    return combined_panoramas


def process_dbscan(
    panoramas: gpd.GeoDataFrame, eps: float, min_samples: int, data_dir: str,
    output_prefix: str = ""
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Process DBSCAN clustering on panorama data.
    
    Parameters
    ----------
    panoramas : gpd.GeoDataFrame
        GeoDataFrame containing panorama data
    eps : float
        DBSCAN epsilon parameter (maximum distance between points)
    min_samples : int
        DBSCAN min_samples parameter (minimum points to form a cluster)
    data_dir : str
        Directory to save output files
    output_prefix : str, default=""
        Prefix for output filenames (e.g., "enriched_" for enriched panorama data)
        
    Returns
    -------
    Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        Tuple containing (clustered panoramas, cluster centroids)
    """
    dbscan_path = os.path.join(data_dir, f"{output_prefix}dbscan.csv")
    centroids_path = os.path.join(data_dir, f"{output_prefix}centroids.csv")
    
    if os.path.exists(dbscan_path) and os.path.exists(centroids_path):
        print(f"Loading existing DBSCAN results from {dbscan_path}")
        dbscan_results = load_from_csv(dbscan_path)
        print(f"Loading existing centroids from {centroids_path}")
        centroids = load_from_csv(centroids_path)
        return dbscan_results, centroids
    
    print(f"Applying DBSCAN clustering with eps={eps}, min_samples={min_samples}")
    # Apply DBSCAN clustering
    dbscan_results = unify_points(panoramas, eps=eps, min_samples=min_samples)
    
    # Compute centroids
    print("Computing cluster centroids")
    centroids = compute_cluster_centroids(dbscan_results)
    
    # Save results
    print(f"Saving DBSCAN results to {dbscan_path}")
    export_to_csv(dbscan_results, dbscan_path)
    print(f"Saving centroids to {centroids_path}")
    export_to_csv(centroids, centroids_path)
    
    print(f"DBSCAN found {len(centroids)} clusters")
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
        renabap_buffered = buffer_region(
            renabap_intersected, barrio_buffer_dist, exclude_original=True
        )
        # Mark pano points as inside or inside_buffered
        panos["inside"] = panos.within(renabap_intersected.union_all()).astype(
            int
        )
        panos["close"] = panos.within(renabap_buffered.union_all()).astype(int)
        panos = gpd.sjoin_nearest(
            panos.to_crs(3857),
            renabap_intersected[["id_renabap", "geometry"]].to_crs(3857),
            how="left",
            distance_col="distance",
        ).to_crs(4326)

        panos["inside_close"] = panos["inside"] + panos["close"]

        panos = panos.rename(columns={"id_renabap": "closest_barrio"}).drop(
            columns=["index_right"]
        )

        # Save joined data
        export_to_csv(panos, joined_path)
    else:
        panos = load_from_csv(joined_path)

    return panos


def calculate_coverage_area(
    polygons: gpd.GeoDataFrame,
    capture_points: gpd.GeoDataFrame,
    buffer_dist: int,
    data_dir: str,
    buffer_polygons: Union[int, None] = None,
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

    Returns
    -------
    pd.DataFrame
        Coverage area metrics
    """

    coverage_per_barrio_path = os.path.join(
        data_dir, "coverage_per_barrio.csv"
    )
    if not os.path.exists(coverage_per_barrio_path):
        coverage_per_barrio = []
        if buffer_polygons is not None:
            buffered_polygons = buffer_region(
                polygons, buffer_dist=buffer_polygons
            )

        capture_points = buffer_region(capture_points, buffer_dist=buffer_dist)

        intersecting_capture_points = capture_points[
            capture_points.intersects(polygons.union_all())
        ]
        osm_polygons = pd.concat(
            [buffered_polygons, intersecting_capture_points]
        ).reset_index(drop=True)
        roads = get_roads_from_polygon(osm_polygons.union_all())
        roads.to_csv(os.path.join(data_dir, "roads.csv"))

        roads = roads.to_crs(3857)
        polygons = polygons.to_crs(3857)
        capture_points = capture_points.to_crs(3857)

        from pyproj import Transformer
        transformer = Transformer.from_crs(3857, 4326, always_xy=True)
        for _, row in polygons.iterrows():
            polygon = row["geometry"]
            intersecting_capture_points = capture_points[
                capture_points.intersects(polygon)
            ]

            barrio_and_capture_points = union(
                buffer(polygon, buffer_polygons),
                intersecting_capture_points.union_all(),
            )

            total_area = barrio_and_capture_points.area
            total = roads.clip(barrio_and_capture_points)["roadlength"].sum()

            street_density = total / total_area

            result = roads.clip(intersecting_capture_points)

            result["roadlength"] = result.geometry.length
            partial = result.roadlength.sum()

            def handle_zeroes(partial, total):
                if (total == 0) or (partial == 0):
                    return 0
                return round(partial / total, 3)

            coverage_per_barrio.append(
                {
                    "id_renabap": row["id_renabap"],
                    "total": total,
                    "partial": partial,
                    "coverage": handle_zeroes(partial, total),
                    "street_density_m_m2": street_density,
                    "geometry": shapely.transform(polygon, transformer.transform, interleaved=False).wkt,
                }
            )

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
            cpid = row["cluster_id"]
            cpgeom = row["geometry"]
            related = panos[panos["cluster_id"] == cpid].copy()
            for _, panorow in related.iterrows():
                pano_id = panorow["pano_id"]
                pano_geom = panorow["geometry"]
                angles = get_angles(pano_geom, cpgeom, max_distance, max_fov)
                for angle in angles:
                    output.append(
                        {
                            "pano_id": pano_id,
                            "cluster_id": cpid,
                            "direction": angle[0],
                            "heading": angle[1],
                            "fov": angle[2],
                            "view_id": pano_id + "_" + angle[0],
                        }
                    )

        output = pd.DataFrame(output)
        output.to_csv(output_path)
    else:
        output = load_from_csv(output_path)

    return output