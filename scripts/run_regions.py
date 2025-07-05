import os
import geopandas as gpd
from src.data_handlers.exporters import export_to_csv
from src.processing.point_unification import unify_points, compute_cluster_centroids
from src.core.geo_utils import create_point_grid_from_gdf
from src.api.streetview import get_panoramas_for_points
from src.core.geo_utils import buffer_region
from src.data_handlers.loaders import load_from_csv, load_panorama_data
from src.core.geo_utils import find_region

def run_region(region_slug: str, region_osm: str):
    output_dir = os.path.join("./data", region_slug)
    os.makedirs(output_dir, exist_ok=True)
    
    # Polygons buffer distance in meters
    buffer_dist = 500
    # Distance between points to point-gridding polygon buffers
    dist_points_grid = 50

    # DBSCAN parameters
    dbscan_eps = 0.000045 # 5 meters at the equator
    dbscan_min_samples = 1

    # Polígonos del RENABAP
    # https://datos.gob.ar/dataset/habitat-registro-nacional-barrios-populares
    renabap = gpd.read_file(
        "https://archivo.habitat.gob.ar/dataset/ssisu/renabap-datos-barrios-geojson"
    )

    region_gdf_path = os.path.join(output_dir, "region_gdf.csv")
    renabap_intersected_path = os.path.join(output_dir, "renabap_intersected.csv")
    renabap_buffered_path = os.path.join(output_dir, "renabap_buffered.csv")
    panos_path = os.path.join(output_dir, "panos.csv")
    dbscan_output_path = os.path.join(output_dir, 'dbscan_results.csv')
    dbscan_centroids_path = os.path.join(output_dir, 'dbscan_centroids.csv')

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
        export_to_csv(renabap_buffered, renabap_buffered_path)
    else:
        renabap_buffered = load_from_csv(renabap_buffered_path)

    if not os.path.exists(panos_path):
        points_gdf = create_point_grid_from_gdf(renabap_buffered, dist_points_grid)
        
        panoramas = get_panoramas_for_points(
            points_gdf, 
            verbose=True
        )

        export_to_csv(panoramas, panos_path)
        print(f"Panoramas saved to {panos_path}")
    else:
        panoramas = load_panorama_data(panos_path)

    if not os.path.exists(dbscan_output_path):
        # Apply DBSCAN unification
        print(f"\nApplying DBSCAN unification with eps={dbscan_eps}, min_samples={dbscan_min_samples}")
        dbscan_results = unify_points(panoramas, eps=dbscan_eps, min_samples=dbscan_min_samples)
        
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


if __name__ == "__main__":
    osm_region = "Partido de Tres de Febrero, Buenos Aires, Argentina"
    run_region("tresdefebrero", osm_region)