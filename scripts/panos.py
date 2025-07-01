from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from src.utils import find_region, create_point_grid, get_panos


def load_mask(wkt_text):
    from shapely import wkt

    mask = wkt.loads(wkt_text)
    return mask


def buffer_region_for_osm(gdf, buffer_dist, mask=None):
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

    # Step 2: Submit one job per point (GeoDataFrame with one row)
    def get_pano_for_point(point_geom):
        point_gdf = gpd.GeoDataFrame(geometry=[point_geom], crs=4326)
        return get_panos(point_gdf)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(get_pano_for_point, geom)
            for geom in points_gdf.geometry
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Fetching Panoramas",
        ):
            result = future.result()
            if result is not None and not result.empty:
                results.append(result)

    return (
        pd.concat(results).reset_index(drop=True)
        if results
        else pd.DataFrame()
    )


if __name__ == "__main__":
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "demo")
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

    # Save region_gdf to CSV in data/demo
    region_gdf_path = os.path.join(output_dir, "region_gdf.csv")
    region_gdf.to_csv(region_gdf_path, index=None)
    print(f"Region GDF saved to {region_gdf_path}")

    # Save renabap intersected with region_gdf
    renabap_intersected = renabap[renabap.intersects(mask)]
    renabap_intersected_path = os.path.join(output_dir, "renabap_intersected.csv")
    renabap_intersected.to_csv(renabap_intersected_path, index=None)
    print(f"Renabap intersected saved to {renabap_intersected_path}")

    # Save intersected renabap buffers
    renabap_buffered = renabap_intersected.copy()
    renabap_buffered["geometry"] = (
        renabap_buffered.to_crs(3857).buffer(buffer_dist).to_crs(4326)
    )
    renabap_buffered_path = os.path.join(output_dir, "renabap_buffered.csv")
    renabap_buffered.to_csv(renabap_buffered_path, index=None)
    print(f"Renabap buffered saved to {renabap_buffered_path}")

    panos = loop_polys_get_panos(
        renabap,
        buffer_dist=buffer_dist,
        dist_points=dist_points_grid,
        mask=mask,
    )
    
    # Save panos to CSV in data/demo
    panos_path = os.path.join(output_dir, "panos.csv")
    panos.to_csv(panos_path, index=None)
    print(f"Panoramas saved to {panos_path}")
