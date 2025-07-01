import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import transform
import pyproj


def create_point_grid(geometry, distance_meters):
    """
    Create a grid of points spaced by `distance_meters` that cover the bounding box of the input geometry.

    Parameters:
    - geometry: A shapely geometry (Polygon, MultiPolygon, etc.)
    - distance_meters: Distance between points in meters.

    Returns:
    - A list of shapely Point objects inside the geometry.
    """

    # Estimate a suitable UTM zone and define projections
    lon, lat = geometry.centroid.x, geometry.centroid.y
    utm_zone = int((lon + 180) / 6) + 1
    is_northern = lat >= 0
    crs_utm = f"+proj=utm +zone={utm_zone} +{'north' if is_northern else 'south'} +datum=WGS84 +units=m +no_defs"

    # Define transformation functions
    project = pyproj.Transformer.from_crs(
        "EPSG:4326", crs_utm, always_xy=True
    ).transform
    project_back = pyproj.Transformer.from_crs(
        crs_utm, "EPSG:4326", always_xy=True
    ).transform

    # Project geometry to meters
    geom_proj = transform(project, geometry)
    bounds = geom_proj.bounds
    minx, miny, maxx, maxy = bounds

    # Generate grid points
    x_coords = np.arange(minx, maxx, distance_meters)
    y_coords = np.arange(miny, maxy, distance_meters)

    grid_points = []
    for x in x_coords:
        for y in y_coords:
            pt = Point(x, y)
            if geom_proj.contains(pt):
                grid_points.append(transform(project_back, pt))

    grid_points = gpd.GeoDataFrame(grid_points, columns=["geometry"], crs=4326)
    return grid_points

import pandas as pd
from streetview import search_panoramas
from tqdm import tqdm


def get_panos(points_gdf, verbose=False):
    panos = []
    for _, row in tqdm(
        points_gdf.iterrows(), total=len(points_gdf), disable=not verbose
    ):
        lat, lon = row["geometry"].y, row["geometry"].x
        panos_there = search_panoramas(lat=lat, lon=lon)

        for pano in panos_there:
            id = pano.pano_id
            lat, lon = pano.lat, pano.lon
            date = pano.date
            panos.append((id, lat, lon, date))

    panos = (
        pd.DataFrame(panos, columns=["pano_id", "lat", "lon", "date"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return panos

import osmnx
import geopandas as gpd
import pandas as pd

TAGS = {"highway": True}


def find_region(query: str | list) -> gpd.GeoDataFrame:
    """
    Find the boundary of a region using OpenStreetMap.

    Parameters:
    - query (str | list): The query to search for the region.

    Returns:
    - gpd.GeoDataFrame: The boundary of the region.
    """
    if isinstance(query, list):
        gdf = gpd.GeoDataFrame(
            pd.concat(
                [osmnx.geocode_to_gdf(query=region) for region in query],
                ignore_index=True,
            ),
            geometry="geometry",
        )
        return gdf
    else:
        gdf = osmnx.geocode_to_gdf(query=query)
        return gdf


def get_roads_from_polygon(polygon):
    roads_full = osmnx.features.features_from_polygon(
        polygon.envelope, tags=TAGS
    )
    roads = roads_full[["highway", "geometry"]]
    roads = gpd.clip(roads, polygon)
    roads = roads.set_crs(4326)
    roads = roads.to_crs(3857)
    roads["roadlength"] = roads.geometry.length
    roads = roads.to_crs(4326)
    roads = roads.rename(columns={"highway": "roadtype"})
    return roads.reset_index()


def get_roads_from_gdf(gdf):
    df_output = []
    data = gdf.copy()
    for ix, row in data.iterrows():
        geometry = row["geometry"]
        roads = get_roads_from_polygon(geometry)
        df_output.append(roads)
    df_output = pd.concat(df_output).reset_index(drop=True)
    df_output = df_output[df_output.element == "way"]
    df_output = df_output[["roadtype", "roadlength", "geometry"]]
    return df_output
