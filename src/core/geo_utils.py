"""
Geographic utilities for Street Level Change Detection.

This module provides utility functions for working with geographic data,
including point grid creation, region finding, and road extraction.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import transform
import pyproj
from typing import List, Union, Optional
from shapely import wkt
from tqdm import tqdm

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
    mask = wkt.loads(wkt_text)
    return mask

def create_point_grid(geometry, distance_meters: float) -> gpd.GeoDataFrame:
    """
    Create a grid of points spaced by `distance_meters` that cover the bounding box of the input geometry.

    Parameters
    ----------
    geometry : shapely.geometry
        A shapely geometry (Polygon, MultiPolygon, etc.)
    distance_meters : float
        Distance between points in meters

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing point geometries inside the input geometry
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

def create_point_grid_from_gdf(gdf: gpd.GeoDataFrame, dist_points: float) -> gpd.GeoDataFrame:
    """
    Create a grid of points from a GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing geometries
    dist_points : float
        Distance between points in meters

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing grid points
    """
    all_points = []
    for geom in tqdm(gdf.geometry, desc="Generating Grid Points"):
        points = create_point_grid(geom, dist_points)
        all_points.append(points)
    points_gdf = pd.concat(all_points).reset_index(drop=True)
    return points_gdf

def find_region(query: Union[str, List[str]]) -> gpd.GeoDataFrame:
    """
    Find the boundary of a region using OpenStreetMap.

    Parameters
    ----------
    query : Union[str, List[str]]
        The query or queries to search for the region

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the boundary of the region(s)
    """
    import osmnx
    
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


def get_roads_from_polygon(polygon) -> gpd.GeoDataFrame:
    """
    Extract road network from a polygon using OpenStreetMap.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon to extract roads from

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing road geometries and attributes
    """
    import osmnx
    
    TAGS = {"highway": True}
    
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


def get_roads_from_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract road network from all polygons in a GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing polygons

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing road geometries and attributes
    """
    df_output = []
    data = gdf.copy()
    for _, row in data.iterrows():
        geometry = row["geometry"]
        roads = get_roads_from_polygon(geometry)
        df_output.append(roads)
    df_output = pd.concat(df_output).reset_index(drop=True)
    df_output = df_output[df_output.element == "way"]
    df_output = df_output[["roadtype", "roadlength", "geometry"]]
    return df_output


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.

    Parameters
    ----------
    lat1 : float
        Latitude of the first point in degrees
    lon1 : float
        Longitude of the first point in degrees
    lat2 : float
        Latitude of the second point in degrees
    lon2 : float
        Longitude of the second point in degrees

    Returns
    -------
    float
        Distance in meters between the two points
    """
    from math import radians, sin, cos, sqrt, atan2
    
    # Earth radius in meters
    R = 6371000
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance


def buffer_region(gdf: gpd.GeoDataFrame, buffer_dist: float, 
                 mask: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
    """
    Buffer a region by a specified distance in meters.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing geometries to buffer
    buffer_dist : float
        Buffer distance in meters
    mask : Optional[gpd.GeoDataFrame], default=None
        Optional mask to intersect with before buffering

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with buffered geometries
    """
    buffered = gdf.copy()
    
    # Apply optional mask
    if mask is not None:
        buffered = buffered[buffered.intersects(mask)]
    
    # Buffer in meters (EPSG:3857), then back to WGS84
    buffered["geometry"] = (
        buffered.to_crs(3857).buffer(buffer_dist).to_crs(4326)
    )
    
    return buffered
