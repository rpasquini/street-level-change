import numpy as np
from shapely.geometry import Point, box
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

    return grid_points
