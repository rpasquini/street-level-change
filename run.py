import src
from shapely import wkt
import geopandas as gpd
import pandas as pd

region = wkt.loads(
    "POLYGON((-58.4259406965 -34.5827413581, -58.4201201766 -34.5827413581, -58.4201201766 -34.5860185822, -58.4259406965 -34.5860185822, -58.4259406965 -34.5827413581))"
)

points = src.utils.create_point_grid(region, 10)


points = gpd.GeoDataFrame(points, columns=["geometry"], crs=4326)

# gpd.GeoDataFrame([region], columns=["geometry"]).to_csv("region.csv")
# points.to_csv("points.csv")


panos = src.sv.get_panos(points)
panos = gpd.GeoDataFrame(panos, columns=["id", "lat", "lon", "date"])
panos["date"] = pd.to_datetime(panos.date).dt.strftime("%Y-%m-%d")
panos.to_csv("panos.csv")
