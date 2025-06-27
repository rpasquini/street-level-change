import osmnx
import geopandas as gpd
import pandas as pd

TAGS = {
    "highway": True
    # "highway": [
    #     "primary",
    #     "primary_link",
    #     "secondary",
    #     "secondary_link",
    #     "tertiary",
    #     "tertiary_link",
    #     "unclassified",
    #     "residential",
    #     "motorway",
    #     "trunk",
    #     "trunk_link",
    #     "road",
    #     "track" "pedestrian",
    #     "service",
    #     "living_street",
    #     "footway",
    #     "bridleway",
    #     "steps",
    #     "corridor",
    #     "path",
    #     "via_ferrata",
    # ]
}


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

