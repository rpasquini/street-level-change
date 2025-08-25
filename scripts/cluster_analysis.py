from src.core.geo_utils import df_to_gdf
from src.core.point_unification import run_dbscan_evaluations
import geopandas as gpd
import pandas as pd
from shapely import wkt

extent = wkt.loads(
    "POLYGON((-58.58198 -34.582224, -58.515231 -34.582224, -58.515231 -34.635357, -58.58198 -34.635357, -58.58198 -34.582224))"
)

panos = df_to_gdf(pd.read_csv('../data/tresdefebrero/panos_enriched.csv'))

rbp = df_to_gdf(pd.read_csv('../data/tresdefebrero/renabap_intersected.csv'))

panos = gpd.sjoin_nearest(
    panos.to_crs(3857),
    rbp[["id_renabap", "geometry"]].to_crs(3857),
    how="left",
    distance_col="distance",
).to_crs(4326)
panos = panos.rename(columns={"id_renabap": "closest_barrio"}).drop(
    columns=["index_right"]
)

panos = panos[panos.intersects(extent)]

eps_range = [5, 2.5, 1]          # in meters
min_samples_range = [1, 2, 3]

results_df = run_dbscan_evaluations(panos, eps_range, min_samples_range)

print(results_df.head())

results_df.to_csv('results_eps_5_2p5.csv',index=None)

import ast
results = pd.read_csv('results_eps_5_2p5.csv')
results['cluster_size_stats'] = results.cluster_size_stats.apply(ast.literal_eval)

results