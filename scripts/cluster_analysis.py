from notebooks.clustering_analysis import eps_range
from src.core.geo_utils import df_to_gdf
from src.core.point_unification import run_dbscan_evaluations
import geopandas as gpd
import pandas as pd
from shapely import wkt

def run_cluster_analysis(
    region_slug: str,
    extent: shapely.geometry.Polygon,
    eps_range: list,
    min_samples_range: list,
):
    panos = df_to_gdf(pd.read_csv(f'../data/{region_slug}/panos_enriched.csv'))

    rbp = df_to_gdf(pd.read_csv(f'../data/{region_slug}/renabap_intersected.csv'))

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

    results_df = run_dbscan_evaluations(panos, eps_range, min_samples_range)

    results_df.to_csv(f'../data/{region_slug}/clustering_grid_results.csv',index=None)

    return results_df

if __name__ == "__main__":
    region_slug = "tresdefebrero"
    
    extent=wkt.loads(
        "POLYGON((-58.58198 -34.582224, -58.515231 -34.582224, -58.515231 -34.635357, -58.58198 -34.635357, -58.58198 -34.582224))"
    )

    eps_range = [5, 2.5, 1] # in meters
    min_samples_range = [1, 2, 3]

    run_cluster_analysis(region_slug, extent, eps_range, min_samples_range)