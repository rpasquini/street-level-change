from src.plotting.static_plotting import plot_h3_results, plot_dbscan_results, plot_panorama_dates
import geopandas as gpd
import pandas as pd

if __name__ == "__main__":
    output_dir = 'data/plots'

    h3_counts = pd.read_csv('data/point_unification_results/h3_counts.csv')
    dbscan_counts = pd.read_csv('data/point_unification_results/dbscan_counts.csv')
    panos_gdf = pd.read_csv('data/demo/panos.csv')
    panos_gdf = gpd.GeoDataFrame(panos_gdf, geometry=gpd.points_from_xy(panos_gdf.lon, panos_gdf.lat))

    # Plot H3 results
    plot_h3_results(h3_counts, output_dir)

    # Plot DBSCAN results
    plot_dbscan_results(dbscan_counts, output_dir)

    # Plot panorama date distribution
    plot_panorama_dates(panos_gdf, output_dir)

    print(f"Visualization plots saved to {output_dir}")