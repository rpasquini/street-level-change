#!/usr/bin/env python
"""
Script to generate visualization plots for Street Level Change Detection.

This script reads point unification results and panorama data,
and generates various visualization plots using the visualization module.
"""

import os
import geopandas as gpd
import pandas as pd

# Import from refactored modules
from src.visualization.static_plotting import (
    plot_date_distribution
)
from src.data_handlers.loaders import load_from_csv
from src.core.panorama import PanoramaCollection


def main():
    """
    Main function to generate visualization plots.
    """
    # Ensure output directory exists
    output_dir = 'data/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load panorama data and convert to GeoDataFrame or PanoramaCollection
    panos_data = load_from_csv('data/panos/panos.csv')
    
    # Try to convert to PanoramaCollection if it has the required columns
    try:
        panos = PanoramaCollection.from_dataframe(panos_data)
        panos_gdf = panos.to_geodataframe()
    except Exception as e:
        print(f"Could not convert to PanoramaCollection: {e}")
        # Fall back to creating a GeoDataFrame directly
        panos_gdf = gpd.GeoDataFrame(
            panos_data, 
            geometry=gpd.points_from_xy(panos_data.lon, panos_data.lat),
            crs="EPSG:4326"
        )

    print(panos_gdf.date.isna().sum())
    # Plot panorama date distribution
    plot_date_distribution(panos_gdf, output_dir=output_dir)
    print("Panorama date distribution plotted")

    print(f"Visualization plots saved to {output_dir}")
    

if __name__ == "__main__":
    main()