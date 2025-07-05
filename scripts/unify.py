#!/usr/bin/env python
"""
Script to test point unification algorithms on panorama data.

This script reads panorama data from a CSV file, applies both H3 and DBSCAN
unification methods, and exports the results to CSV files.
"""

import os
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point

# Import from refactored modules
from src.processing.point_unification import unify_points_dbscan
from src.core.panorama import PanoramaCollection
from src.data_handlers.loaders import load_from_csv
from src.data_handlers.exporters import export_to_csv

def load_panorama_data(file_path):
    """
    Load panorama data from a CSV file and convert to GeoDataFrame or PanoramaCollection.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file containing panorama data
        
    Returns
    -------
    GeoDataFrame
    """
    print(f"Loading panorama data from {file_path}")
    
    try:
        # Try to load using data_handlers module
        df = load_from_csv(file_path)
        print(f"Loaded {len(df)} panoramas as DataFrame")
        
        # Try to convert to PanoramaCollection if it has the required columns
        try:
            panoramas = PanoramaCollection.from_dataframe(df)
            print(f"Converted to PanoramaCollection with {len(panoramas)} panoramas")
            return panoramas.to_geodataframe()
        except Exception as e:
            print(f"Could not convert to PanoramaCollection: {e}")
            
            # Fall back to creating a GeoDataFrame
            if 'geometry' in df.columns:
                try:
                    geometries = df['geometry'].apply(wkt.loads)
                    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
                    return gdf
                except Exception as e:
                    print(f"Error converting geometry column: {e}")
            
            # If no geometry column or conversion failed, try using lat/lon columns
            if 'lat' in df.columns and 'lon' in df.columns:
                geometries = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
                gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
                return gdf
            
            raise ValueError("Could not find geometry or lat/lon columns in the data")
    except Exception as e:
        print(f"Error loading panorama data: {e}")
        raise

def main():
    """
    Main function to test point unification algorithms.
    """
    # Default configuration
    input_file = 'data/panos/panos.csv'
    output_dir = 'data/point_unification_results'
    dbscan_eps = 0.000045 # 5 meters at the equator
    dbscan_min_samples = 1

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load panorama data
    panorama_data = load_panorama_data(input_file)
    
    # Apply DBSCAN unification
    print(f"\nApplying DBSCAN unification with eps={dbscan_eps}, min_samples={dbscan_min_samples}")
    dbscan_results = unify_points_dbscan(panorama_data, eps=dbscan_eps, min_samples=dbscan_min_samples)
    
    # Save results using data_handlers exporters
    dbscan_output_path = os.path.join(output_dir, 'dbscan_results.csv')
    export_to_csv(dbscan_results, dbscan_output_path)
    print(f"DBSCAN results saved to {dbscan_output_path}")
    
if __name__ == "__main__":
    main()
