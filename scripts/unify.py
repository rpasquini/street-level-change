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
from src.processing.point_unification import unify_points_h3, unify_points_dbscan
from src.core.panorama import PanoramaCollection
from src.data_handlers.loaders import load_from_csv
from src.data_handlers.exporters import export_to_csv, export_to_geojson


def load_panorama_data(file_path):
    """
    Load panorama data from a CSV file and convert to GeoDataFrame or PanoramaCollection.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file containing panorama data
        
    Returns
    -------
    gpd.GeoDataFrame or PanoramaCollection
        GeoDataFrame or PanoramaCollection with panorama data
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
            return panoramas
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


def analyze_h3_results(df):
    """
    Analyze H3 unification results.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with location_id column containing H3 indices
        
    Returns
    -------
    pd.DataFrame
        DataFrame with H3 index and count of points
    """
    # Count points per H3 index
    h3_counts = df['location_id'].value_counts().reset_index()
    h3_counts.columns = ['h3_index', 'point_count']
    
    # Calculate statistics
    total_points = len(df)
    unique_locations = len(h3_counts)
    avg_points_per_location = h3_counts['point_count'].mean()
    max_points = h3_counts['point_count'].max()
    
    print(f"\nH3 Unification Results:")
    print(f"Total points: {total_points}")
    print(f"Unique locations: {unique_locations}")
    print(f"Average points per location: {avg_points_per_location:.2f}")
    print(f"Maximum points in a location: {max_points}")
    
    return h3_counts


def analyze_dbscan_results(df):
    """
    Analyze DBSCAN unification results.
    
    Parameters
    ----------
    df : pd.DataFrame or gpd.GeoDataFrame
        DataFrame with location_id column containing cluster IDs
        
    Returns
    -------
    tuple
        (cluster_counts, cluster_centers) - DataFrames with cluster statistics and center points
    """
    # Count points per cluster
    cluster_counts = df['location_id'].value_counts().reset_index()
    cluster_counts.columns = ['cluster_id', 'point_count']
    
    # Calculate statistics
    total_points = len(df)
    unique_clusters = len(cluster_counts)
    avg_points_per_cluster = cluster_counts['point_count'].mean()
    max_points = cluster_counts['point_count'].max()
    noise_points = len(df[df['location_id'] == -1]) if -1 in df['location_id'].values else 0
    
    print(f"\nDBSCAN Unification Results:")
    print(f"Total points: {total_points}")
    print(f"Unique clusters: {unique_clusters}")
    print(f"Average points per cluster: {avg_points_per_cluster:.2f}")
    print(f"Maximum points in a cluster: {max_points}")
    print(f"Noise points: {noise_points}")
    
    # Ensure we have a GeoDataFrame with geometry column
    if not isinstance(df, gpd.GeoDataFrame) or 'geometry' not in df.columns:
        # Try to convert using lat/lon columns if available
        if 'lat' in df.columns and 'lon' in df.columns:
            from shapely.geometry import Point
            geometries = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
            gdf = gpd.GeoDataFrame(df.copy(), geometry=geometries, crs="EPSG:4326")
        else:
            print("Warning: Cannot calculate cluster centers - no geometry column or lat/lon columns found")
            # Return empty GeoDataFrame for centers
            empty_centers = gpd.GeoDataFrame(
                columns=['cluster_id', 'center_lat', 'center_lon', 'point_count'],
                geometry=[],
                crs="EPSG:4326"
            )
            return cluster_counts, empty_centers
    else:
        gdf = df
    
    # Calculate cluster centers (centroid of points in each cluster)
    cluster_centers = []
    
    # Skip noise points (cluster_id = -1)
    for cluster_id in sorted(gdf['location_id'].unique()):
        if cluster_id == -1:
            continue
            
        # Get points in this cluster
        cluster_points = gdf[gdf['location_id'] == cluster_id]
        
        # Calculate centroid
        center_lon = cluster_points.geometry.x.mean()
        center_lat = cluster_points.geometry.y.mean()
        
        cluster_centers.append({
            'cluster_id': cluster_id,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'point_count': len(cluster_points)
        })
    
    # Create DataFrame with cluster centers
    if cluster_centers:
        cluster_centers_df = pd.DataFrame(cluster_centers)
        
        # Create geometry column for the centers
        from shapely.geometry import Point
        geometries = [Point(row['center_lon'], row['center_lat']) for _, row in cluster_centers_df.iterrows()]
        cluster_centers_gdf = gpd.GeoDataFrame(
            cluster_centers_df, 
            geometry=geometries,
            crs="EPSG:4326"
        )
    else:
        # Create empty GeoDataFrame if no clusters
        cluster_centers_gdf = gpd.GeoDataFrame(
            columns=['cluster_id', 'center_lat', 'center_lon', 'point_count'],
            geometry=[],
            crs="EPSG:4326"
        )
    
    return cluster_counts, cluster_centers_gdf




def main():
    """
    Main function to test point unification algorithms.
    """
    # Default configuration
    input_file = 'data/panos/panos.csv'
    output_dir = 'data/point_unification_results'
    h3_resolution = 11
    dbscan_eps = 0.000045 # 5 meters at the equator
    dbscan_min_samples = 1
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load panorama data
    panorama_data = load_panorama_data(input_file)
    
    # Apply H3 unification
    print(f"\nApplying H3 unification with resolution={h3_resolution}")
    h3_results = unify_points_h3(panorama_data, resolution=h3_resolution)
    
    # Apply DBSCAN unification
    print(f"\nApplying DBSCAN unification with eps={dbscan_eps}, min_samples={dbscan_min_samples}")
    dbscan_results = unify_points_dbscan(panorama_data, eps=dbscan_eps, min_samples=dbscan_min_samples)
    
    # Analyze and save H3 results
    h3_counts = analyze_h3_results(h3_results)
    
    # Save results using data_handlers exporters
    h3_output_path = os.path.join(output_dir, 'h3_results.csv')
    
    # Convert to DataFrame if it's a PanoramaCollection
    if isinstance(h3_results, PanoramaCollection):
        h3_results_df = h3_results.to_dataframe()
        export_to_csv(h3_results_df, h3_output_path)
    else:
        export_to_csv(h3_results, h3_output_path)
    
    h3_counts_path = os.path.join(output_dir, 'h3_counts.csv')
    export_to_csv(h3_counts, h3_counts_path)
    print(f"H3 results saved to {h3_output_path}")
    print(f"H3 counts saved to {h3_counts_path}")
    
    # Analyze and save DBSCAN results
    dbscan_counts, dbscan_centers = analyze_dbscan_results(dbscan_results)
    
    # Save main results with cluster IDs
    dbscan_output_path = os.path.join(output_dir, 'dbscan_results.csv')
    if isinstance(dbscan_results, PanoramaCollection):
        dbscan_results_df = dbscan_results.to_dataframe()
        export_to_csv(dbscan_results_df, dbscan_output_path)
    else:
        export_to_csv(dbscan_results, dbscan_output_path)
    print(f"DBSCAN results saved to {dbscan_output_path}")
    
    # Save cluster counts
    dbscan_counts_path = os.path.join(output_dir, 'dbscan_counts.csv')
    export_to_csv(dbscan_counts, dbscan_counts_path)
    print(f"DBSCAN counts saved to {dbscan_counts_path}")
    
    # Save cluster centers
    dbscan_centers_path = os.path.join(output_dir, 'dbscan_centers.csv')
    export_to_csv(dbscan_centers, dbscan_centers_path)
    print(f"DBSCAN cluster centers saved to {dbscan_centers_path}")
    
    # Save cluster centers as GeoJSON for easy visualization
    dbscan_centers_geojson_path = os.path.join(output_dir, 'dbscan_centers.geojson')
    if len(dbscan_centers) > 0:
        export_to_geojson(dbscan_centers, dbscan_centers_geojson_path)
        print(f"DBSCAN cluster centers saved as GeoJSON to {dbscan_centers_geojson_path}")
        
    # Optional: Create interactive visualization
    create_interactive = False
    if create_interactive:
        from src.visualization.interactive_plotting import create_cluster_map, save_map
        
        # Create interactive map with DBSCAN clusters
        if isinstance(dbscan_results, PanoramaCollection):
            gdf = dbscan_results.to_geodataframe()
        else:
            gdf = dbscan_results
            
        if 'location_id' in gdf.columns:
            m = create_cluster_map(gdf, cluster_column='location_id')
            interactive_path = os.path.join(output_dir, 'dbscan_interactive_map.html')
            save_map(m, interactive_path)
            print(f"Interactive DBSCAN map saved to {interactive_path}")
            
        # Create interactive map with H3 clusters
        if isinstance(h3_results, PanoramaCollection):
            gdf = h3_results.to_geodataframe()
        else:
            gdf = h3_results
            
        if 'location_id' in gdf.columns:
            m = create_cluster_map(gdf, cluster_column='location_id')
            interactive_path = os.path.join(output_dir, 'h3_interactive_map.html')
            save_map(m, interactive_path)
            print(f"Interactive H3 map saved to {interactive_path}")
    

if __name__ == "__main__":
    main()
