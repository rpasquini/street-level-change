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
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add the project root to the path so we can import the point_unification module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.point_unification import h3_unification, dbscan_unification


def load_panorama_data(file_path):
    """
    Load panorama data from a CSV file and convert to GeoDataFrame.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file containing panorama data
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with panorama data
    """
    print(f"Loading panorama data from {file_path}")
    
    try:
        # Try to load as GeoDataFrame first
        gdf = gpd.read_file(file_path)
        print(f"Loaded {len(gdf)} panoramas as GeoDataFrame")
        return gdf
    except Exception:
        # If that fails, try loading as DataFrame and converting geometry column
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} panoramas as DataFrame")
            
            # Check if 'geometry' column exists
            if 'geometry' in df.columns:
                # Try to convert WKT strings to geometry objects
                try:
                    geometries = df['geometry'].apply(wkt.loads)
                    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
                    return gdf
                except Exception as e:
                    print(f"Error converting geometry column: {e}")
            
            # If no geometry column or conversion failed, try using lat/lon columns
            if 'lat' in df.columns and 'lon' in df.columns:
                from shapely.geometry import Point
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


def plot_results(h3_df, dbscan_df, output_dir):
    """
    Create plots to visualize the results.
    
    Parameters
    ----------
    h3_df : pd.DataFrame
        DataFrame with H3 unification results
    dbscan_df : pd.DataFrame
        DataFrame with DBSCAN unification results
    output_dir : str
        Directory to save plots
    """
    # Plot H3 point count distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(h3_df['point_count'], kde=True)
    plt.title('Distribution of Points per H3 Hexagon')
    plt.xlabel('Number of Points')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'h3_distribution.png'))
    
    # Plot DBSCAN point count distribution (excluding noise points)
    plt.figure(figsize=(10, 6))
    dbscan_no_noise = dbscan_df[dbscan_df['cluster_id'] != -1]
    if len(dbscan_no_noise) > 0:
        sns.histplot(dbscan_no_noise['point_count'], kde=True)
        plt.title('Distribution of Points per DBSCAN Cluster (Excluding Noise)')
        plt.xlabel('Number of Points')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dbscan_distribution.png'))


def main():
    # Default configuration
    input_file = 'data/demo/panos.csv'
    output_dir = 'data/point_unification_results'
    h3_resolution = 11
    dbscan_eps = 5
    dbscan_min_samples = 1
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load panorama data
    panos_gdf = load_panorama_data(input_file)
    
    # Apply H3 unification
    print(f"\nApplying H3 unification with resolution={h3_resolution}")
    h3_results = h3_unification(panos_gdf, resolution=h3_resolution)
    
    # Apply DBSCAN unification
    print(f"\nApplying DBSCAN unification with eps={dbscan_eps}m, min_samples={dbscan_min_samples}")
    dbscan_results = dbscan_unification(panos_gdf, eps_meters=dbscan_eps, min_samples=dbscan_min_samples)
    
    # Analyze and save H3 results
    h3_counts = analyze_h3_results(h3_results)
    h3_output_path = os.path.join(output_dir, 'h3_results.csv')
    h3_results.to_csv(h3_output_path, index=False)
    h3_counts_path = os.path.join(output_dir, 'h3_counts.csv')
    h3_counts.to_csv(h3_counts_path, index=False)
    print(f"H3 results saved to {h3_output_path}")
    print(f"H3 counts saved to {h3_counts_path}")
    
    # Analyze and save DBSCAN results
    dbscan_counts, dbscan_centers = analyze_dbscan_results(dbscan_results)
    
    # Save main results with cluster IDs
    dbscan_output_path = os.path.join(output_dir, 'dbscan_results.csv')
    dbscan_results.to_csv(dbscan_output_path, index=False)
    print(f"DBSCAN results saved to {dbscan_output_path}")
    
    # Save cluster counts
    dbscan_counts_path = os.path.join(output_dir, 'dbscan_counts.csv')
    dbscan_counts.to_csv(dbscan_counts_path, index=False)
    print(f"DBSCAN counts saved to {dbscan_counts_path}")
    
    # Save cluster centers
    dbscan_centers_path = os.path.join(output_dir, 'dbscan_centers.csv')
    dbscan_centers.to_csv(dbscan_centers_path, index=False)
    print(f"DBSCAN cluster centers saved to {dbscan_centers_path}")
    
    # Save cluster centers as GeoJSON for easy visualization
    dbscan_centers_geojson_path = os.path.join(output_dir, 'dbscan_centers.geojson')
    if len(dbscan_centers) > 0:
        dbscan_centers.to_file(dbscan_centers_geojson_path, driver='GeoJSON')
        print(f"DBSCAN cluster centers saved as GeoJSON to {dbscan_centers_geojson_path}")
    
    # Create visualization plots
    try:
        plot_results(h3_counts, dbscan_counts, output_dir)
        print(f"Visualization plots saved to {output_dir}")
    except Exception as e:
        print(f"Error creating plots: {e}")


if __name__ == "__main__":
    main()
