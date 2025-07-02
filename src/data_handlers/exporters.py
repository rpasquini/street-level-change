"""
Data exporting functions for Street Level Change Detection.

This module provides functions for exporting panorama data to various formats,
including CSV files, GeoJSON files, and databases.
"""

import os
import pandas as pd
import geopandas as gpd
from typing import Optional, Union, Dict, Any, List

from src.core.panorama import PanoramaCollection


def ensure_directory_exists(file_path: str) -> None:
    """
    Ensure the directory for a file exists.
    
    Parameters
    ----------
    file_path : str
        Path to the file
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def export_panorama_data(
    data: Union[pd.DataFrame, gpd.GeoDataFrame, PanoramaCollection],
    file_path: str,
    format: Optional[str] = None
) -> str:
    """
    Export panorama data to a file.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, gpd.GeoDataFrame, PanoramaCollection]
        Data to export
    file_path : str
        Path to save the file
    format : Optional[str], default=None
        Format to export (csv, geojson, etc.). If None, inferred from file extension.
        
    Returns
    -------
    str
        Path to the exported file
    """
    # Ensure directory exists
    ensure_directory_exists(file_path)
    
    # Convert PanoramaCollection to GeoDataFrame if needed
    if isinstance(data, PanoramaCollection):
        data = data.to_geodataframe()
    
    # Determine format from file extension if not specified
    if format is None:
        _, ext = os.path.splitext(file_path)
        format = ext.lower().lstrip('.')
    
    # Export based on format
    if format in ['csv']:
        data.to_csv(file_path, index=False)
    elif format in ['geojson', 'json']:
        # Ensure we have a GeoDataFrame
        if isinstance(data, pd.DataFrame) and not isinstance(data, gpd.GeoDataFrame):
            if 'lat' in data.columns and 'lon' in data.columns:
                from shapely.geometry import Point
                geometries = [Point(lon, lat) for lon, lat in zip(data['lon'], data['lat'])]
                data = gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:4326")
            else:
                raise ValueError("Cannot convert to GeoJSON without geometry or lat/lon columns")
        
        # Export to GeoJSON
        data.to_file(file_path, driver='GeoJSON')
    else:
        raise ValueError(f"Unsupported export format: {format}")
    
    print(f"Exported data to {file_path}")
    return file_path


def export_h3_results(
    results: Union[pd.DataFrame, gpd.GeoDataFrame],
    output_dir: str,
    base_filename: str = 'h3_results'
) -> Dict[str, str]:
    """
    Export H3 unification results to files.
    
    Parameters
    ----------
    results : Union[pd.DataFrame, gpd.GeoDataFrame]
        H3 unification results
    output_dir : str
        Directory to save the files
    base_filename : str, default='h3_results'
        Base filename for the output files
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping file types to file paths
    """
    ensure_directory_exists(output_dir)
    
    # Export main results
    results_path = os.path.join(output_dir, f'{base_filename}.csv')
    results.to_csv(results_path, index=False)
    
    # Calculate and export counts
    if 'location_id' in results.columns:
        h3_counts = results['location_id'].value_counts().reset_index()
        h3_counts.columns = ['h3_index', 'point_count']
        
        counts_path = os.path.join(output_dir, f'{base_filename}_counts.csv')
        h3_counts.to_csv(counts_path, index=False)
        
        return {
            'results': results_path,
            'counts': counts_path
        }
    
    return {'results': results_path}


def export_dbscan_results(
    results: Union[pd.DataFrame, gpd.GeoDataFrame],
    output_dir: str,
    base_filename: str = 'dbscan_results',
    include_centers: bool = True
) -> Dict[str, str]:
    """
    Export DBSCAN unification results to files.
    
    Parameters
    ----------
    results : Union[pd.DataFrame, gpd.GeoDataFrame]
        DBSCAN unification results
    output_dir : str
        Directory to save the files
    base_filename : str, default='dbscan_results'
        Base filename for the output files
    include_centers : bool, default=True
        Whether to calculate and export cluster centers
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping file types to file paths
    """
    ensure_directory_exists(output_dir)
    
    # Export main results
    results_path = os.path.join(output_dir, f'{base_filename}.csv')
    results.to_csv(results_path, index=False)
    
    output_files = {'results': results_path}
    
    # Calculate and export counts
    if 'location_id' in results.columns:
        cluster_counts = results['location_id'].value_counts().reset_index()
        cluster_counts.columns = ['cluster_id', 'point_count']
        
        counts_path = os.path.join(output_dir, f'{base_filename}_counts.csv')
        cluster_counts.to_csv(counts_path, index=False)
        output_files['counts'] = counts_path
        
        # Calculate and export cluster centers if requested
        if include_centers and isinstance(results, gpd.GeoDataFrame):
            # Skip noise points (cluster_id = -1)
            cluster_centers = []
            
            for cluster_id in sorted(results['location_id'].unique()):
                if cluster_id == -1:
                    continue
                    
                # Get points in this cluster
                cluster_points = results[results['location_id'] == cluster_id]
                
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
                geometries = [Point(row['center_lon'], row['center_lat']) 
                             for _, row in cluster_centers_df.iterrows()]
                cluster_centers_gdf = gpd.GeoDataFrame(
                    cluster_centers_df, 
                    geometry=geometries,
                    crs="EPSG:4326"
                )
                
                # Export centers as CSV
                centers_path = os.path.join(output_dir, f'{base_filename}_centers.csv')
                cluster_centers_gdf.to_csv(centers_path, index=False)
                output_files['centers_csv'] = centers_path
                
                # Export centers as GeoJSON
                centers_geojson_path = os.path.join(output_dir, f'{base_filename}_centers.geojson')
                cluster_centers_gdf.to_file(centers_geojson_path, driver='GeoJSON')
                output_files['centers_geojson'] = centers_geojson_path
    
    return output_files


def export_to_csv(data: Union[pd.DataFrame, gpd.GeoDataFrame, PanoramaCollection], file_path: str) -> str:
    """
    Export data to a CSV file.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, gpd.GeoDataFrame, PanoramaCollection]
        Data to export
    file_path : str
        Path to save the CSV file
        
    Returns
    -------
    str
        Path to the exported file
    """
    return export_panorama_data(data, file_path, format='csv')


def export_to_geojson(data: Union[gpd.GeoDataFrame, PanoramaCollection], file_path: str) -> str:
    """
    Export data to a GeoJSON file.
    
    Parameters
    ----------
    data : Union[gpd.GeoDataFrame, PanoramaCollection]
        Data to export
    file_path : str
        Path to save the GeoJSON file
        
    Returns
    -------
    str
        Path to the exported file
    """
    return export_panorama_data(data, file_path, format='geojson')


def export_plot(
    fig,
    output_dir: str,
    filename: str,
    formats: List[str] = ['png']
) -> Dict[str, str]:
    """
    Export a matplotlib figure to image files.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to export
    output_dir : str
        Directory to save the files
    filename : str
        Base filename for the output files (without extension)
    formats : List[str], default=['png']
        List of formats to export (png, pdf, svg, etc.)
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping formats to file paths
    """
    ensure_directory_exists(output_dir)
    
    output_files = {}
    for fmt in formats:
        output_path = os.path.join(output_dir, f'{filename}.{fmt}')
        fig.savefig(output_path)
        output_files[fmt] = output_path
    
    return output_files
