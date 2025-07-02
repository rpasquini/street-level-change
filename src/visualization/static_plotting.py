"""
Static plotting functions for Street Level Change Detection.

This module provides functions for creating static plots and visualizations
of panorama data, including maps, histograms, and comparison plots.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from typing import Optional, Dict, Any, List, Tuple, Union
import seaborn as sns
import contextily as ctx
from datetime import datetime, timedelta
import os

from src.core.panorama import PanoramaCollection


def plot_points_on_map(
    gdf: gpd.GeoDataFrame,
    column: Optional[str] = None,
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (10, 10),
    title: str = 'Points on Map',
    basemap: bool = True,
    alpha: float = 0.7,
    markersize: int = 20,
    legend: bool = True
) -> plt.Figure:
    """
    Plot points on a map.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with points to plot
    column : Optional[str], default=None
        Column to use for coloring points
    cmap : str, default='viridis'
        Colormap to use
    figsize : Tuple[int, int], default=(10, 10)
        Figure size
    title : str, default='Points on Map'
        Plot title
    basemap : bool, default=True
        Whether to add a basemap
    alpha : float, default=0.7
        Alpha value for points
    markersize : int, default=20
        Size of markers
    legend : bool, default=True
        Whether to add a legend
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Create a copy to avoid modifying the original
    plot_gdf = gdf.copy()
    
    # Ensure the GeoDataFrame has a proper CRS
    if plot_gdf.crs is None:
        plot_gdf.set_crs(epsg=4326, inplace=True)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points
    if column is not None and column in plot_gdf.columns:
        plot_gdf.plot(
            column=column,
            cmap=cmap,
            ax=ax,
            alpha=alpha,
            markersize=markersize,
            legend=legend
        )
    else:
        plot_gdf.plot(
            ax=ax,
            alpha=alpha,
            markersize=markersize
        )
    
    # Add basemap if requested
    if basemap:
        # Convert to Web Mercator for contextily
        plot_gdf_web = plot_gdf.to_crs(epsg=3857)
        
        # Add basemap
        ctx.add_basemap(
            ax,
            crs=plot_gdf_web.crs.to_string(),
            source=ctx.providers.CartoDB.Positron
        )
    
    # Add title
    ax.set_title(title, fontsize=14)
    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    return fig


def plot_date_distribution(
    gdf: gpd.GeoDataFrame,
    date_column: str = 'date',
    figsize: Tuple[int, int] = (15, 10),
    title: str = 'Panorama Date Distribution',
    output_dir: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Plot distribution of panorama dates.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with panorama data
    date_column : str, default='date'
        Name of the column containing dates
    figsize : Tuple[int, int], default=(15, 10)
        Figure size
    title : str, default='Panorama Date Distribution'
        Plot title
    output_dir : str, default='data/plots'
        Directory to save plots
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary with figures
    """
    figures = {}
    
    # Check if date column exists
    if date_column not in gdf.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'No {date_column} column found', 
                ha='center', va='center', fontsize=14)
        figures['error'] = fig
        return figures
    
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(gdf[date_column]):
        try:
            gdf = gdf.copy()
            gdf[date_column] = pd.to_datetime(gdf[date_column])
        except Exception as e:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'Error converting {date_column} to datetime: {e}', 
                    ha='center', va='center', fontsize=14)
            figures['error'] = fig
            return figures
    
    # Plot histogram of dates
    hist_fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(gdf[date_column], bins=20, ax=ax)
    ax.set_title(f'{title} - Histogram', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    figures['histogram'] = hist_fig
    
    # Group by year and month
    gdf = gdf.copy()
    gdf['year_month'] = gdf[date_column].dt.strftime('%Y-%m')
    
    # Plot count by year-month
    year_month_counts = gdf['year_month'].value_counts().sort_index()
    
    time_fig, ax = plt.subplots(figsize=figsize)
    year_month_counts.plot(kind='bar', ax=ax)
    ax.set_title(f'{title} - Count by Year-Month', fontsize=14)
    ax.set_xlabel('Year-Month', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    figures['time_series'] = time_fig
    
    if output_dir:
        # Save figures
        for fig_name, fig in figures.items():
            fig_path = os.path.join(output_dir, f'{fig_name}.png')
            fig.savefig(fig_path)
            print(f"Saved {fig_name} plot to {fig_path}")
    
    return figures


def plot_comparison_results(
    comparison_results: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (15, 10),
    title: str = 'Unification Methods Comparison'
) -> Dict[str, plt.Figure]:
    """
    Plot comparison results of different unification methods.
    
    Parameters
    ----------
    comparison_results : Dict[str, Dict[str, Any]]
        Dictionary with comparison results
    figsize : Tuple[int, int], default=(15, 10)
        Figure size
    title : str, default='Unification Methods Comparison'
        Plot title
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary with figures
    """
    figures = {}
    
    # Extract statistics
    methods = list(comparison_results.keys())
    stats = {}
    
    for method in methods:
        if 'stats' in comparison_results[method]:
            stats[method] = comparison_results[method]['stats']
    
    if not stats:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No statistics found in comparison results', 
                ha='center', va='center', fontsize=14)
        figures['error'] = fig
        return figures
    
    # Plot number of clusters/cells
    clusters_fig, ax = plt.subplots(figsize=figsize)
    cluster_counts = {
        method: stats[method].get('unique_clusters', 
                               stats[method].get('unique_h3_cells',
                                             stats[method].get('unique_boxes', 0)))
        for method in methods
    }
    ax.bar(cluster_counts.keys(), cluster_counts.values())
    ax.set_title(f'{title} - Number of Clusters/Cells', fontsize=14)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    figures['clusters'] = clusters_fig
    
    # Plot average points per cluster/cell
    avg_fig, ax = plt.subplots(figsize=figsize)
    avg_points = {
        method: stats[method].get('avg_points_per_cluster',
                               stats[method].get('avg_points_per_cell',
                                             stats[method].get('avg_points_per_box', 0)))
        for method in methods
    }
    ax.bar(avg_points.keys(), avg_points.values())
    ax.set_title(f'{title} - Average Points per Cluster/Cell', fontsize=14)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Average Points', fontsize=12)
    figures['avg_points'] = avg_fig
    
    # Plot max points per cluster/cell
    max_fig, ax = plt.subplots(figsize=figsize)
    max_points = {
        method: stats[method].get('max_points_per_cluster',
                               stats[method].get('max_points_per_cell',
                                             stats[method].get('max_points_per_box', 0)))
        for method in methods
    }
    ax.bar(max_points.keys(), max_points.values())
    ax.set_title(f'{title} - Maximum Points per Cluster/Cell', fontsize=14)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Maximum Points', fontsize=12)
    figures['max_points'] = max_fig
    
    return figures


def plot_change_detection_results(
    change_results: List[Dict[str, Any]],
    figsize: Tuple[int, int] = (15, 10),
    title: str = 'Change Detection Results'
) -> Dict[str, plt.Figure]:
    """
    Plot change detection results.
    
    Parameters
    ----------
    change_results : List[Dict[str, Any]]
        List of change detection results
    figsize : Tuple[int, int], default=(15, 10)
        Figure size
    title : str, default='Change Detection Results'
        Plot title
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary with figures
    """
    figures = {}
    
    if not change_results:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No change detection results provided', 
                ha='center', va='center', fontsize=14)
        figures['error'] = fig
        return figures
    
    # Create DataFrame from results
    df = pd.DataFrame(change_results)
    
    # Plot similarity vs. change score
    scatter_fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df['similarity'], df['change_score'], alpha=0.7)
    ax.set_title(f'{title} - Similarity vs. Change Score', fontsize=14)
    ax.set_xlabel('Similarity Score', fontsize=12)
    ax.set_ylabel('Change Score', fontsize=12)
    ax.grid(True, alpha=0.3)
    figures['scatter'] = scatter_fig
    
    # Plot time difference vs. change score if time_difference_days exists
    if 'time_difference_days' in df.columns and df['time_difference_days'].notna().any():
        time_fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(df['time_difference_days'], df['change_score'], alpha=0.7)
        ax.set_title(f'{title} - Time Difference vs. Change Score', fontsize=14)
        ax.set_xlabel('Time Difference (days)', fontsize=12)
        ax.set_ylabel('Change Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        figures['time_diff'] = time_fig
    
    # Plot histogram of change scores
    hist_fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(df['change_score'], bins=20, ax=ax)
    ax.set_title(f'{title} - Change Score Distribution', fontsize=14)
    ax.set_xlabel('Change Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    figures['histogram'] = hist_fig
    
    return figures
