"""
Static plotting functions for Street Level Change Detection.

This module provides functions for creating static plots and visualizations
of panorama data, including maps, histograms, and comparison plots.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple
import seaborn as sns
import os


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
    
    nans = gdf[date_column].isna().sum()
    total = len(gdf)
    share = f"{nans / total * 100:.2f}%"

    # Plot histogram of dates
    hist_fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(gdf[date_column], bins=20, ax=ax)
    ax.set_title(f'{title} - Histogram ({share} NaN)', fontsize=14)
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
    ax.set_title(f'{title} - Count by Year-Month ({share} NaN)', fontsize=14)
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
