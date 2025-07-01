"""
Static plotting functions for visualizing point unification results.

This module provides functions to create static plots for H3 and DBSCAN
point unification results, as well as temporal analysis of panorama data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np


def plot_h3_results(h3_df, output_dir):
    """
    Create plots to visualize H3 unification results.
    
    Parameters
    ----------
    h3_df : pd.DataFrame
        DataFrame with H3 unification results, must contain 'point_count' column
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
    plt.close()


def plot_dbscan_results(dbscan_df, output_dir):
    """
    Create plots to visualize DBSCAN unification results.
    
    Parameters
    ----------
    dbscan_df : pd.DataFrame
        DataFrame with DBSCAN unification results, must contain 'cluster_id' and 'point_count' columns
    output_dir : str
        Directory to save plots
    """
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
    plt.close()


def plot_panorama_dates(panos_gdf, output_dir):
    """
    Create a plot showing the count of panorama images by date.
    
    Parameters
    ----------
    panos_gdf : gpd.GeoDataFrame or pd.DataFrame
        GeoDataFrame or DataFrame containing panorama data with a 'date' column
    output_dir : str
        Directory to save plots
    """
    # Check if the dataframe has a date column
    if 'date' not in panos_gdf.columns:
        raise ValueError("The panorama dataframe must contain a 'date' column")
    
    # Convert date strings to datetime objects if needed
    date_col = panos_gdf['date']
    if isinstance(date_col.iloc[0], str):
        # Try to parse dates with different formats
        try:
            date_col = pd.to_datetime(date_col)
        except Exception as e:
            print(f"Warning: Could not parse dates automatically: {e}")
            # Try a specific format if automatic parsing fails
            try:
                date_col = pd.to_datetime(date_col, format='%Y-%m-%d')
            except Exception as e:
                print(f"Error: Failed to parse dates with format '%Y-%m-%d': {e}")
                return
    
    # Extract year and month for grouping
    panos_gdf = panos_gdf.copy()
    panos_gdf['year'] = date_col.dt.year
    panos_gdf['month'] = date_col.dt.month
    panos_gdf['year_month'] = date_col.dt.strftime('%Y-%m')
    
    # Count images by year-month
    date_counts = panos_gdf.groupby('year_month').size().reset_index(name='count')
    date_counts = date_counts.sort_values('year_month')
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    ax = sns.barplot(x='year_month', y='count', data=date_counts)
    plt.title('Panorama Image Count by Date')
    plt.xlabel('Date (Year-Month)')
    plt.ylabel('Number of Images')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, count in enumerate(date_counts['count']):
        ax.text(i, count + 0.1, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'panorama_date_distribution.png'))
    plt.close()
    
    # Also create a yearly aggregation
    year_counts = panos_gdf.groupby('year').size().reset_index(name='count')
    year_counts = year_counts.sort_values('year')
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='year', y='count', data=year_counts)
    plt.title('Panorama Image Count by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Images')
    
    # Add value labels on top of bars
    for i, count in enumerate(year_counts['count']):
        ax.text(i, count + 0.1, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'panorama_yearly_distribution.png'))
    plt.close()
