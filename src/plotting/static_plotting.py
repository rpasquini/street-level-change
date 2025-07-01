"""
Static plotting functions for visualizing point unification results.

This module provides functions to create static plots for H3 and DBSCAN
point unification results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
