"""
Interactive plotting functions for Street Level Change Detection.

This module provides functions for creating interactive plots and visualizations
of panorama data, including interactive maps and dashboards.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
import folium
from folium import plugins
import branca.colormap as cm
import json
from datetime import datetime

from src.core.panorama import PanoramaCollection


def create_interactive_map(
    center: Tuple[float, float] = (0, 0),
    zoom_start: int = 2,
    tiles: str = 'OpenStreetMap'
) -> folium.Map:
    """
    Create an interactive map.
    
    Parameters
    ----------
    center : Tuple[float, float], default=(0, 0)
        Center coordinates (latitude, longitude)
    zoom_start : int, default=2
        Initial zoom level
    tiles : str, default='OpenStreetMap'
        Tile provider
        
    Returns
    -------
    folium.Map
        Folium map object
    """
    return folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles=tiles
    )


def add_points_to_map(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
    column: Optional[str] = None,
    popup_columns: Optional[List[str]] = None,
    color: str = 'blue',
    radius: int = 5,
    fill: bool = True,
    colormap: Optional[str] = None
) -> folium.Map:
    """
    Add points to an interactive map.
    
    Parameters
    ----------
    m : folium.Map
        Folium map object
    gdf : gpd.GeoDataFrame
        GeoDataFrame with points to add
    column : Optional[str], default=None
        Column to use for coloring points
    popup_columns : Optional[List[str]], default=None
        Columns to include in popups
    color : str, default='blue'
        Color of points (used if column is None)
    radius : int, default=5
        Radius of points
    fill : bool, default=True
        Whether to fill points
    colormap : Optional[str], default=None
        Colormap to use if column is not None
        
    Returns
    -------
    folium.Map
        Updated Folium map object
    """
    # Make a copy to avoid modifying the original
    plot_gdf = gdf.copy()
    
    # Ensure the GeoDataFrame has a proper CRS
    if plot_gdf.crs is None:
        plot_gdf.set_crs(epsg=4326, inplace=True)
    elif plot_gdf.crs != "EPSG:4326":
        plot_gdf = plot_gdf.to_crs(epsg=4326)
    
    # Set default popup columns if not provided
    if popup_columns is None:
        popup_columns = [col for col in plot_gdf.columns 
                        if col != 'geometry' and len(plot_gdf[col].unique()) < 100]
    
    # Create colormap if needed
    if column is not None and column in plot_gdf.columns:
        if isinstance(plot_gdf[column].iloc[0], (int, float, np.number)):
            # Numeric column
            vmin = plot_gdf[column].min()
            vmax = plot_gdf[column].max()
            
            if colormap is None:
                colormap = 'viridis'
            
            # Create colormap
            cmap = cm.LinearColormap(
                colors=['blue', 'cyan', 'yellow', 'red'],
                vmin=vmin,
                vmax=vmax,
                caption=column
            )
            
            # Add colormap to map
            m.add_child(cmap)
            
            # Add points with colors based on column values
            for idx, row in plot_gdf.iterrows():
                # Create popup content
                popup_content = '<table>'
                for col in popup_columns:
                    if col in row and pd.notna(row[col]):
                        popup_content += f'<tr><td><b>{col}</b></td><td>{row[col]}</td></tr>'
                popup_content += '</table>'
                
                # Get point coordinates
                lat, lon = row.geometry.y, row.geometry.x
                
                # Get color based on column value
                point_color = cmap(row[column])
                
                # Add marker
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    color=point_color,
                    fill=fill,
                    fill_color=point_color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_content, max_width=300)
                ).add_to(m)
        else:
            # Categorical column
            categories = plot_gdf[column].unique()
            
            # Create color dictionary
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            if len(categories) <= 10:
                colors = list(mcolors.TABLEAU_COLORS.values())
            else:
                cmap = plt.get_cmap('tab20')
                colors = [mcolors.rgb2hex(cmap(i)) for i in range(min(20, len(categories)))]
                
                # If more than 20 categories, cycle through colors
                if len(categories) > 20:
                    colors = colors * (len(categories) // 20 + 1)
            
            color_dict = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}
            
            # Add points with colors based on categories
            for idx, row in plot_gdf.iterrows():
                # Create popup content
                popup_content = '<table>'
                for col in popup_columns:
                    if col in row and pd.notna(row[col]):
                        popup_content += f'<tr><td><b>{col}</b></td><td>{row[col]}</td></tr>'
                popup_content += '</table>'
                
                # Get point coordinates
                lat, lon = row.geometry.y, row.geometry.x
                
                # Get color based on category
                if pd.notna(row[column]) and row[column] in color_dict:
                    point_color = color_dict[row[column]]
                else:
                    point_color = 'gray'
                
                # Add marker
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    color=point_color,
                    fill=fill,
                    fill_color=point_color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_content, max_width=300)
                ).add_to(m)
            
            # Add legend
            legend_html = '''
                <div style="position: fixed; 
                            bottom: 50px; right: 50px; width: 150px; height: auto; 
                            border:2px solid grey; z-index:9999; font-size:12px;
                            background-color:white; padding: 10px;
                            overflow-y: auto; max-height: 300px;">
                <p><b>{}</b></p>
            '''.format(column)
            
            for cat, col in color_dict.items():
                legend_html += f'<p><span style="color:{col};">●</span> {cat}</p>'
            
            legend_html += '</div>'
            m.get_root().html.add_child(folium.Element(legend_html))
    else:
        # No column specified, use single color
        for idx, row in plot_gdf.iterrows():
            # Create popup content
            popup_content = '<table>'
            for col in popup_columns:
                if col in row and pd.notna(row[col]):
                    popup_content += f'<tr><td><b>{col}</b></td><td>{row[col]}</td></tr>'
            popup_content += '</table>'
            
            # Get point coordinates
            lat, lon = row.geometry.y, row.geometry.x
            
            # Add marker
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=fill,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(m)
    
    return m


def add_cluster_centers_to_map(
    m: folium.Map,
    centers_gdf: gpd.GeoDataFrame,
    popup_columns: Optional[List[str]] = None,
    color: str = 'red',
    radius: int = 10
) -> folium.Map:
    """
    Add cluster centers to an interactive map.
    
    Parameters
    ----------
    m : folium.Map
        Folium map object
    centers_gdf : gpd.GeoDataFrame
        GeoDataFrame with cluster centers
    popup_columns : Optional[List[str]], default=None
        Columns to include in popups
    color : str, default='red'
        Color of cluster centers
    radius : int, default=10
        Radius of cluster centers
        
    Returns
    -------
    folium.Map
        Updated Folium map object
    """
    # Make a copy to avoid modifying the original
    plot_gdf = centers_gdf.copy()
    
    # Ensure the GeoDataFrame has a proper CRS
    if plot_gdf.crs is None:
        plot_gdf.set_crs(epsg=4326, inplace=True)
    elif plot_gdf.crs != "EPSG:4326":
        plot_gdf = plot_gdf.to_crs(epsg=4326)
    
    # Set default popup columns if not provided
    if popup_columns is None:
        popup_columns = [col for col in plot_gdf.columns 
                        if col != 'geometry' and len(plot_gdf[col].unique()) < 100]
    
    # Add cluster centers to map
    for idx, row in plot_gdf.iterrows():
        # Create popup content
        popup_content = '<table>'
        for col in popup_columns:
            if col in row and pd.notna(row[col]):
                popup_content += f'<tr><td><b>{col}</b></td><td>{row[col]}</td></tr>'
        popup_content += '</table>'
        
        # Get point coordinates
        lat, lon = row.geometry.y, row.geometry.x
        
        # Add marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(m)
    
    return m


def create_time_slider_map(
    gdf: gpd.GeoDataFrame,
    date_column: str = 'date',
    popup_columns: Optional[List[str]] = None,
    color: str = 'blue',
    radius: int = 5
) -> folium.Map:
    """
    Create an interactive map with a time slider.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with points to add
    date_column : str, default='date'
        Column containing dates
    popup_columns : Optional[List[str]], default=None
        Columns to include in popups
    color : str, default='blue'
        Color of points
    radius : int, default=5
        Radius of points
        
    Returns
    -------
    folium.Map
        Folium map object with time slider
    """
    # Make a copy to avoid modifying the original
    plot_gdf = gdf.copy()
    
    # Ensure the GeoDataFrame has a proper CRS
    if plot_gdf.crs is None:
        plot_gdf.set_crs(epsg=4326, inplace=True)
    elif plot_gdf.crs != "EPSG:4326":
        plot_gdf = plot_gdf.to_crs(epsg=4326)
    
    # Check if date column exists
    if date_column not in plot_gdf.columns:
        # Create a basic map without time slider
        center = (plot_gdf.geometry.y.mean(), plot_gdf.geometry.x.mean())
        m = create_interactive_map(center=center, zoom_start=12)
        return add_points_to_map(m, plot_gdf, popup_columns=popup_columns, color=color, radius=radius)
    
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(plot_gdf[date_column]):
        try:
            plot_gdf[date_column] = pd.to_datetime(plot_gdf[date_column])
        except Exception as e:
            # Create a basic map without time slider
            center = (plot_gdf.geometry.y.mean(), plot_gdf.geometry.x.mean())
            m = create_interactive_map(center=center, zoom_start=12)
            return add_points_to_map(m, plot_gdf, popup_columns=popup_columns, color=color, radius=radius)
    
    # Create map centered on the mean coordinates
    center = (plot_gdf.geometry.y.mean(), plot_gdf.geometry.x.mean())
    m = create_interactive_map(center=center, zoom_start=12)
    
    # Set default popup columns if not provided
    if popup_columns is None:
        popup_columns = [col for col in plot_gdf.columns 
                        if col != 'geometry' and col != date_column and len(plot_gdf[col].unique()) < 100]
    
    # Create feature group for each time period
    # Group by year-month
    plot_gdf['year_month'] = plot_gdf[date_column].dt.strftime('%Y-%m')
    
    # Create time features
    features = []
    
    for year_month, group in plot_gdf.groupby('year_month'):
        # Create feature group for this time period
        feature_group = folium.FeatureGroup(name=year_month)
        
        # Add points to feature group
        for idx, row in group.iterrows():
            # Create popup content
            popup_content = '<table>'
            for col in popup_columns + [date_column]:
                if col in row and pd.notna(row[col]):
                    popup_content += f'<tr><td><b>{col}</b></td><td>{row[col]}</td></tr>'
            popup_content += '</table>'
            
            # Get point coordinates
            lat, lon = row.geometry.y, row.geometry.x
            
            # Add marker to feature group
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(feature_group)
        
        # Add feature group to map
        feature_group.add_to(m)
        
        # Add to features list with timestamp
        try:
            timestamp = datetime.strptime(year_month, '%Y-%m').timestamp() * 1000
            features.append({
                'feature_group': feature_group,
                'timestamp': timestamp,
                'label': year_month
            })
        except Exception as e:
            # Skip this time period if there's an error
            pass
    
    # Sort features by timestamp
    features.sort(key=lambda x: x['timestamp'])
    
    # Create time slider
    if features:
        time_slider = plugins.TimestampedGeoJson(
            {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [0, 0]  # Dummy coordinates
                        },
                        'properties': {
                            'times': [feature['timestamp'] for feature in features]
                        }
                    }
                ]
            },
            period='P1M',  # 1 month period
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=10,
            loop_button=True,
            date_options='YYYY-MM',
            time_slider_drag_update=True
        )
        
        # Add time slider to map
        m.add_child(time_slider)
    
    return m


def create_heatmap(
    gdf: gpd.GeoDataFrame,
    weight_column: Optional[str] = None,
    radius: int = 15,
    blur: int = 10,
    gradient: Optional[Dict[float, str]] = None
) -> folium.Map:
    """
    Create a heatmap from points.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with points
    weight_column : Optional[str], default=None
        Column to use for point weights
    radius : int, default=15
        Radius of each point in the heatmap
    blur : int, default=10
        Blur factor
    gradient : Optional[Dict[float, str]], default=None
        Color gradient
        
    Returns
    -------
    folium.Map
        Folium map with heatmap
    """
    # Make a copy to avoid modifying the original
    plot_gdf = gdf.copy()
    
    # Ensure the GeoDataFrame has a proper CRS
    if plot_gdf.crs is None:
        plot_gdf.set_crs(epsg=4326, inplace=True)
    elif plot_gdf.crs != "EPSG:4326":
        plot_gdf = plot_gdf.to_crs(epsg=4326)
    
    # Create map centered on the mean coordinates
    center = (plot_gdf.geometry.y.mean(), plot_gdf.geometry.x.mean())
    m = create_interactive_map(center=center, zoom_start=12)
    
    # Prepare heatmap data
    heat_data = []
    
    for idx, row in plot_gdf.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        
        if weight_column is not None and weight_column in row and pd.notna(row[weight_column]):
            weight = float(row[weight_column])
            heat_data.append([lat, lon, weight])
        else:
            heat_data.append([lat, lon])
    
    # Set default gradient if not provided
    if gradient is None:
        gradient = {
            0.4: 'blue',
            0.6: 'cyan',
            0.8: 'yellow',
            1.0: 'red'
        }
    
    # Add heatmap to map
    plugins.HeatMap(
        heat_data,
        radius=radius,
        blur=blur,
        gradient=gradient
    ).add_to(m)
    
    return m


def create_cluster_map(
    gdf: gpd.GeoDataFrame,
    cluster_column: str = 'location_id',
    popup_columns: Optional[List[str]] = None
) -> folium.Map:
    """
    Create a map with points colored by cluster.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with points
    cluster_column : str, default='location_id'
        Column containing cluster IDs
    popup_columns : Optional[List[str]], default=None
        Columns to include in popups
        
    Returns
    -------
    folium.Map
        Folium map with clustered points
    """
    # Make a copy to avoid modifying the original
    plot_gdf = gdf.copy()
    
    # Ensure the GeoDataFrame has a proper CRS
    if plot_gdf.crs is None:
        plot_gdf.set_crs(epsg=4326, inplace=True)
    elif plot_gdf.crs != "EPSG:4326":
        plot_gdf = plot_gdf.to_crs(epsg=4326)
    
    # Create map centered on the mean coordinates
    center = (plot_gdf.geometry.y.mean(), plot_gdf.geometry.x.mean())
    m = create_interactive_map(center=center, zoom_start=12)
    
    # Add points to map colored by cluster
    if cluster_column in plot_gdf.columns:
        m = add_points_to_map(
            m,
            plot_gdf,
            column=cluster_column,
            popup_columns=popup_columns
        )
    else:
        m = add_points_to_map(
            m,
            plot_gdf,
            popup_columns=popup_columns
        )
    
    return m


def save_map(m: folium.Map, output_path: str) -> str:
    """
    Save a Folium map to an HTML file.
    
    Parameters
    ----------
    m : folium.Map
        Folium map object
    output_path : str
        Path to save the HTML file
        
    Returns
    -------
    str
        Path to the saved HTML file
    """
    m.save(output_path)
    return output_path
