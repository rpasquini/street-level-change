"""
Google Street View API interactions.

This module provides functions for interacting with the Google Street View API,
including searching for panoramas and downloading images.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from src.core.panorama import Panorama, PanoramaCollection
from streetview import get_panorama_meta
from PIL import Image

def search_panoramas(
    lat: float, 
    lon: float, 
) -> List[Panorama]:
    """
    Search for panoramas near a location.
    
    Parameters
    ----------
    lat : float
        Latitude coordinate
    lon : float
        Longitude coordinate
        
    Returns
    -------
    List[Panorama]
        List of Panorama objects found near the location
    """
    from streetview import search_panoramas as sv_search_panoramas
    
    # Call the streetview library function
    panos_found = sv_search_panoramas(lat=lat, lon=lon)
    
    # Convert to our Panorama objects
    panoramas = []
    for pano in panos_found:
        panorama = Panorama(
            pano_id=pano.pano_id,
            lat=pano.lat,
            lon=pano.lon,
            date=pano.date
        )
        panoramas.append(panorama)
    
    return panoramas


def get_panoramas_for_point(
    point_geometry,
) -> PanoramaCollection:
    """
    Get panoramas for a specific point.
    
    Parameters
    ----------
    point_geometry : shapely.geometry.Point
        Point geometry to search near
        
    Returns
    -------
    PanoramaCollection
        Collection of panoramas found near the point
    """
    lat, lon = point_geometry.y, point_geometry.x
    try:
        panoramas = search_panoramas(lat=lat, lon=lon)
    except Exception as e:
        print(f"Error getting panoramas for lat {lat}, lon {lon}: {e}")
        return PanoramaCollection()
        
    return PanoramaCollection(panoramas)


def get_panoramas_for_points(
    points_gdf: gpd.GeoDataFrame,
    max_workers: int = 20,
    verbose: bool = False
) -> PanoramaCollection:
    """
    Get panoramas for multiple points using parallel processing.
    
    Parameters
    ----------
    points_gdf : gpd.GeoDataFrame
        GeoDataFrame containing point geometries
    max_workers : int, default=10
        Maximum number of parallel workers
    verbose : bool, default=False
        Whether to display progress information
        
    Returns
    -------
    PanoramaCollection
        Collection of all panoramas found
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    all_panoramas = PanoramaCollection()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(get_panoramas_for_point, geom)
            for geom in points_gdf.geometry
        ]
        
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Fetching Panoramas",
            disable=not verbose
        ):
            result = future.result()
            if result and len(result) > 0:
                for panorama in result:
                    all_panoramas.add(panorama)
    
    return all_panoramas


def download_panorama_image(
    pano_id: str,
    zoom: int = 3,
    output_path: Optional[str] = None
) -> Optional[Image]:
    """
    Download an image for a panorama.
    
    Parameters
    ----------
    pano_id : str
        Panorama ID
    zoom : int, default=3
        Zoom level (0-5, where 5 is highest resolution)
    output_path : Optional[str], default=None
        Path to save the image. If None, image is not saved to disk.
        
    Returns
    -------
    Optional[str]
        Image object if successful, otherwise None
    """
    from streetview import get_panorama
    
    try:
        pano = get_panorama(
            pano_id,
            zoom=zoom,
        )
    except Exception as e:
        print(f"Error downloading panorama {pano_id}: {e}")
        return None

    if output_path:
        pano.save(output_path)
    return pano

def get_panorama_metadata(
    pano_id: str,
    api_key: str,
) -> Dict[str, Any]:
    """
    Get metadata for a panorama.

    Quota: This function doesn't use up any quota or charge on your API_KEY.

    Endpoint documented at:
    https://developers.google.com/maps/documentation/streetview/metadata
    
    Parameters
    ----------
    pano_id : str
        Panorama ID
    api_key : str
        Google Street View API key
    
    Returns
    -------
    Dict[str, Any]
        Metadata for the panorama
    """
    return dict(get_panorama_meta(pano_id, api_key))

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def fetch_metadata_parallel(pano_ids, api_key, max_workers=10):
    """
    Fetch panorama metadata in parallel.
    
    Parameters
    ----------
    pano_ids : list
        List of panorama IDs.
    api_key : str
        API key for the metadata function.
    max_workers : int
        Number of parallel workers (default=10).
        
    Returns
    -------
    metadata_dates : dict
        Mapping pano_id -> date.
    failed_set : list
        List of pano_ids that failed.
    """

    metadata_dates = {}
    failed_set = []

    def task(pid):
        """Wrapper to fetch metadata for one pano_id."""
        try:
            metadata = get_panorama_metadata(pid, api_key)
            return pid, metadata['date'], None
        except Exception as e:
            return pid, None, e

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task, pid): pid for pid in pano_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Metadata"):
            pid, date, error = future.result()
            if error:
                failed_set.append(pid)
            else:
                metadata_dates[pid] = date

    return metadata_dates, failed_set
