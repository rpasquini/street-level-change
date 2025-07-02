"""
Image and metadata storage management for Street Level Change Detection.

This module provides classes and functions for managing the storage of
panorama images and metadata, including downloading, organizing, and retrieving.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
from pathlib import Path
import shutil

from src.core.panorama import Panorama, PanoramaCollection


class PanoramaImageManager:
    """
    Manages downloading and storing panorama images.
    
    This class provides methods for downloading, organizing, and retrieving
    panorama images and metadata.
    """
    
    def __init__(self, storage_dir: str):
        """
        Initialize a PanoramaImageManager.
        
        Parameters
        ----------
        storage_dir : str
            Base directory for storing panorama images and metadata
        """
        self.storage_dir = storage_dir
        self._ensure_directory_exists(storage_dir)
    
    def _ensure_directory_exists(self, directory: str) -> None:
        """
        Ensure a directory exists, creating it if necessary.
        
        Parameters
        ----------
        directory : str
            Directory path to ensure exists
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    def get_image_path(self, panorama: Panorama, zoom: int = 3) -> str:
        """
        Get the path where a panorama image should be stored.
        
        Parameters
        ----------
        panorama : Panorama
            Panorama object
        zoom : int, default=3
            Zoom level (0-5, where 5 is highest resolution)
            
        Returns
        -------
        str
            Path where the image should be stored
        """
        # Create a directory structure based on date and location
        year_month = "unknown_date"
        if panorama.date:
            year_month = panorama.date.strftime("%Y-%m")
        
        # Round coordinates to create location-based directory
        lat_rounded = round(panorama.lat, 3)
        lon_rounded = round(panorama.lon, 3)
        location_dir = f"{lat_rounded}_{lon_rounded}"
        
        # Create the directory path
        image_dir = os.path.join(self.storage_dir, year_month, location_dir)
        self._ensure_directory_exists(image_dir)
        
        # Create the image filename
        image_filename = f"{panorama.pano_id}_zoom{zoom}.jpg"
        
        return os.path.join(image_dir, image_filename)
    
    def download_image(
        self, 
        panorama: Panorama, 
        zoom: int = 3,
        force_download: bool = False
    ) -> Optional[str]:
        """
        Download an image for a panorama.
        
        Parameters
        ----------
        panorama : Panorama
            Panorama object
        zoom : int, default=3
            Zoom level (0-5, where 5 is highest resolution)
        force_download : bool, default=False
            Whether to download the image even if it already exists
            
        Returns
        -------
        Optional[str]
            Path to the downloaded image, or None if download failed
        """
        from src.api.streetview import download_panorama_image
        
        # Get the path where the image should be stored
        image_path = self.get_image_path(panorama, zoom)
        
        # Check if the image already exists
        if os.path.exists(image_path) and not force_download:
            return image_path
        
        # Download the image
        return download_panorama_image(panorama, zoom, image_path)
    
    def download_images(
        self,
        panoramas: Union[List[Panorama], PanoramaCollection],
        zoom: int = 3,
        force_download: bool = False,
        max_workers: int = 10
    ) -> Dict[str, Optional[str]]:
        """
        Download images for multiple panoramas using parallel processing.
        
        Parameters
        ----------
        panoramas : Union[List[Panorama], PanoramaCollection]
            List or collection of panoramas
        zoom : int, default=3
            Zoom level (0-5, where 5 is highest resolution)
        force_download : bool, default=False
            Whether to download images even if they already exist
        max_workers : int, default=10
            Maximum number of parallel workers
            
        Returns
        -------
        Dict[str, Optional[str]]
            Dictionary mapping panorama IDs to image paths
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        # Convert PanoramaCollection to list if needed
        if isinstance(panoramas, PanoramaCollection):
            panoramas = panoramas.panoramas
        
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.download_image, pano, zoom, force_download): pano.pano_id
                for pano in panoramas
            }
            
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Downloading panorama images"
            ):
                pano_id = futures[future]
                try:
                    image_path = future.result()
                    results[pano_id] = image_path
                except Exception as e:
                    print(f"Error downloading panorama {pano_id}: {e}")
                    results[pano_id] = None
        
        return results
    
    def load_image(self, panorama: Panorama, zoom: int = 3) -> Optional[str]:
        """
        Load an image for a panorama.
        
        Parameters
        ----------
        panorama : Panorama
            Panorama object
        zoom : int, default=3
            Zoom level (0-5, where 5 is highest resolution)
            
        Returns
        -------
        Optional[str]
            Path to the image if it exists, otherwise None
        """
        image_path = self.get_image_path(panorama, zoom)
        
        if os.path.exists(image_path):
            return image_path
        
        return None
    
    def save_metadata(self, panorama: Panorama) -> str:
        """
        Save metadata for a panorama.
        
        Parameters
        ----------
        panorama : Panorama
            Panorama object
            
        Returns
        -------
        str
            Path to the saved metadata file
        """
        # Get the directory where the metadata should be stored
        image_path = self.get_image_path(panorama)
        metadata_path = f"{os.path.splitext(image_path)[0]}.json"
        
        # Save the metadata as JSON
        with open(metadata_path, 'w') as f:
            json.dump(panorama.to_dict(), f, indent=2)
        
        return metadata_path
    
    def load_metadata(self, panorama_id: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata for a panorama.
        
        Parameters
        ----------
        panorama_id : str
            Panorama ID
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Metadata dictionary if found, otherwise None
        """
        # Search for metadata file
        for root, _, files in os.walk(self.storage_dir):
            for file in files:
                if file.startswith(panorama_id) and file.endswith('.json'):
                    metadata_path = os.path.join(root, file)
                    with open(metadata_path, 'r') as f:
                        return json.load(f)
        
        return None
    
    def get_all_panoramas(self) -> PanoramaCollection:
        """
        Get all panoramas in the storage directory.
        
        Returns
        -------
        PanoramaCollection
            Collection of all panoramas
        """
        panoramas = []
        
        # Walk through the storage directory
        for root, _, files in os.walk(self.storage_dir):
            for file in files:
                if file.endswith('.json'):
                    metadata_path = os.path.join(root, file)
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        panoramas.append(Panorama.from_dict(metadata))
        
        return PanoramaCollection(panoramas)
    
    def clear_storage(self) -> None:
        """
        Clear all images and metadata from the storage directory.
        """
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)
            os.makedirs(self.storage_dir, exist_ok=True)


class PanoramaDatabase:
    """
    Simple database for storing and retrieving panorama metadata.
    
    This class provides methods for storing, retrieving, and querying
    panorama metadata using a simple file-based database.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize a PanoramaDatabase.
        
        Parameters
        ----------
        db_path : str
            Path to the database file
        """
        self.db_path = db_path
        self._ensure_directory_exists(os.path.dirname(db_path))
        
        # Initialize the database if it doesn't exist
        if not os.path.exists(db_path):
            self._initialize_db()
    
    def _ensure_directory_exists(self, directory: str) -> None:
        """
        Ensure a directory exists, creating it if necessary.
        
        Parameters
        ----------
        directory : str
            Directory path to ensure exists
        """
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_db(self) -> None:
        """
        Initialize an empty database.
        """
        empty_df = pd.DataFrame(columns=[
            'pano_id', 'lat', 'lon', 'date', 'metadata'
        ])
        empty_df.to_csv(self.db_path, index=False)
    
    def add_panorama(self, panorama: Panorama) -> None:
        """
        Add a panorama to the database.
        
        Parameters
        ----------
        panorama : Panorama
            Panorama to add
        """
        # Load the database
        df = pd.read_csv(self.db_path)
        
        # Check if the panorama already exists
        if panorama.pano_id in df['pano_id'].values:
            # Update the existing record
            idx = df[df['pano_id'] == panorama.pano_id].index[0]
            df.at[idx, 'lat'] = panorama.lat
            df.at[idx, 'lon'] = panorama.lon
            df.at[idx, 'date'] = panorama.date.isoformat() if panorama.date else None
            df.at[idx, 'metadata'] = json.dumps(panorama.metadata)
        else:
            # Add a new record
            new_row = {
                'pano_id': panorama.pano_id,
                'lat': panorama.lat,
                'lon': panorama.lon,
                'date': panorama.date.isoformat() if panorama.date else None,
                'metadata': json.dumps(panorama.metadata)
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save the database
        df.to_csv(self.db_path, index=False)
    
    def add_panoramas(self, panoramas: Union[List[Panorama], PanoramaCollection]) -> None:
        """
        Add multiple panoramas to the database.
        
        Parameters
        ----------
        panoramas : Union[List[Panorama], PanoramaCollection]
            List or collection of panoramas to add
        """
        # Convert PanoramaCollection to list if needed
        if isinstance(panoramas, PanoramaCollection):
            panoramas = panoramas.panoramas
        
        # Load the database
        df = pd.read_csv(self.db_path)
        
        # Prepare new records
        new_rows = []
        for panorama in panoramas:
            new_row = {
                'pano_id': panorama.pano_id,
                'lat': panorama.lat,
                'lon': panorama.lon,
                'date': panorama.date.isoformat() if panorama.date else None,
                'metadata': json.dumps(panorama.metadata)
            }
            new_rows.append(new_row)
        
        # Add new records
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        
        # Remove duplicates (keep the last occurrence)
        df = df.drop_duplicates(subset=['pano_id'], keep='last')
        
        # Save the database
        df.to_csv(self.db_path, index=False)
    
    def get_panorama(self, pano_id: str) -> Optional[Panorama]:
        """
        Get a panorama from the database.
        
        Parameters
        ----------
        pano_id : str
            Panorama ID
            
        Returns
        -------
        Optional[Panorama]
            Panorama if found, otherwise None
        """
        # Load the database
        df = pd.read_csv(self.db_path)
        
        # Find the panorama
        if pano_id in df['pano_id'].values:
            row = df[df['pano_id'] == pano_id].iloc[0]
            
            # Parse the date
            date = None
            if pd.notna(row['date']):
                try:
                    date = datetime.fromisoformat(row['date'])
                except ValueError:
                    pass
            
            # Parse the metadata
            metadata = {}
            if pd.notna(row['metadata']):
                try:
                    metadata = json.loads(row['metadata'])
                except json.JSONDecodeError:
                    pass
            
            return Panorama(
                pano_id=row['pano_id'],
                lat=row['lat'],
                lon=row['lon'],
                date=date,
                metadata=metadata
            )
        
        return None
    
    def get_all_panoramas(self) -> PanoramaCollection:
        """
        Get all panoramas from the database.
        
        Returns
        -------
        PanoramaCollection
            Collection of all panoramas
        """
        # Load the database
        df = pd.read_csv(self.db_path)
        
        panoramas = []
        for _, row in df.iterrows():
            # Parse the date
            date = None
            if pd.notna(row['date']):
                try:
                    date = datetime.fromisoformat(row['date'])
                except ValueError:
                    pass
            
            # Parse the metadata
            metadata = {}
            if pd.notna(row['metadata']):
                try:
                    metadata = json.loads(row['metadata'])
                except json.JSONDecodeError:
                    pass
            
            panorama = Panorama(
                pano_id=row['pano_id'],
                lat=row['lat'],
                lon=row['lon'],
                date=date,
                metadata=metadata
            )
            panoramas.append(panorama)
        
        return PanoramaCollection(panoramas)
    
    def query_panoramas(self, **kwargs) -> PanoramaCollection:
        """
        Query panoramas from the database.
        
        Parameters
        ----------
        **kwargs
            Query parameters (e.g., lat_min, lat_max, lon_min, lon_max, date_min, date_max)
            
        Returns
        -------
        PanoramaCollection
            Collection of panoramas matching the query
        """
        # Load the database
        df = pd.read_csv(self.db_path)
        
        # Apply filters
        if 'lat_min' in kwargs:
            df = df[df['lat'] >= kwargs['lat_min']]
        if 'lat_max' in kwargs:
            df = df[df['lat'] <= kwargs['lat_max']]
        if 'lon_min' in kwargs:
            df = df[df['lon'] >= kwargs['lon_min']]
        if 'lon_max' in kwargs:
            df = df[df['lon'] <= kwargs['lon_max']]
        
        # Filter by date
        if 'date_min' in kwargs or 'date_max' in kwargs:
            # Convert date column to datetime
            date_series = pd.to_datetime(df['date'], errors='coerce')
            
            if 'date_min' in kwargs:
                date_min = pd.to_datetime(kwargs['date_min'])
                df = df[date_series >= date_min]
            
            if 'date_max' in kwargs:
                date_max = pd.to_datetime(kwargs['date_max'])
                df = df[date_series <= date_max]
        
        # Convert filtered dataframe to PanoramaCollection
        panoramas = []
        for _, row in df.iterrows():
            # Parse the date
            date = None
            if pd.notna(row['date']):
                try:
                    date = datetime.fromisoformat(row['date'])
                except ValueError:
                    pass
            
            # Parse the metadata
            metadata = {}
            if pd.notna(row['metadata']):
                try:
                    metadata = json.loads(row['metadata'])
                except json.JSONDecodeError:
                    pass
            
            panorama = Panorama(
                pano_id=row['pano_id'],
                lat=row['lat'],
                lon=row['lon'],
                date=date,
                metadata=metadata
            )
            panoramas.append(panorama)
        
        return PanoramaCollection(panoramas)
