"""
Panorama data model for Street Level Change Detection.

This module provides a standardized data model for Google Street View panoramas,
including metadata handling, geometry operations, and collection management.
"""

from datetime import datetime
from typing import Optional, List, Dict, Union, Any, Tuple
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon


class Panorama:
    """
    Represents a Google Street View panorama with metadata.
    
    This class provides a standardized representation of panorama data,
    including location information, dates, and additional metadata.
    It also provides utility methods for working with panorama data.
    """
    
    def __init__(
        self,
        pano_id: str,
        lat: float,
        lon: float,
        date: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a panorama object.
        
        Parameters
        ----------
        pano_id : str
            Unique identifier for the panorama
        lat : float
            Latitude coordinate
        lon : float
            Longitude coordinate
        date : Optional[Union[str, datetime]], default=None
            Date when the panorama was captured
        metadata : Optional[Dict[str, Any]], default=None
            Additional metadata for the panorama
        """
        self.pano_id = pano_id
        self.lat = lat
        self.lon = lon
        self.date = date
        
        self.metadata = metadata or {}
        
    @property
    def geometry(self) -> Point:
        """
        Return a shapely Point geometry representing the panorama location.
        
        Returns
        -------
        Point
            Shapely Point geometry with (lon, lat) coordinates
        """
        return Point(self.lon, self.lat)
    
    @property
    def coordinates(self) -> Tuple[float, float]:
        """
        Return the coordinates as a (lat, lon) tuple.
        
        Returns
        -------
        Tuple[float, float]
            (latitude, longitude) coordinates
        """
        return (self.lat, self.lon)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the panorama to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the panorama
        """
        result = {
            'pano_id': self.pano_id,
            'lat': self.lat,
            'lon': self.lon,
            'date': self.date
        }
        
        result.update(self.metadata)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Panorama':
        """
        Create a Panorama object from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing panorama data
            
        Returns
        -------
        Panorama
            New Panorama object
        """
        # Extract core fields
        pano_id = data.pop('pano_id')
        lat = data.pop('lat')
        lon = data.pop('lon')
        date = data.pop('date')
        
        return cls(pano_id=pano_id, lat=lat, lon=lon, date=date, metadata=data)
    
    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> 'Panorama':
        """
        Create a Panorama object from a Street View API response.
        
        Parameters
        ----------
        response : Dict[str, Any]
            API response dictionary
            
        Returns
        -------
        Panorama
            New Panorama object
        
        Notes
        -----
        This method assumes the response follows the format of the
        streetview library's search_panoramas function.
        """
        return cls(
            pano_id=response.get('pano_id'),
            lat=response.get('lat'),
            lon=response.get('lon'),
            date=response.get('date'),
            metadata={k: v for k, v in response.items() 
                     if k not in ('pano_id', 'lat', 'lon', 'date')}
        )


class PanoramaCollection:
    """
    A collection of Panorama objects with utilities for bulk operations.
    
    This class provides methods for working with collections of panoramas,
    including loading from and saving to various formats, filtering, and
    conversion to other data structures like DataFrames.
    """
    
    def __init__(self, panoramas: Optional[List[Panorama]] = None):
        """
        Initialize a collection of panoramas.
        
        Parameters
        ----------
        panoramas : Optional[List[Panorama]], default=None
            List of Panorama objects
        """
        self.panoramas = panoramas or []
    
    def __len__(self) -> int:
        """Return the number of panoramas in the collection."""
        return len(self.panoramas)
    
    def __getitem__(self, idx) -> Union[Panorama, List[Panorama]]:
        """Get panorama(s) by index."""
        return self.panoramas[idx]
    
    def add(self, panorama: Panorama) -> None:
        """
        Add a panorama to the collection.
        
        Parameters
        ----------
        panorama : Panorama
            Panorama to add
        """
        self.panoramas.append(panorama)
    
    def filter(self, predicate) -> 'PanoramaCollection':
        """
        Filter panoramas based on a predicate function.
        
        Parameters
        ----------
        predicate : callable
            Function that takes a Panorama and returns a boolean
            
        Returns
        -------
        PanoramaCollection
            New collection with filtered panoramas
        """
        filtered = [p for p in self.panoramas if predicate(p)]
        return PanoramaCollection(filtered)
    
    def clean(self, boundary: Union[gpd.GeoDataFrame, Polygon, MultiPolygon]):
        if isinstance(boundary, gpd.GeoDataFrame):
            boundary = boundary.union_all()

        gdf = self.to_geodataframe()
        return gdf[gdf.intersects(boundary)].drop_duplicates()

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the collection to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing panorama data
        """
        if not self.panoramas:
            return pd.DataFrame()
        
        return pd.DataFrame([p.to_dict() for p in self.panoramas])
    
    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Convert the collection to a GeoPandas GeoDataFrame.
        
        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing panorama data with geometry column
        """
        df = self.to_dataframe()
        
        if df.empty:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        geometries = [p.geometry for p in self.panoramas]
        return gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'PanoramaCollection':
        """
        Create a PanoramaCollection from a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing panorama data
            
        Returns
        -------
        PanoramaCollection
            New PanoramaCollection object
        
        Notes
        -----
        The DataFrame must contain at least 'pano_id', 'lat', and 'lon' columns.
        """
        if df.empty:
            return cls()
        
        panoramas = []
        for _, row in df.iterrows():
            data = row.to_dict()
            panoramas.append(Panorama.from_dict(data))
        
        return cls(panoramas)
    
    @classmethod
    def from_geodataframe(cls, gdf: gpd.GeoDataFrame) -> 'PanoramaCollection':
        """
        Create a PanoramaCollection from a GeoDataFrame.
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing panorama data
            
        Returns
        -------
        PanoramaCollection
            New PanoramaCollection object
        
        Notes
        -----
        If the GeoDataFrame has a geometry column, coordinates will be extracted from it.
        Otherwise, it must contain 'lat' and 'lon' columns.
        """
        if gdf.empty:
            return cls()
        
        panoramas = []
        for _, row in gdf.iterrows():
            data = row.to_dict()
            
            # Handle geometry column
            if 'geometry' in data:
                point = data.pop('geometry')
                if 'lat' not in data:
                    data['lat'] = point.y
                if 'lon' not in data:
                    data['lon'] = point.x
            
            panoramas.append(Panorama.from_dict(data))
        
        return cls(panoramas)
    
    @classmethod
    def from_csv(cls, file_path: str) -> 'PanoramaCollection':
        """
        Load panoramas from a CSV file.
        
        Parameters
        ----------
        file_path : str
            Path to the CSV file
            
        Returns
        -------
        PanoramaCollection
            New PanoramaCollection object
        """
        try:
            df = pd.read_csv(file_path)
            return cls.from_dataframe(df)
        except Exception as e:
            raise ValueError(f"Error loading panoramas from CSV: {e}")
    
    def to_csv(self, file_path: str) -> None:
        """
        Save panoramas to a CSV file.
        
        Parameters
        ----------
        file_path : str
            Path to save the CSV file
        """
        df = self.to_dataframe()
        df.to_csv(file_path, index=False)
