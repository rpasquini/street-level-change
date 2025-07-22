import requests
import os
from PIL import Image
from io import BytesIO
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class StreetViewFetcher:
    def __init__(self, api_key=None):
        """
        Initialize the Street View fetcher with your API key
        Args:
            api_key (str, optional): Your Google Street View Static API key. If None, loads from environment
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv('GOOGLE_STREET_VIEW_API_KEY')
            if api_key is None:
                raise ValueError("No API key provided. Set GOOGLE_STREET_VIEW_API_KEY in .env file or pass api_key parameter")
        
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/streetview"
        self.metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        
    def get_available_dates(self, location):
        """
        Get available dates for Street View imagery at a location
        Args:
            location (tuple or str): Either (lat, lng) or a string address
        Returns:
            list: List of available dates as strings in YYYY-MM format
        """
        params = {'key': self.api_key}
        
        # Handle location input
        if isinstance(location, tuple):
            params['location'] = f"{location[0]},{location[1]}"
        else:
            params['location'] = location
            
        # Make the request
        response = requests.get(self.metadata_url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch metadata: {response.status_code}")
            
        data = response.json()
        if data['status'] != 'OK':
            raise Exception(f"Location not found or no imagery available")
            
        # The date will be in the response if historical imagery is available
        return data.get('date')

    def get_panorama(self, location, heading=0, pitch=0, fov=90, size=(640, 640), date=None):
        """
        Fetch a Street View image for a specific location
        Args:
            location (tuple or str): Either (lat, lng) or a string address
            heading (float): Camera heading in degrees (0 to 360)
            pitch (float): Camera pitch in degrees (-90 to 90)
            fov (float): Field of view in degrees (max 120)
            size (tuple): Image size as (width, height)
            date (str, optional): Date of imagery in YYYY-MM format
        Returns:
            PIL.Image: The street view image
        """
        params = {
            'size': f"{size[0]}x{size[1]}",
            'heading': heading,
            'pitch': pitch,
            'fov': fov,
            'key': self.api_key
        }
        
        # Handle location input
        if isinstance(location, tuple):
            params['location'] = f"{location[0]},{location[1]}"
        else:
            params['location'] = location
            
        # Add date parameter if specified
        if date:
            params['date'] = date
            
        # Make the request
        response = requests.get(self.base_url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch image: {response.status_code}")
            
        # Convert to PIL Image
        img = Image.open(BytesIO(response.content))
        return img
    
    def get_panorama_sequence(self, location, n_images=4, size=(640, 640)):
        """
        Fetch a sequence of images rotating around a point
        Args:
            location: Location coordinates or address
            n_images (int): Number of images to fetch
            size (tuple): Size of each image
        Returns:
            list: List of PIL Images
        """
        images = []
        headings = np.linspace(0, 360, n_images, endpoint=False)
        
        for heading in headings:
            img = self.get_panorama(location, heading=heading, size=size)
            images.append(img)
            
        return images

    def get_panorama_by_id(self, panoid, heading=0, pitch=0, fov=90, size=(640, 640)):
        """
        Fetch a Street View image using a specific panorama ID
        Args:
            panoid (str): The specific panorama ID
            heading (float): Camera heading in degrees (0 to 360)
            pitch (float): Camera pitch in degrees (-90 to 90)
            fov (float): Field of view in degrees (max 120)
            size (tuple): Image size as (width, height)
        Returns:
            PIL.Image: The street view image
        """
        params = {
            'size': f"{size[0]}x{size[1]}",
            'heading': heading,
            'pitch': pitch,
            'fov': fov,
            'pano': panoid,  # Use the panorama ID instead of location
            'key': self.api_key
        }
            
        # Make the request
        response = requests.get(self.base_url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch image: {response.status_code}")
            
        # Convert to PIL Image
        img = Image.open(BytesIO(response.content))
        return img

    def fetch_and_save_specific_view(self, location, output_path, heading=0, pitch=0, fov=90, size=(640, 640)):
        """
        Fetch a specific Street View image and save it
        Args:
            location (tuple): (latitude, longitude)
            output_path (str): Path to save the image
            heading (float): Camera heading in degrees (0 to 360)
            pitch (float): Camera pitch in degrees (-90 to 90)
            fov (float): Field of view in degrees (max 120)
            size (tuple): Image size as (width, height)
        Returns:
            str: Path to the saved image
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Fetch the image
        img = self.get_panorama(
            location=location,
            heading=heading,
            pitch=pitch,
            fov=fov,
            size=size
        )
        
        # Save the image
        img.save(output_path)
        return output_path

def save_panorama_sequence(images, output_dir):
    """
    Save a sequence of panorama images
    Args:
        images (list): List of PIL Images
        output_dir (str): Directory to save images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img in enumerate(images):
        img.save(os.path.join(output_dir, f'panorama_{i:03d}.jpg'))

if __name__ == "__main__":
    # Example usage
    fetcher = StreetViewFetcher()  # Will load API key from .env
    
    # Use the specific panorama ID from the URL
    panoid = "IRjUkZh19iAqvgNSyqOR_w"
    
    # Create test_images directory if it doesn't exist
    os.makedirs("test_images", exist_ok=True)
    
    # Fetch and save specific view using the parameters from the URL
    output_path = os.path.join("test_images", "buenos_aires_view_2015.jpg")
    img = fetcher.get_panorama_by_id(
        panoid=panoid,
        heading=134.12,  # From the URL's yaw parameter
        pitch=5.94,      # From the URL's pitch parameter
        fov=75
    )
    img.save(output_path)
    print(f"Image saved to: {output_path}") 