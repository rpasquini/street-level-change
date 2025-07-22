from src.api.streetview_fetcher import StreetViewFetcher
import os

if __name__ == "__main__":
    # Example usage
    fetcher = StreetViewFetcher(data_dir="data/street_view_images")  # Will load API key from .env
    
    # Use the specific panorama ID from the URL
    panoid = "IRjUkZh19iAqvgNSyqOR_w"
    
    os.makedirs(fetcher.data_dir, exist_ok=True)
    # Fetch and save specific view using the parameters from the URL
    output_path = os.path.join(fetcher.data_dir, "buenos_aires_view_2015.jpg")
    img = fetcher.get_panorama_by_id(
        panoid=panoid,
        heading=134.12,  # From the URL's yaw parameter
        pitch=5.94,      # From the URL's pitch parameter
        fov=75
    )
    img.save(output_path)
    print(f"Image saved to: {output_path}") 