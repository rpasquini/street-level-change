from src.ml.segment_street_view import ImageSegmenter
import os
from os.path import join

def download_model(model_path, model_name):
    import wget
    saved_model_url_root = "https://marhamilresearch4.blob.core.windows.net/stego-public/saved_models/"
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    if not os.path.exists(join(model_path, model_name)):
        print(f"Downloading {model_name} to {model_path}")
        wget.download(
            saved_model_url_root + model_name,
            join(model_path, model_name)
        )

if __name__ == "__main__":
    # Example usage
    STEGO_MODELS_FOLDER = "/src/ml/stego/saved_models"
    STEGO_MODEL_NAME = "cocostuff27_vit_base_5.ckpt"  # Path to STEGO checkpoint
    download_model(STEGO_MODELS_FOLDER, STEGO_MODEL_NAME)
    model = join(STEGO_MODELS_FOLDER, STEGO_MODEL_NAME)
    # Initialize segmenter
    segmenter = ImageSegmenter(model)
    
    # Example: Segment a single image
    image_path = "/data/street_view_images/buenos_aires_view_2015.jpg"  # Replace with your image path
    metrics = segmenter.segment_image(image_path, "/data/street_view_images/segmentation_results")
    print("\nSingle Image Metrics:")
    print("Class Distribution:")
    for class_name, stats in metrics['class_distribution'].items():
        print(f"{class_name}: {stats['percentage']:.2f}%")