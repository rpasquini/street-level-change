from src.ml.segment_street_view import ImageSegmenter

if __name__ == "__main__":
    # Example usage
    STEGO_CHECKPOINT = "stego/saved_models/cityscapes_vit_base_1.ckpt"  # Path to STEGO checkpoint
    #STEGO_CHECKPOINT = "saved_models/cocostuff27_vit_base_5.ckpt"  # Path to STEGO checkpoint
    # Initialize segmenter
    segmenter = ImageSegmenter(STEGO_CHECKPOINT)
    
    # Example: Segment a single image
    image_path = "data/street_view_images/buenos_aires_view_2015.jpg"  # Replace with your image path
    metrics = segmenter.segment_image(image_path, "data/street_view_images/segmentation_results")
    print("\nSingle Image Metrics:")
    print("Class Distribution:")
    for class_name, stats in metrics['class_distribution'].items():
        print(f"{class_name}: {stats['percentage']:.2f}%")