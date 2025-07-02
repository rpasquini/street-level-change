"""
Feature extraction for Street Level Change Detection.

This module provides functions for extracting features from panorama images
for use in machine learning models.
"""

import os
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
import cv2
from PIL import Image
import torch
from torchvision import transforms, models
import tensorflow as tf

from src.core.panorama import Panorama


def load_image_for_ml(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load and preprocess an image for machine learning.
    
    Parameters
    ----------
    image_path : str
        Path to the image file
    target_size : Tuple[int, int], default=(224, 224)
        Target size for resizing
        
    Returns
    -------
    np.ndarray
        Preprocessed image
    """
    # Load image
    img = cv2.imread(image_path)
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    return img


def extract_features_vgg16(
    image: np.ndarray,
    layer_name: str = 'fc2',
    pretrained: bool = True
) -> np.ndarray:
    """
    Extract features from an image using VGG16.
    
    Parameters
    ----------
    image : np.ndarray
        Image as a numpy array
    layer_name : str, default='fc2'
        Name of the layer to extract features from
    pretrained : bool, default=True
        Whether to use pretrained weights
        
    Returns
    -------
    np.ndarray
        Feature vector
    """
    # Normalize image
    image = image.astype(np.float32) / 255.0
    
    # Convert to PyTorch tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    # Load model
    model = models.vgg16(pretrained=pretrained)
    
    # Create a new model that outputs the desired layer
    if layer_name == 'fc2':
        new_model = torch.nn.Sequential(*list(model.children())[:-1])
    else:
        # For other layers, you would need to modify this
        new_model = model
    
    # Extract features
    with torch.no_grad():
        features = new_model(image_tensor)
    
    # Convert to numpy array
    return features.numpy().flatten()


def extract_features_resnet50(
    image: np.ndarray,
    pretrained: bool = True
) -> np.ndarray:
    """
    Extract features from an image using ResNet50.
    
    Parameters
    ----------
    image : np.ndarray
        Image as a numpy array
    pretrained : bool, default=True
        Whether to use pretrained weights
        
    Returns
    -------
    np.ndarray
        Feature vector
    """
    # Normalize image
    image = image.astype(np.float32) / 255.0
    
    # Convert to PyTorch tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    # Load model
    model = models.resnet50(pretrained=pretrained)
    
    # Remove the last fully connected layer
    new_model = torch.nn.Sequential(*list(model.children())[:-1])
    
    # Extract features
    with torch.no_grad():
        features = new_model(image_tensor)
    
    # Convert to numpy array
    return features.numpy().flatten()


def extract_features_mobilenet(
    image: np.ndarray,
    pretrained: bool = True
) -> np.ndarray:
    """
    Extract features from an image using MobileNetV2.
    
    Parameters
    ----------
    image : np.ndarray
        Image as a numpy array
    pretrained : bool, default=True
        Whether to use pretrained weights
        
    Returns
    -------
    np.ndarray
        Feature vector
    """
    # Normalize image
    image = image.astype(np.float32) / 255.0
    
    # Convert to PyTorch tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    # Load model
    model = models.mobilenet_v2(pretrained=pretrained)
    
    # Remove the last fully connected layer
    new_model = torch.nn.Sequential(*list(model.children())[:-1])
    
    # Extract features
    with torch.no_grad():
        features = new_model(image_tensor)
    
    # Convert to numpy array
    return features.numpy().flatten()


def extract_features_tensorflow(
    image: np.ndarray,
    model_name: str = 'MobileNetV2',
    pretrained: bool = True
) -> np.ndarray:
    """
    Extract features from an image using a TensorFlow model.
    
    Parameters
    ----------
    image : np.ndarray
        Image as a numpy array
    model_name : str, default='MobileNetV2'
        Name of the model to use
    pretrained : bool, default=True
        Whether to use pretrained weights
        
    Returns
    -------
    np.ndarray
        Feature vector
    """
    # Normalize image
    image = image.astype(np.float32) / 255.0
    
    # Expand dimensions for batch
    image = np.expand_dims(image, axis=0)
    
    # Load model
    if model_name == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet' if pretrained else None
        )
    elif model_name == 'ResNet50':
        base_model = tf.keras.applications.ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet' if pretrained else None
        )
    elif model_name == 'VGG16':
        base_model = tf.keras.applications.VGG16(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet' if pretrained else None
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Create a new model that outputs the desired layer
    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.output
    )
    
    # Extract features
    features = model.predict(image)
    
    # Flatten and return
    return features.flatten()


def extract_features_from_panorama(
    panorama: Panorama,
    image_manager,
    zoom: int = 3,
    model: str = 'resnet50',
    framework: str = 'pytorch'
) -> np.ndarray:
    """
    Extract features from a panorama image.
    
    Parameters
    ----------
    panorama : Panorama
        Panorama object
    image_manager : PanoramaImageManager
        Image manager for loading images
    zoom : int, default=3
        Zoom level (0-5, where 5 is highest resolution)
    model : str, default='resnet50'
        Model to use for feature extraction
    framework : str, default='pytorch'
        Framework to use ('pytorch' or 'tensorflow')
        
    Returns
    -------
    np.ndarray
        Feature vector
    """
    # Load image
    image_path = image_manager.load_image(panorama, zoom)
    
    if image_path is None:
        # Try to download the image
        image_path = image_manager.download_image(panorama, zoom)
        
        if image_path is None:
            raise ValueError(f"Could not load or download image for panorama {panorama.pano_id}")
    
    # Load and preprocess image
    image = load_image_for_ml(image_path)
    
    # Extract features
    if framework == 'pytorch':
        if model == 'vgg16':
            return extract_features_vgg16(image)
        elif model == 'resnet50':
            return extract_features_resnet50(image)
        elif model == 'mobilenet':
            return extract_features_mobilenet(image)
        else:
            raise ValueError(f"Unknown PyTorch model: {model}")
    elif framework == 'tensorflow':
        return extract_features_tensorflow(image, model_name=model)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def extract_features_batch(
    panoramas: List[Panorama],
    image_manager,
    zoom: int = 3,
    model: str = 'resnet50',
    framework: str = 'pytorch',
    max_workers: int = 10
) -> Dict[str, np.ndarray]:
    """
    Extract features from multiple panorama images.
    
    Parameters
    ----------
    panoramas : List[Panorama]
        List of panorama objects
    image_manager : PanoramaImageManager
        Image manager for loading images
    zoom : int, default=3
        Zoom level (0-5, where 5 is highest resolution)
    model : str, default='resnet50'
        Model to use for feature extraction
    framework : str, default='pytorch'
        Framework to use ('pytorch' or 'tensorflow')
    max_workers : int, default=10
        Maximum number of parallel workers
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping panorama IDs to feature vectors
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    features = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                extract_features_from_panorama,
                panorama,
                image_manager,
                zoom,
                model,
                framework
            ): panorama.pano_id
            for panorama in panoramas
        }
        
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Extracting features using {model}"
        ):
            pano_id = futures[future]
            try:
                feature_vector = future.result()
                features[pano_id] = feature_vector
            except Exception as e:
                print(f"Error extracting features for panorama {pano_id}: {e}")
    
    return features


def create_feature_dataset(
    panoramas: List[Panorama],
    image_manager,
    zoom: int = 3,
    model: str = 'resnet50',
    framework: str = 'pytorch',
    max_workers: int = 10
) -> Tuple[np.ndarray, List[str]]:
    """
    Create a feature dataset from panorama images.
    
    Parameters
    ----------
    panoramas : List[Panorama]
        List of panorama objects
    image_manager : PanoramaImageManager
        Image manager for loading images
    zoom : int, default=3
        Zoom level (0-5, where 5 is highest resolution)
    model : str, default='resnet50'
        Model to use for feature extraction
    framework : str, default='pytorch'
        Framework to use ('pytorch' or 'tensorflow')
    max_workers : int, default=10
        Maximum number of parallel workers
        
    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Feature matrix and list of panorama IDs
    """
    # Extract features
    features_dict = extract_features_batch(
        panoramas,
        image_manager,
        zoom,
        model,
        framework,
        max_workers
    )
    
    # Convert to matrix
    pano_ids = list(features_dict.keys())
    feature_matrix = np.array([features_dict[pano_id] for pano_id in pano_ids])
    
    return feature_matrix, pano_ids
