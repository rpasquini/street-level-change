"""
Image processing functions for Street Level Change Detection.

This module provides functions for processing panorama images,
including loading, transforming, and feature extraction.
"""

import os
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
import cv2
from PIL import Image

from src.core.panorama import Panorama


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from a file.
    
    Parameters
    ----------
    image_path : str
        Path to the image file
        
    Returns
    -------
    np.ndarray
        Image as a numpy array
    """
    return cv2.imread(image_path)


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to RGB color space.
    
    Parameters
    ----------
    image : np.ndarray
        Image as a numpy array
        
    Returns
    -------
    np.ndarray
        RGB image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize an image.
    
    Parameters
    ----------
    image : np.ndarray
        Image to resize
    width : Optional[int], default=None
        Target width
    height : Optional[int], default=None
        Target height
    keep_aspect_ratio : bool, default=True
        Whether to maintain the aspect ratio
        
    Returns
    -------
    np.ndarray
        Resized image
    """
    if width is None and height is None:
        return image
    
    h, w = image.shape[:2]
    
    if keep_aspect_ratio:
        if width is None:
            aspect_ratio = height / h
            width = int(w * aspect_ratio)
        elif height is None:
            aspect_ratio = width / w
            height = int(h * aspect_ratio)
        else:
            # Both width and height are specified, use the smaller scale
            scale_w = width / w
            scale_h = height / h
            scale = min(scale_w, scale_h)
            width = int(w * scale)
            height = int(h * scale)
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def crop_image(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int
) -> np.ndarray:
    """
    Crop an image.
    
    Parameters
    ----------
    image : np.ndarray
        Image to crop
    x : int
        X-coordinate of the top-left corner
    y : int
        Y-coordinate of the top-left corner
    width : int
        Width of the crop
    height : int
        Height of the crop
        
    Returns
    -------
    np.ndarray
        Cropped image
    """
    return image[y:y+height, x:x+width]


def extract_features_sift(image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Extract SIFT features from an image.
    
    Parameters
    ----------
    image : np.ndarray
        Image to extract features from
        
    Returns
    -------
    Tuple[List[cv2.KeyPoint], np.ndarray]
        Keypoints and descriptors
    """
    # Convert to grayscale for feature extraction
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors


def match_features(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    ratio_threshold: float = 0.7
) -> List[Tuple[int, int]]:
    """
    Match features between two images using the ratio test.
    
    Parameters
    ----------
    descriptors1 : np.ndarray
        Descriptors from the first image
    descriptors2 : np.ndarray
        Descriptors from the second image
    ratio_threshold : float, default=0.7
        Ratio test threshold
        
    Returns
    -------
    List[Tuple[int, int]]
        List of matches (index in descriptors1, index in descriptors2)
    """
    # Create FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append((m.queryIdx, m.trainIdx))
    
    return good_matches


def compute_image_similarity(
    image1: np.ndarray,
    image2: np.ndarray,
    method: str = 'sift'
) -> float:
    """
    Compute similarity between two images.
    
    Parameters
    ----------
    image1 : np.ndarray
        First image
    image2 : np.ndarray
        Second image
    method : str, default='sift'
        Similarity method ('sift', 'ssim', or 'histogram')
        
    Returns
    -------
    float
        Similarity score (0-1, where 1 is most similar)
    """
    if method == 'sift':
        # Extract SIFT features
        keypoints1, descriptors1 = extract_features_sift(image1)
        keypoints2, descriptors2 = extract_features_sift(image2)
        
        if descriptors1 is None or descriptors2 is None:
            return 0.0
        
        # Match features
        matches = match_features(descriptors1, descriptors2)
        
        # Compute similarity score
        if len(keypoints1) == 0 or len(keypoints2) == 0:
            return 0.0
        
        return len(matches) / min(len(keypoints1), len(keypoints2))
    
    elif method == 'ssim':
        # Convert to grayscale
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = image1
        
        if len(image2.shape) == 3:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = image2
        
        # Resize to the same dimensions
        h1, w1 = gray1.shape
        h2, w2 = gray2.shape
        
        if h1 != h2 or w1 != w2:
            gray2 = cv2.resize(gray2, (w1, h1))
        
        # Compute SSIM
        from skimage.metrics import structural_similarity
        return structural_similarity(gray1, gray2)
    
    elif method == 'histogram':
        # Convert to HSV
        if len(image1.shape) == 3:
            hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
        else:
            # For grayscale images, use them directly
            return cv2.compareHist(
                cv2.calcHist([image1], [0], None, [256], [0, 256]),
                cv2.calcHist([image2], [0], None, [256], [0, 256]),
                cv2.HISTCMP_CORREL
            )
        
        # Calculate histograms
        hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Compare histograms
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def detect_changes(
    image1: np.ndarray,
    image2: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, np.ndarray]:
    """
    Detect changes between two images.
    
    Parameters
    ----------
    image1 : np.ndarray
        First image
    image2 : np.ndarray
        Second image
    threshold : float, default=0.5
        Threshold for change detection
        
    Returns
    -------
    Tuple[float, np.ndarray]
        Change score and change mask
    """
    # Convert to grayscale
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = image1
    
    if len(image2.shape) == 3:
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = image2
    
    # Resize to the same dimensions
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    
    if h1 != h2 or w1 != w2:
        gray2 = cv2.resize(gray2, (w1, h1))
    
    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Threshold the difference
    _, mask = cv2.threshold(diff, int(threshold * 255), 255, cv2.THRESH_BINARY)
    
    # Calculate change score
    change_score = np.sum(mask) / (mask.shape[0] * mask.shape[1] * 255)
    
    return change_score, mask


def compare_panorama_images(
    panorama1: Panorama,
    panorama2: Panorama,
    image_manager,
    zoom: int = 3,
    method: str = 'sift'
) -> Dict[str, Any]:
    """
    Compare images from two panoramas.
    
    Parameters
    ----------
    panorama1 : Panorama
        First panorama
    panorama2 : Panorama
        Second panorama
    image_manager : PanoramaImageManager
        Image manager for loading images
    zoom : int, default=3
        Zoom level (0-5, where 5 is highest resolution)
    method : str, default='sift'
        Comparison method ('sift', 'ssim', or 'histogram')
        
    Returns
    -------
    Dict[str, Any]
        Comparison results
    """
    # Load images
    image_path1 = image_manager.load_image(panorama1, zoom)
    image_path2 = image_manager.load_image(panorama2, zoom)
    
    if image_path1 is None or image_path2 is None:
        return {
            'error': 'Could not load images',
            'similarity': 0.0,
            'change_score': 1.0
        }
    
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)
    
    # Compute similarity
    similarity = compute_image_similarity(image1, image2, method)
    
    # Detect changes
    change_score, change_mask = detect_changes(image1, image2)
    
    return {
        'similarity': similarity,
        'change_score': change_score,
        'panorama1_id': panorama1.pano_id,
        'panorama2_id': panorama2.pano_id,
        'panorama1_date': panorama1.date,
        'panorama2_date': panorama2.date,
        'time_difference_days': (panorama2.date - panorama1.date).days if panorama1.date and panorama2.date else None
    }
