"""
Image preprocessing for PNG2SVG.

Handles image binarization, denoising, and optional deskewing to prepare
PNG images for geometric feature detection and analysis.
"""

import logging
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np

from .config import Config


@dataclass
class PreprocessResult:
    """Result of image preprocessing."""
    img_gray: np.ndarray         # Grayscale image (uint8)
    img_bw: np.ndarray          # Binary image (foreground=255, background=0)
    transform: np.ndarray       # 2x3 affine transformation matrix for deskewing
    skew_angle: float           # Detected skew angle in degrees
    original_size: Tuple[int, int]  # (width, height) of original image


def run(img_bgr: np.ndarray, config: Config) -> PreprocessResult:
    """
    Preprocess image for geometric feature detection.
    
    Steps:
    1. Convert to grayscale
    2. Apply Gaussian blur for denoising
    3. Adaptive thresholding for binarization
    4. Optional deskewing based on dominant line orientation
    
    Args:
        img_bgr: Input image in BGR format
        config: Configuration object
        
    Returns:
        PreprocessResult: Processed images and transformation info
    """
    logger = logging.getLogger(__name__)
    
    h, w = img_bgr.shape[:2]
    original_size = (w, h)
    
    # Step 1: Convert to grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Denoising with Gaussian blur
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    # Step 3: Adaptive thresholding for binarization
    img_bw = cv2.adaptiveThreshold(
        img_gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Foreground (text/lines) = 255, background = 0
        35,  # Block size
        15   # C constant
    )
    
    # Step 4: Optional deskewing
    skew_angle = 0.0
    transform = np.eye(2, 3, dtype=np.float32)  # Identity transformation
    
    if config.deskew:
        try:
            skew_angle = detect_skew_angle(img_bw)
            logger.debug(f"Detected skew angle: {skew_angle:.2f} degrees")
            
            # Apply deskewing if angle is significant (> 0.5 degrees)
            if abs(skew_angle) > 0.5:
                img_gray, img_bw, transform = apply_deskew(img_gray, img_bw, skew_angle)
                logger.debug(f"Applied deskewing correction: {skew_angle:.2f} degrees")
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            skew_angle = 0.0
    
    return PreprocessResult(
        img_gray=img_gray,
        img_bw=img_bw,
        transform=transform,
        skew_angle=skew_angle,
        original_size=original_size
    )


def detect_skew_angle(img_bw: np.ndarray, angle_range: float = 15.0) -> float:
    """
    Detect skew angle using Hough line transform.
    
    Args:
        img_bw: Binary image
        angle_range: Range of angles to search (±degrees)
        
    Returns:
        float: Detected skew angle in degrees
    """
    # Apply morphological operations to enhance line detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    img_processed = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel)
    
    # Edge detection
    edges = cv2.Canny(img_processed, 50, 150, apertureSize=3)
    
    # Hough line detection
    lines = cv2.HoughLines(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=max(50, int(min(img_bw.shape) * 0.3))
    )
    
    if lines is None or len(lines) == 0:
        return 0.0
    
    # Extract angles and find dominant direction
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle_deg = math.degrees(theta)
        
        # Convert to [-90, 90] range
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180
        
        # Filter to reasonable skew range
        if abs(angle_deg) <= angle_range:
            angles.append(angle_deg)
        # Also consider angles around ±90 degrees (vertical lines)
        elif abs(abs(angle_deg) - 90) <= angle_range:
            # Convert vertical lines to horizontal equivalent
            if angle_deg > 0:
                angles.append(angle_deg - 90)
            else:
                angles.append(angle_deg + 90)
    
    if not angles:
        return 0.0
    
    # Find dominant angle using histogram approach
    angles = np.array(angles)
    
    # Create histogram of angles
    hist, bin_edges = np.histogram(angles, bins=30, range=(-angle_range, angle_range))
    
    # Find peak
    peak_idx = np.argmax(hist)
    peak_angle = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
    
    return peak_angle


def apply_deskew(
    img_gray: np.ndarray, 
    img_bw: np.ndarray, 
    angle: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply deskewing transformation to images.
    
    Args:
        img_gray: Grayscale image
        img_bw: Binary image
        angle: Skew angle in degrees
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (deskewed_gray, deskewed_bw, transform_matrix)
    """
    h, w = img_gray.shape
    
    # Calculate rotation matrix
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new image dimensions to avoid cropping
    cos_angle = abs(rotation_matrix[0, 0])
    sin_angle = abs(rotation_matrix[0, 1])
    
    new_w = int((h * sin_angle) + (w * cos_angle))
    new_h = int((h * cos_angle) + (w * sin_angle))
    
    # Adjust translation to center the rotated image
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2
    
    # Apply transformation
    img_gray_deskewed = cv2.warpAffine(
        img_gray, 
        rotation_matrix, 
        (new_w, new_h), 
        flags=cv2.INTER_CUBIC,
        borderValue=255  # White background
    )
    
    img_bw_deskewed = cv2.warpAffine(
        img_bw, 
        rotation_matrix, 
        (new_w, new_h), 
        flags=cv2.INTER_NEAREST,
        borderValue=0  # Black background for binary image
    )
    
    return img_gray_deskewed, img_bw_deskewed, rotation_matrix


def enhance_for_line_detection(img_bw: np.ndarray) -> np.ndarray:
    """
    Enhance binary image for better line detection.
    
    Args:
        img_bw: Binary image
        
    Returns:
        np.ndarray: Enhanced binary image
    """
    # Remove small noise with opening
    kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_clean = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel_noise)
    
    # Enhance horizontal lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    enhanced_h = cv2.morphologyEx(img_clean, cv2.MORPH_CLOSE, kernel_h)
    
    # Enhance vertical lines
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    enhanced_v = cv2.morphologyEx(img_clean, cv2.MORPH_CLOSE, kernel_v)
    
    # Combine enhancements
    enhanced = cv2.bitwise_or(enhanced_h, enhanced_v)
    
    return enhanced


def enhance_for_text_detection(img_bw: np.ndarray) -> np.ndarray:
    """
    Enhance binary image for better text detection.
    
    Args:
        img_bw: Binary image
        
    Returns:
        np.ndarray: Enhanced binary image
    """
    # Remove very small components (noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_clean = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel)
    
    # Slightly dilate to connect broken characters
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_enhanced = cv2.dilate(img_clean, kernel_dilate, iterations=1)
    
    return img_enhanced


def get_image_stats(img_gray: np.ndarray, img_bw: np.ndarray) -> dict:
    """
    Calculate image statistics for debugging and quality assessment.
    
    Args:
        img_gray: Grayscale image
        img_bw: Binary image
        
    Returns:
        dict: Dictionary with image statistics
    """
    h, w = img_gray.shape
    
    # Basic stats
    mean_intensity = np.mean(img_gray)
    std_intensity = np.std(img_gray)
    
    # Binary image stats
    foreground_pixels = np.sum(img_bw > 0)
    foreground_ratio = foreground_pixels / (h * w)
    
    # Edge density (rough measure of content complexity)
    edges = cv2.Canny(img_gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    edge_density = edge_pixels / (h * w)
    
    return {
        'size': (w, h),
        'mean_intensity': float(mean_intensity),
        'std_intensity': float(std_intensity),
        'foreground_ratio': float(foreground_ratio),
        'edge_density': float(edge_density)
    }