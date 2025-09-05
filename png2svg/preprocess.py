"""
Preprocessing module for PNG2SVG system.
Handles image preprocessing including grayscale conversion, binarization, and deskewing.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple, Optional
import math

from .config import Config


@dataclass
class PreprocessOut:
    """Output of preprocessing pipeline."""
    img_gray: np.ndarray     # uint8 grayscale image
    img_bw: np.ndarray       # Binary image (foreground=255)
    transform: np.ndarray    # 2x3 affine transformation matrix for deskewing
    original_shape: Tuple[int, int]  # (height, width) of original image
    skew_angle: float = 0.0  # Detected skew angle in degrees


def run(img_bgr: np.ndarray, cfg: Config) -> PreprocessOut:
    """
    Run preprocessing pipeline on input image.
    
    - Grayscale conversion + light gaussian blur
    - Adaptive thresholding (THRESH_BINARY_INV)
    - Optional: Hough line-based skew detection + rotation correction
    - Return binary image and deskew transform
    
    Args:
        img_bgr: Input BGR image
        cfg: Configuration object
        
    Returns:
        PreprocessOut: Processed images and transformation info
    """
    logger = logging.getLogger('png2svg.preprocess')
    
    original_shape = img_bgr.shape[:2]  # (height, width)
    logger.debug(f"Input image shape: {original_shape}")
    
    # Step 1: Convert to grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    logger.debug("Converted to grayscale")
    
    # Step 2: Light gaussian blur to reduce noise
    ksize = cfg.algorithms.preprocess.gaussian_blur_ksize
    img_gray = cv2.GaussianBlur(img_gray, (ksize, ksize), 0)
    logger.debug(f"Applied Gaussian blur with kernel size {ksize}")
    
    # Step 3: Adaptive thresholding
    blocksize = cfg.algorithms.preprocess.adaptive_thresh_blocksize
    c_value = cfg.algorithms.preprocess.adaptive_thresh_c
    
    img_bw = cv2.adaptiveThreshold(
        img_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blocksize,
        c_value
    )
    logger.debug(f"Applied adaptive thresholding (blocksize={blocksize}, C={c_value})")
    
    # Step 4: Optional deskewing
    transform = np.eye(2, 3, dtype=np.float32)  # Identity transform
    skew_angle = 0.0
    
    if cfg.deskew:
        try:
            skew_angle = detect_skew_angle(img_bw, cfg)
            logger.debug(f"Detected skew angle: {skew_angle:.2f} degrees")
            
            # Apply deskewing if angle is significant
            if abs(skew_angle) > 0.5:  # Only correct if angle > 0.5 degrees
                img_gray, img_bw, transform = apply_deskew(
                    img_gray, img_bw, skew_angle
                )
                logger.info(f"Applied deskewing correction: {skew_angle:.2f} degrees")
            else:
                logger.debug("Skew angle too small, skipping correction")
                
        except Exception as e:
            logger.warning(f"Deskewing failed, continuing without correction: {e}")
    
    return PreprocessOut(
        img_gray=img_gray,
        img_bw=img_bw,
        transform=transform,
        original_shape=original_shape,
        skew_angle=skew_angle
    )


def detect_skew_angle(img_bw: np.ndarray, cfg: Config) -> float:
    """
    Detect skew angle using Hough line transform.
    
    Args:
        img_bw: Binary image
        cfg: Configuration object
        
    Returns:
        Skew angle in degrees
    """
    logger = logging.getLogger('png2svg.preprocess')
    
    # Edge detection
    edges = cv2.Canny(img_bw, 50, 150, apertureSize=3)
    
    # Hough line transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=cfg.algorithms.hough_lines.threshold,
        minLineLength=cfg.min_line_len * 2,  # Use longer lines for skew detection
        maxLineGap=cfg.algorithms.hough_lines.max_line_gap
    )
    
    if lines is None or len(lines) == 0:
        logger.debug("No lines found for skew detection")
        return 0.0
    
    # Calculate angles of all lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate angle in degrees
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Normalize to [-90, 90] range
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        angles.append(angle)
    
    if not angles:
        return 0.0
    
    # Find the dominant angle using histogram
    angles = np.array(angles)
    
    # Remove outliers (angles too far from horizontal/vertical)
    angles = angles[abs(angles) < 45]  # Keep only reasonably horizontal/vertical lines
    
    if len(angles) == 0:
        return 0.0
    
    # Calculate histogram of angles
    hist, bin_edges = np.histogram(angles, bins=90, range=(-45, 45))
    
    # Find the most frequent angle
    max_bin_idx = np.argmax(hist)
    dominant_angle = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
    
    logger.debug(f"Analyzed {len(angles)} lines, dominant angle: {dominant_angle:.2f}°")
    
    return dominant_angle


def apply_deskew(img_gray: np.ndarray, img_bw: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply deskewing rotation to images.
    
    Args:
        img_gray: Grayscale image
        img_bw: Binary image
        angle: Rotation angle in degrees
        
    Returns:
        Tuple of (rotated_gray, rotated_bw, transform_matrix)
    """
    h, w = img_gray.shape
    center = (w // 2, h // 2)
    
    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new image dimensions to avoid clipping
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    
    new_w = int((h * sin_a) + (w * cos_a))
    new_h = int((h * cos_a) + (w * sin_a))
    
    # Adjust the rotation matrix to account for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Apply rotation
    img_gray_rotated = cv2.warpAffine(
        img_gray, M, (new_w, new_h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=255  # White background
    )
    
    img_bw_rotated = cv2.warpAffine(
        img_bw, M, (new_w, new_h), 
        flags=cv2.INTER_NEAREST,  # Use nearest neighbor for binary image
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0  # Black background for binary
    )
    
    return img_gray_rotated, img_bw_rotated, M


def enhance_image_quality(img: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """
    Enhance image quality using various methods.
    
    Args:
        img: Input grayscale image
        method: Enhancement method ('clahe', 'equalize', 'none')
        
    Returns:
        Enhanced image
    """
    if method == 'clahe':
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    elif method == 'equalize':
        # Global histogram equalization
        return cv2.equalizeHist(img)
    else:
        return img


def remove_noise(img_bw: np.ndarray, method: str = 'morphology') -> np.ndarray:
    """
    Remove noise from binary image.
    
    Args:
        img_bw: Binary image
        method: Denoising method ('morphology', 'median', 'bilateral')
        
    Returns:
        Denoised image
    """
    if method == 'morphology':
        # Morphological operations to clean up binary image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Remove small noise
        img_clean = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill small holes
        img_clean = cv2.morphologyEx(img_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return img_clean
    
    elif method == 'median':
        # Median filter for noise reduction
        return cv2.medianBlur(img_bw, 3)
    
    else:
        return img_bw


def auto_crop_margins(img: np.ndarray, margin: int = 10) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Automatically crop white margins from image.
    
    Args:
        img: Input image (grayscale or binary)
        margin: Additional margin to keep (pixels)
        
    Returns:
        Tuple of (cropped_image, (top, bottom, left, right))
    """
    # Find non-white pixels
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Find bounding box of content
    coords = cv2.findNonZero(255 - gray)  # Invert to find non-white pixels
    
    if coords is None:
        # Image is completely white, return original
        return img, (0, 0, 0, 0)
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add margins
    h_img, w_img = img.shape[:2]
    top = max(0, y - margin)
    bottom = min(h_img, y + h + margin)
    left = max(0, x - margin)
    right = min(w_img, x + w + margin)
    
    # Crop image
    cropped = img[top:bottom, left:right]
    
    return cropped, (top, bottom, left, right)


def validate_preprocessing_result(preprocess_out: PreprocessOut) -> bool:
    """
    Validate preprocessing output for quality issues.
    
    Args:
        preprocess_out: Preprocessing output
        
    Returns:
        True if output is valid, False otherwise
    """
    logger = logging.getLogger('png2svg.preprocess')
    
    # Check if images are not empty
    if preprocess_out.img_gray.size == 0 or preprocess_out.img_bw.size == 0:
        logger.error("Preprocessing produced empty images")
        return False
    
    # Check if binary image has reasonable content
    white_pixels = np.sum(preprocess_out.img_bw == 255)
    total_pixels = preprocess_out.img_bw.size
    white_ratio = white_pixels / total_pixels
    
    if white_ratio < 0.01:  # Less than 1% white pixels
        logger.warning(f"Very low content ratio: {white_ratio:.3f}")
        return False
    
    if white_ratio > 0.99:  # More than 99% white pixels
        logger.warning(f"Almost no content detected: {white_ratio:.3f}")
        return False
    
    logger.debug(f"Preprocessing validation passed, content ratio: {white_ratio:.3f}")
    return True