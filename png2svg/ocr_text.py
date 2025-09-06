"""
Text recognition (OCR) for PNG2SVG.

Provides text recognition using PaddleOCR (preferred) with Tesseract fallback.
Includes support for mathematical symbols, multi-language text, and confidence scoring.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import uuid

import cv2
import numpy as np

from .config import Config
from .preprocess import PreprocessResult


@dataclass
class OCRResult:
    """Result of OCR detection on a text region."""
    text: str                           # Recognized text content
    bbox: Tuple[int, int, int, int]    # Bounding box (x, y, w, h)
    confidence: float                  # Recognition confidence [0, 1]
    language: str = "en"               # Detected/assumed language
    text_type: str = "label"           # "label", "number", "angle", "variable"
    cleaned_text: str = ""             # Cleaned/normalized text
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class OCRResults:
    """Collection of OCR results with metadata."""
    items: List[OCRResult]
    stats: dict = field(default_factory=dict)


def run(
    img_bgr: np.ndarray, 
    preprocess_result: PreprocessResult, 
    config: Config
) -> OCRResults:
    """
    Run OCR on the image to extract text labels and annotations.
    
    Args:
        img_bgr: Original color image
        preprocess_result: Preprocessed images
        config: Configuration object
        
    Returns:
        OCRResults: Detected text with bounding boxes and confidence
    """
    logger = logging.getLogger(__name__)
    
    ocr_results = []
    
    # Choose OCR engine based on configuration and availability
    if config.use_paddle_ocr and _is_paddleocr_available():
        logger.debug("Using PaddleOCR for text recognition")
        ocr_results = _run_paddleocr(img_bgr, config)
    else:
        logger.debug("Using Tesseract for text recognition")
        ocr_results = _run_tesseract(img_bgr, preprocess_result.img_gray, config)
    
    # Apply Test Time Augmentation if enabled
    if config.confidence_tta and len(ocr_results) < 10:  # Only for smaller result sets
        logger.debug("Applying TTA for OCR confidence improvement")
        tta_results = _apply_tta_ocr(img_bgr, preprocess_result, config)
        ocr_results = _merge_tta_results(ocr_results, tta_results)
    
    # Post-process and clean text
    for result in ocr_results:
        result.cleaned_text = _clean_text(result.text)
        result.text_type = _classify_text_type(result.cleaned_text)
    
    # Filter out very low confidence results
    filtered_results = [r for r in ocr_results if r.confidence > 0.1]
    
    stats = {
        'total_detections': len(ocr_results),
        'filtered_detections': len(filtered_results),
        'avg_confidence': np.mean([r.confidence for r in filtered_results]) if filtered_results else 0,
        'text_types': {t: sum(1 for r in filtered_results if r.text_type == t) 
                      for t in ['label', 'number', 'angle', 'variable']}
    }
    
    logger.info(f"OCR found {len(filtered_results)} text regions "
               f"(avg confidence: {stats['avg_confidence']:.2f})")
    
    return OCRResults(items=filtered_results, stats=stats)


def _is_paddleocr_available() -> bool:
    """Check if PaddleOCR is available for import."""
    try:
        import paddleocr
        return True
    except ImportError:
        return False


def _run_paddleocr(img_bgr: np.ndarray, config: Config) -> List[OCRResult]:
    """
    Run PaddleOCR on the image.
    
    Args:
        img_bgr: Color image
        config: Configuration object
        
    Returns:
        List[OCRResult]: OCR detection results
    """
    try:
        import paddleocr
        
        # Initialize PaddleOCR
        ocr = paddleocr.PaddleOCR(
            use_angle_cls=True,  # Enable text angle classification
            lang='en',           # Primary language
            show_log=False       # Suppress PaddleOCR logs
        )
        
        # Run OCR
        results = ocr.ocr(img_bgr)
        
        ocr_items = []
        if results and results[0]:
            for detection in results[0]:
                # Extract bounding box and text
                bbox_points, (text, confidence) = detection
                
                # Convert bbox points to (x, y, w, h)
                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]
                x, y = int(min(x_coords)), int(min(y_coords))
                w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                
                if text.strip() and confidence > 0.1:
                    ocr_items.append(OCRResult(
                        text=text,
                        bbox=(x, y, w, h),
                        confidence=float(confidence)
                    ))
        
        return ocr_items
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"PaddleOCR failed: {e}")
        return []


def _run_tesseract(
    img_bgr: np.ndarray, 
    img_gray: np.ndarray, 
    config: Config
) -> List[OCRResult]:
    """
    Run Tesseract OCR on the image.
    
    Args:
        img_bgr: Color image
        img_gray: Grayscale image
        config: Configuration object
        
    Returns:
        List[OCRResult]: OCR detection results
    """
    try:
        import pytesseract
        
        # Enhanced OCR configuration for mathematical text
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789°′″+-=×÷√∞∠∥⊥△□○()[]{}.,\'"'
        
        # Get detailed OCR data
        data = pytesseract.image_to_data(
            img_gray, 
            config=custom_config, 
            output_type=pytesseract.Output.DICT
        )
        
        ocr_items = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            confidence = float(data['conf'][i]) / 100.0  # Convert to [0, 1]
            
            if text and confidence > 0.1:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                ocr_items.append(OCRResult(
                    text=text,
                    bbox=(x, y, w, h),
                    confidence=confidence
                ))
        
        return ocr_items
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Tesseract failed: {e}")
        return []


def _apply_tta_ocr(
    img_bgr: np.ndarray, 
    preprocess_result: PreprocessResult, 
    config: Config
) -> List[List[OCRResult]]:
    """
    Apply Test Time Augmentation for OCR.
    
    Args:
        img_bgr: Original color image
        preprocess_result: Preprocessed images
        config: Configuration object
        
    Returns:
        List[List[OCRResult]]: OCR results for each augmentation
    """
    tta_results = []
    
    # Original image (already processed)
    # We'll skip this to avoid duplicate processing
    
    # Horizontal flip
    img_flipped = cv2.flip(img_bgr, 1)
    if config.use_paddle_ocr and _is_paddleocr_available():
        flipped_results = _run_paddleocr(img_flipped, config)
    else:
        gray_flipped = cv2.flip(preprocess_result.img_gray, 1)
        flipped_results = _run_tesseract(img_flipped, gray_flipped, config)
    
    # Transform coordinates back
    h, w = img_bgr.shape[:2]
    for result in flipped_results:
        x, y, box_w, box_h = result.bbox
        result.bbox = (w - x - box_w, y, box_w, box_h)
    
    tta_results.append(flipped_results)
    
    # Slight rotation (±2 degrees)
    for angle in [-2, 2]:
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_rotated = cv2.warpAffine(img_bgr, rotation_matrix, (w, h), borderValue=(255, 255, 255))
        
        if config.use_paddle_ocr and _is_paddleocr_available():
            rotated_results = _run_paddleocr(img_rotated, config)
        else:
            gray_rotated = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)
            rotated_results = _run_tesseract(img_rotated, gray_rotated, config)
        
        # Transform coordinates back (simplified - assumes small rotation)
        # For production, should implement proper inverse transformation
        tta_results.append(rotated_results)
    
    return tta_results


def _merge_tta_results(
    original_results: List[OCRResult], 
    tta_results: List[List[OCRResult]]
) -> List[OCRResult]:
    """
    Merge TTA results with original results using confidence voting.
    
    Args:
        original_results: Original OCR results
        tta_results: TTA OCR results
        
    Returns:
        List[OCRResult]: Merged results with improved confidence
    """
    # Simple implementation: just add high-confidence TTA results
    # that don't overlap significantly with original results
    
    merged = original_results.copy()
    
    for tta_batch in tta_results:
        for tta_result in tta_batch:
            # Check if this result overlaps with existing ones
            overlaps = False
            for existing in merged:
                if _boxes_overlap(tta_result.bbox, existing.bbox, threshold=0.5):
                    # If texts are similar, boost confidence
                    if _text_similarity(tta_result.text, existing.text) > 0.7:
                        existing.confidence = min(1.0, existing.confidence + 0.1)
                    overlaps = True
                    break
            
            if not overlaps and tta_result.confidence > 0.3:
                merged.append(tta_result)
    
    return merged


def _boxes_overlap(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int], threshold: float = 0.3) -> bool:
    """Check if two bounding boxes overlap significantly."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    
    if ix1 < ix2 and iy1 < iy2:
        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = w1 * h1 + w2 * h2 - intersection
        iou = intersection / union if union > 0 else 0
        return iou > threshold
    
    return False


def _text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity based on character overlap."""
    if not text1 or not text2:
        return 0.0
    
    text1, text2 = text1.lower(), text2.lower()
    if text1 == text2:
        return 1.0
    
    # Simple character-based similarity
    chars1, chars2 = set(text1), set(text2)
    intersection = len(chars1 & chars2)
    union = len(chars1 | chars2)
    
    return intersection / union if union > 0 else 0.0


def _clean_text(text: str) -> str:
    """
    Clean and normalize OCR text.
    
    Args:
        text: Raw OCR text
        
    Returns:
        str: Cleaned text
    """
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', text.strip())
    
    # Fix common OCR errors for mathematical text
    replacements = {
        'o': '0',     # o -> 0 for numbers
        'O': '0',     # O -> 0 for numbers  
        'l': '1',     # l -> 1 for numbers
        'I': '1',     # I -> 1 for numbers
        '|': '1',     # | -> 1 for numbers
        'S': '5',     # S -> 5 for numbers (context-dependent)
        'G': '6',     # G -> 6 for numbers (context-dependent)
    }
    
    # Apply replacements only if the text looks like it should be numeric
    if re.match(r'^[0-9oOlI|SG.,°′″+-]+$', cleaned):
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
    
    # Normalize special characters
    cleaned = cleaned.replace('°', '°')  # Degree symbol
    cleaned = cleaned.replace("'", '′')  # Prime symbol
    cleaned = cleaned.replace('"', '″')  # Double prime symbol
    
    return cleaned


def _classify_text_type(text: str) -> str:
    """
    Classify the type of text based on content.
    
    Args:
        text: Text to classify
        
    Returns:
        str: Text type classification
    """
    if not text:
        return "label"
    
    # Angle measurements
    if re.search(r'\d+°', text) or re.search(r'\d+\s*deg', text.lower()):
        return "angle"
    
    # Pure numbers
    if re.match(r'^[0-9.,+-]+$', text):
        return "number"
    
    # Mathematical variables (single letters, often)
    if re.match(r'^[A-Za-z][\'\″′]*$', text):
        return "variable"
    
    # Point labels (single letters or short combinations)
    if len(text) <= 3 and re.match(r'^[A-Z][A-Z0-9]*$', text):
        return "label"
    
    # Default to label
    return "label"


def extract_text_regions_from_contours(img_gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Extract potential text regions using contour analysis.
    
    Args:
        img_gray: Grayscale image
        
    Returns:
        List[Tuple[int, int, int, int]]: List of bounding boxes (x, y, w, h)
    """
    # Threshold image
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter based on size (potential text characteristics)
        aspect_ratio = w / h if h > 0 else 0
        area = w * h
        
        if (5 <= w <= 200 and 5 <= h <= 100 and  # Size constraints
            0.1 <= aspect_ratio <= 10 and        # Aspect ratio constraints
            50 <= area <= 10000):                 # Area constraints
            text_regions.append((x, y, w, h))
    
    return text_regions