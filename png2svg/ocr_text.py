"""
OCR text detection module for PNG2SVG system.
Detects text and mathematical labels using PaddleOCR or Tesseract.
"""

import cv2
import numpy as np
import logging
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from pathlib import Path

from .config import Config
from .preprocess import PreprocessOut


@dataclass
class OCRItem:
    """OCR detection result with text and location."""
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    conf: float
    category: str = "text"  # text/label/angle/formula


def run(img_bgr: np.ndarray, pre: PreprocessOut, cfg: Config) -> List[OCRItem]:
    """
    Run OCR text detection on image.
    
    - Prefer PaddleOCR for better multilingual and small text support
    - Fall back to pytesseract if PaddleOCR unavailable
    - Clean and categorize detected text (angles, labels, formulas)
    - Return results with bounding boxes and confidence
    
    Args:
        img_bgr: Original BGR image
        pre: Preprocessed images and metadata
        cfg: Configuration object
        
    Returns:
        List of OCRItem objects with detected text
    """
    logger = logging.getLogger('png2svg.ocr_text')
    
    # Try PaddleOCR first
    if cfg.use_paddle_ocr:
        try:
            ocr_results = run_paddleocr(img_bgr, pre, cfg)
            logger.info(f"PaddleOCR detected {len(ocr_results)} text items")
            return ocr_results
        except ImportError:
            logger.warning("PaddleOCR not available, falling back to Tesseract")
            cfg.use_paddle_ocr = False
        except Exception as e:
            logger.warning(f"PaddleOCR failed: {e}, falling back to Tesseract")
    
    # Fall back to Tesseract
    try:
        ocr_results = run_tesseract(img_bgr, pre, cfg)
        logger.info(f"Tesseract detected {len(ocr_results)} text items")
        return ocr_results
    except ImportError:
        logger.error("Neither PaddleOCR nor Tesseract available for OCR")
        return []
    except Exception as e:
        logger.error(f"Tesseract failed: {e}")
        return []


def run_paddleocr(img_bgr: np.ndarray, pre: PreprocessOut, cfg: Config) -> List[OCRItem]:
    """
    Run OCR using PaddleOCR.
    
    Args:
        img_bgr: Original BGR image
        pre: Preprocessed images 
        cfg: Configuration object
        
    Returns:
        List of OCRItem objects
    """
    from paddleocr import PaddleOCR
    
    logger = logging.getLogger('png2svg.ocr_text')
    
    # Initialize PaddleOCR
    # Use English model by default, can be configured for other languages
    ocr = PaddleOCR(
        use_angle_cls=True,  # Enable angle classification
        lang='en',            # Language
        show_log=False,       # Disable verbose logging
        use_gpu=False         # CPU mode for compatibility
    )
    
    # Run OCR
    results = ocr.ocr(img_bgr, cls=True)
    
    ocr_items = []
    if results and results[0] is not None:
        for line in results[0]:
            if line is None:
                continue
            
            # Parse PaddleOCR output format
            # Each line is [box_points, (text, confidence)]
            box_points = line[0]
            text_info = line[1]
            
            if text_info is None or len(text_info) < 2:
                continue
            
            text = text_info[0]
            conf = float(text_info[1])
            
            # Convert box points to bbox (x, y, w, h)
            box_points = np.array(box_points)
            x_min = int(np.min(box_points[:, 0]))
            y_min = int(np.min(box_points[:, 1]))
            x_max = int(np.max(box_points[:, 0]))
            y_max = int(np.max(box_points[:, 1]))
            
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            # Clean and categorize text
            cleaned_text = clean_ocr_text(text)
            category = categorize_text(cleaned_text)
            
            if cleaned_text:  # Only add non-empty text
                ocr_items.append(OCRItem(
                    text=cleaned_text,
                    bbox=bbox,
                    conf=conf,
                    category=category
                ))
    
    # Apply TTA if enabled
    if cfg.confidence_tta:
        ocr_items = apply_ocr_tta(img_bgr, ocr_items, ocr, cfg)
    
    return ocr_items


def run_tesseract(img_bgr: np.ndarray, pre: PreprocessOut, cfg: Config) -> List[OCRItem]:
    """
    Run OCR using Tesseract.
    
    Args:
        img_bgr: Original BGR image
        pre: Preprocessed images
        cfg: Configuration object
        
    Returns:
        List of OCRItem objects
    """
    import pytesseract
    
    logger = logging.getLogger('png2svg.ocr_text')
    
    # Convert to RGB for Tesseract
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Get OCR data with bounding boxes
    try:
        data = pytesseract.image_to_data(
            img_rgb, 
            output_type=pytesseract.Output.DICT,
            config='--psm 11'  # Sparse text mode
        )
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}")
        return []
    
    ocr_items = []
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        text = str(data['text'][i]).strip()
        
        if not text:  # Skip empty detections
            continue
        
        # Get bounding box
        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]
        
        # Get confidence (Tesseract returns -1 for no confidence)
        conf = data['conf'][i]
        if conf == -1:
            conf = 50  # Default confidence
        conf = conf / 100.0  # Convert to 0-1 range
        
        # Clean and categorize text
        cleaned_text = clean_ocr_text(text)
        category = categorize_text(cleaned_text)
        
        if cleaned_text and conf > 0.3:  # Filter low confidence
            ocr_items.append(OCRItem(
                text=cleaned_text,
                bbox=(x, y, w, h),
                conf=conf,
                category=category
            ))
    
    # Apply TTA if enabled
    if cfg.confidence_tta:
        ocr_items = apply_tesseract_tta(img_bgr, ocr_items, cfg)
    
    return ocr_items


def clean_ocr_text(text: str) -> str:
    """
    Clean OCR text output and fix common issues.
    
    Args:
        text: Raw OCR text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = text.strip()
    
    # Fix common OCR errors for mathematical text
    replacements = {
        '°': '°',      # Degree symbol
        '′': "'",      # Prime symbol
        '″': "''",     # Double prime
        'α': 'α',      # Greek alpha
        'β': 'β',      # Greek beta
        'γ': 'γ',      # Greek gamma
        'θ': 'θ',      # Greek theta
        'π': 'π',      # Greek pi
        'Δ': 'Δ',      # Greek Delta
        '∠': '∠',      # Angle symbol
        '∥': '∥',      # Parallel symbol
        '⊥': '⊥',      # Perpendicular symbol
        '≈': '≈',      # Approximately equal
        '≡': '≡',      # Identical to
        '±': '±',      # Plus-minus
        '∞': '∞',      # Infinity
        '√': '√',      # Square root
        '∑': '∑',      # Summation
        '∫': '∫',      # Integral
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Fix common misdetections
    # O (letter) vs 0 (zero) - context dependent
    if re.match(r'^[A-Z]$', text):  # Single uppercase letter
        pass  # Keep as is (likely a point label)
    elif re.match(r'^\d+$', text):  # Only digits
        text = text.replace('O', '0').replace('o', '0')
        text = text.replace('l', '1').replace('I', '1')
    
    # Fix degree symbol
    text = re.sub(r'(\d+)\s*[oO0]\s*(?![0-9])', r'\1°', text)  # 30o -> 30°
    
    # Fix prime notation
    text = re.sub(r"([A-Z])['`']", r"\1'", text)  # A` -> A'
    
    # Remove noise characters
    text = re.sub(r'[^\w\s°\-+=/().,\'″∠∥⊥αβγθπΔ≈≡±∞√∑∫]', '', text)
    
    return text.strip()


def categorize_text(text: str) -> str:
    """
    Categorize text based on content.
    
    Args:
        text: Cleaned OCR text
        
    Returns:
        Category: 'label', 'angle', 'length', 'formula', or 'text'
    """
    if not text:
        return "text"
    
    # Point labels (single uppercase letters, optionally with prime)
    if re.match(r'^[A-Z][\'″]?$', text):
        return "label"
    
    # Angle measurements (numbers with degree symbol)
    if re.match(r'^\d+\.?\d*°$', text):
        return "angle"
    
    # Length measurements (numbers, possibly with units)
    if re.match(r'^\d+\.?\d*\s*(cm|mm|m|km|in|ft)?$', text):
        return "length"
    
    # Mathematical formulas (contains operators or special symbols)
    if any(sym in text for sym in ['+', '-', '=', '/', '*', '√', '∑', '∫', 'π']):
        return "formula"
    
    # Variable names (single lowercase letters)
    if re.match(r'^[a-z]$', text):
        return "variable"
    
    # Greek letters
    if any(char in text for char in 'αβγδεζηθικλμνξοπρστυφχψω'):
        return "greek"
    
    # Default to generic text
    return "text"


def apply_ocr_tta(img_bgr: np.ndarray, ocr_items: List[OCRItem], ocr, cfg: Config) -> List[OCRItem]:
    """
    Apply Test-Time Augmentation for PaddleOCR to improve accuracy.
    
    Args:
        img_bgr: Original image
        ocr_items: Initial OCR results
        ocr: PaddleOCR instance
        cfg: Configuration object
        
    Returns:
        Enhanced OCR results with improved confidence
    """
    logger = logging.getLogger('png2svg.ocr_text')
    
    # Store all detections from different augmentations
    all_detections = {item.text: [item] for item in ocr_items}
    
    # Augmentation: Horizontal flip
    img_flip = cv2.flip(img_bgr, 1)
    results_flip = ocr.ocr(img_flip, cls=True)
    
    if results_flip and results_flip[0] is not None:
        h, w = img_bgr.shape[:2]
        for line in results_flip[0]:
            if line is None:
                continue
            
            box_points = line[0]
            text_info = line[1]
            
            if text_info is None or len(text_info) < 2:
                continue
            
            text = clean_ocr_text(text_info[0])
            conf = float(text_info[1])
            
            # Mirror bbox coordinates
            box_points = np.array(box_points)
            x_min = w - int(np.max(box_points[:, 0]))
            y_min = int(np.min(box_points[:, 1]))
            x_max = w - int(np.min(box_points[:, 0]))
            y_max = int(np.max(box_points[:, 1]))
            
            if text:
                if text not in all_detections:
                    all_detections[text] = []
                all_detections[text].append(OCRItem(
                    text=text,
                    bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                    conf=conf,
                    category=categorize_text(text)
                ))
    
    # Merge detections and boost confidence
    merged_items = []
    for text, detections in all_detections.items():
        if len(detections) == 1:
            merged_items.append(detections[0])
        else:
            # Average bbox and boost confidence
            avg_x = np.mean([d.bbox[0] for d in detections])
            avg_y = np.mean([d.bbox[1] for d in detections])
            avg_w = np.mean([d.bbox[2] for d in detections])
            avg_h = np.mean([d.bbox[3] for d in detections])
            max_conf = max(d.conf for d in detections)
            
            merged_items.append(OCRItem(
                text=text,
                bbox=(int(avg_x), int(avg_y), int(avg_w), int(avg_h)),
                conf=min(1.0, max_conf * 1.1),  # Boost confidence by 10%
                category=detections[0].category
            ))
    
    logger.debug(f"TTA enhanced OCR from {len(ocr_items)} to {len(merged_items)} detections")
    return merged_items


def apply_tesseract_tta(img_bgr: np.ndarray, ocr_items: List[OCRItem], cfg: Config) -> List[OCRItem]:
    """
    Apply Test-Time Augmentation for Tesseract to improve accuracy.
    
    Args:
        img_bgr: Original image
        ocr_items: Initial OCR results
        cfg: Configuration object
        
    Returns:
        Enhanced OCR results
    """
    import pytesseract
    
    logger = logging.getLogger('png2svg.ocr_text')
    
    # For Tesseract, we'll try different PSM modes
    psm_modes = [11, 8, 6]  # Sparse text, single word, single block
    
    all_detections = {item.text: [item] for item in ocr_items}
    
    for psm in psm_modes[1:]:  # Skip first since we already used it
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            data = pytesseract.image_to_data(
                img_rgb, 
                output_type=pytesseract.Output.DICT,
                config=f'--psm {psm}'
            )
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = clean_ocr_text(str(data['text'][i]).strip())
                
                if not text:
                    continue
                
                conf = data['conf'][i]
                if conf == -1:
                    conf = 40
                conf = conf / 100.0
                
                if conf > 0.3:
                    if text not in all_detections:
                        all_detections[text] = []
                    all_detections[text].append(OCRItem(
                        text=text,
                        bbox=(data['left'][i], data['top'][i], 
                              data['width'][i], data['height'][i]),
                        conf=conf,
                        category=categorize_text(text)
                    ))
        except:
            continue
    
    # Merge detections
    merged_items = []
    for text, detections in all_detections.items():
        if len(detections) == 1:
            merged_items.append(detections[0])
        else:
            # Use detection with highest confidence
            best = max(detections, key=lambda d: d.conf)
            merged_items.append(best)
    
    logger.debug(f"TTA enhanced OCR from {len(ocr_items)} to {len(merged_items)} detections")
    return merged_items


def filter_ocr_by_region(ocr_items: List[OCRItem], region: Tuple[int, int, int, int]) -> List[OCRItem]:
    """
    Filter OCR results to only include those within a specific region.
    
    Args:
        ocr_items: List of OCR detections
        region: Region of interest (x, y, w, h)
        
    Returns:
        Filtered OCR items
    """
    rx, ry, rw, rh = region
    filtered = []
    
    for item in ocr_items:
        x, y, w, h = item.bbox
        
        # Check if bbox overlaps with region
        if (x < rx + rw and x + w > rx and 
            y < ry + rh and y + h > ry):
            filtered.append(item)
    
    return filtered


def merge_nearby_text(ocr_items: List[OCRItem], distance_threshold: int = 20) -> List[OCRItem]:
    """
    Merge nearby text items that likely belong together.
    
    Args:
        ocr_items: List of OCR detections
        distance_threshold: Maximum distance to consider items as nearby
        
    Returns:
        Merged OCR items
    """
    if len(ocr_items) <= 1:
        return ocr_items
    
    # Sort by position (top to bottom, left to right)
    sorted_items = sorted(ocr_items, key=lambda item: (item.bbox[1], item.bbox[0]))
    
    merged = []
    current_group = [sorted_items[0]]
    
    for item in sorted_items[1:]:
        # Check if item is close to the last item in current group
        last_item = current_group[-1]
        
        # Calculate distance
        x1, y1, w1, h1 = last_item.bbox
        x2, y2, w2, h2 = item.bbox
        
        # Check vertical alignment and horizontal distance
        vertical_overlap = (y1 < y2 + h2 and y2 < y1 + h1)
        horizontal_distance = max(0, x2 - (x1 + w1))
        
        if vertical_overlap and horizontal_distance <= distance_threshold:
            # Items are on the same line and close together
            current_group.append(item)
        else:
            # Start a new group
            if current_group:
                merged.append(merge_group(current_group))
            current_group = [item]
    
    # Don't forget the last group
    if current_group:
        merged.append(merge_group(current_group))
    
    return merged


def merge_group(group: List[OCRItem]) -> OCRItem:
    """
    Merge a group of OCR items into a single item.
    
    Args:
        group: List of OCR items to merge
        
    Returns:
        Merged OCR item
    """
    if len(group) == 1:
        return group[0]
    
    # Combine text
    texts = [item.text for item in group]
    combined_text = ' '.join(texts)
    
    # Calculate combined bounding box
    x_min = min(item.bbox[0] for item in group)
    y_min = min(item.bbox[1] for item in group)
    x_max = max(item.bbox[0] + item.bbox[2] for item in group)
    y_max = max(item.bbox[1] + item.bbox[3] for item in group)
    
    # Average confidence
    avg_conf = np.mean([item.conf for item in group])
    
    # Recategorize
    category = categorize_text(combined_text)
    
    return OCRItem(
        text=combined_text,
        bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
        conf=avg_conf,
        category=category
    )