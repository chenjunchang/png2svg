"""
Symbol detection module for PNG2SVG system.
Detects mathematical symbols using YOLO (if available) or rule-based methods.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import uuid
from pathlib import Path
import math

from .config import Config
from .preprocess import PreprocessOut


@dataclass
class Symbol:
    """Detected symbol with bounding box and classification."""
    cls: str                    # right_angle / arc_1 / arc_2 / arc_3 / tick_1/2/3 / parallel / arrow_head / dot_filled / dot_hollow
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    conf: float
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class Symbols:
    """Container for all detected symbols."""
    items: List[Symbol] = field(default_factory=list)


def run(pre: PreprocessOut, cfg: Config) -> Symbols:
    """
    Run symbol detection on preprocessed image.
    
    - If ONNX weights found, use onnxruntime inference
    - Otherwise fallback to rule-based detection with template matching/shape approximation
    - Output unified Symbol list
    
    Args:
        pre: Preprocessed images and metadata
        cfg: Configuration object
        
    Returns:
        Symbols: Detected mathematical symbols
    """
    logger = logging.getLogger('png2svg.detect_symbols')
    
    if cfg.use_yolo_symbols:
        try:
            symbols = detect_symbols_yolo(pre.img_gray, pre.img_bw, cfg)
            logger.info(f"YOLO detected {len(symbols.items)} symbols")
            return symbols
        except Exception as e:
            logger.warning(f"YOLO detection failed, falling back to rules: {e}")
    
    # Fallback to rule-based detection
    symbols = detect_symbols_rules(pre.img_gray, pre.img_bw, cfg)
    logger.info(f"Rule-based detection found {len(symbols.items)} symbols")
    return symbols


def detect_symbols_yolo(img_gray: np.ndarray, img_bw: np.ndarray, cfg: Config) -> Symbols:
    """
    Detect symbols using YOLO model (ONNX or PyTorch).
    
    Args:
        img_gray: Grayscale image
        img_bw: Binary image
        cfg: Configuration object
        
    Returns:
        Symbols: Detected symbols
    """
    logger = logging.getLogger('png2svg.detect_symbols')
    
    # Check if weights file exists
    weights_path = Path(cfg.yolo_symbols_weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    
    # Try ONNX Runtime first
    try:
        return detect_symbols_onnx(img_gray, weights_path, cfg)
    except ImportError:
        logger.debug("ONNX Runtime not available, trying Ultralytics")
    except Exception as e:
        logger.debug(f"ONNX detection failed: {e}, trying Ultralytics")
    
    # Try Ultralytics YOLO
    try:
        return detect_symbols_ultralytics(img_gray, weights_path, cfg)
    except ImportError:
        raise ImportError("Neither ONNX Runtime nor Ultralytics available for YOLO inference")


def detect_symbols_onnx(img_gray: np.ndarray, weights_path: Path, cfg: Config) -> Symbols:
    """
    Detect symbols using ONNX Runtime.
    
    Args:
        img_gray: Grayscale image
        weights_path: Path to ONNX model
        cfg: Configuration object
        
    Returns:
        Symbols: Detected symbols
    """
    import onnxruntime as ort
    
    # Load ONNX model
    session = ort.InferenceSession(str(weights_path))
    
    # Prepare input
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Resize and normalize image for YOLO input
    img_resized = cv2.resize(img_gray, (input_shape[3], input_shape[2]))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(np.expand_dims(img_normalized, 0), 0)  # Add batch and channel dims
    
    # Run inference
    outputs = session.run(None, {input_name: img_batch})
    
    # Parse outputs (assuming YOLO format)
    symbols = parse_yolo_outputs(outputs[0], img_gray.shape, input_shape[2:])
    
    return Symbols(items=symbols)


def detect_symbols_ultralytics(img_gray: np.ndarray, weights_path: Path, cfg: Config) -> Symbols:
    """
    Detect symbols using Ultralytics YOLO.
    
    Args:
        img_gray: Grayscale image  
        weights_path: Path to model weights
        cfg: Configuration object
        
    Returns:
        Symbols: Detected symbols
    """
    from ultralytics import YOLO
    
    # Load model
    model = YOLO(str(weights_path))
    
    # Run inference
    results = model(img_gray, conf=0.3, iou=0.5, verbose=False)
    
    # Parse results
    symbols = []
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Map class ID to symbol name
                cls_name = get_symbol_class_name(cls_id)
                
                symbols.append(Symbol(
                    cls=cls_name,
                    bbox=(int(x1), int(y1), int(x2-x1), int(y2-y1)),
                    conf=conf
                ))
    
    return Symbols(items=symbols)


def parse_yolo_outputs(outputs: np.ndarray, orig_shape: Tuple[int, int], model_shape: Tuple[int, int]) -> List[Symbol]:
    """
    Parse YOLO model outputs to Symbol objects.
    
    Args:
        outputs: Raw YOLO outputs
        orig_shape: Original image shape (H, W)
        model_shape: Model input shape (H, W)
        
    Returns:
        List of Symbol objects
    """
    symbols = []
    
    # Parse YOLO format: [batch, detections, (x, y, w, h, conf, cls...)]
    if len(outputs.shape) == 3:
        detections = outputs[0]  # Remove batch dimension
        
        # Filter by confidence
        conf_mask = detections[:, 4] > 0.3
        detections = detections[conf_mask]
        
        # Scale coordinates back to original image
        scale_x = orig_shape[1] / model_shape[1]
        scale_y = orig_shape[0] / model_shape[0]
        
        for detection in detections:
            x_center, y_center, w, h, conf = detection[:5]
            cls_scores = detection[5:]
            cls_id = np.argmax(cls_scores)
            
            # Convert to corner coordinates
            x1 = int((x_center - w/2) * scale_x)
            y1 = int((y_center - h/2) * scale_y)
            width = int(w * scale_x)
            height = int(h * scale_y)
            
            # Create symbol
            cls_name = get_symbol_class_name(cls_id)
            symbols.append(Symbol(
                cls=cls_name,
                bbox=(x1, y1, width, height),
                conf=float(conf)
            ))
    
    return symbols


def get_symbol_class_name(cls_id: int) -> str:
    """
    Map class ID to symbol class name.
    
    Args:
        cls_id: Class ID from model
        
    Returns:
        Symbol class name
    """
    # This mapping should match your YOLO model's class definitions
    class_names = {
        0: "right_angle",
        1: "arc_1",
        2: "arc_2", 
        3: "arc_3",
        4: "tick_1",
        5: "tick_2",
        6: "tick_3",
        7: "parallel_mark",
        8: "perp_mark",
        9: "arrow_head",
        10: "dot_filled",
        11: "dot_hollow",
        12: "hatch_region",
        13: "extend_mark"
    }
    
    return class_names.get(cls_id, f"unknown_{cls_id}")


def detect_symbols_rules(img_gray: np.ndarray, img_bw: np.ndarray, cfg: Config) -> Symbols:
    """
    Detect symbols using rule-based methods and template matching.
    
    Args:
        img_gray: Grayscale image
        img_bw: Binary image
        cfg: Configuration object
        
    Returns:
        Symbols: Detected symbols
    """
    logger = logging.getLogger('png2svg.detect_symbols')
    symbols = []
    
    # Detect different symbol types
    symbols.extend(detect_right_angles(img_bw))
    symbols.extend(detect_arrow_heads(img_bw))
    symbols.extend(detect_dots(img_bw))
    symbols.extend(detect_ticks(img_bw))
    
    logger.debug(f"Rule-based detection found {len(symbols)} symbols")
    
    return Symbols(items=symbols)


def detect_right_angles(img_bw: np.ndarray) -> List[Symbol]:
    """
    Detect right angle markers (small squares) using shape analysis.
    
    Args:
        img_bw: Binary image
        
    Returns:
        List of right angle symbols
    """
    symbols = []
    
    # Find contours
    contours, _ = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Filter by area
        area = cv2.contourArea(contour)
        if area < 20 or area > 500:  # Right angle markers should be small
            continue
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's roughly a rectangle (4 corners)
        if len(approx) == 4:
            # Check aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Right angle markers should be roughly square
            if 0.7 <= aspect_ratio <= 1.3:
                symbols.append(Symbol(
                    cls="right_angle",
                    bbox=(x, y, w, h),
                    conf=0.6
                ))
    
    return symbols


def detect_arrow_heads(img_bw: np.ndarray) -> List[Symbol]:
    """
    Detect arrow heads using shape analysis.
    
    Args:
        img_bw: Binary image
        
    Returns:
        List of arrow head symbols
    """
    symbols = []
    
    # Find contours
    contours, _ = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Filter by area
        area = cv2.contourArea(contour)
        if area < 30 or area > 1000:
            continue
        
        # Approximate contour
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Arrow heads typically have 3-7 vertices
        if 3 <= len(approx) <= 7:
            # Check if it has a pointed shape
            x, y, w, h = cv2.boundingRect(contour)
            
            # Arrows are usually longer in one direction
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 1.5:
                symbols.append(Symbol(
                    cls="arrow_head",
                    bbox=(x, y, w, h),
                    conf=0.5
                ))
    
    return symbols


def detect_dots(img_bw: np.ndarray) -> List[Symbol]:
    """
    Detect filled and hollow dots using circle detection.
    
    Args:
        img_bw: Binary image
        
    Returns:
        List of dot symbols
    """
    symbols = []
    
    # Find contours
    contours, _ = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Filter by area
        area = cv2.contourArea(contour)
        if area < 10 or area > 200:  # Dots should be small
            continue
        
        # Check circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Dots should be roughly circular
        if circularity > 0.6:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's filled or hollow by analyzing the interior
            mask = np.zeros(img_bw.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Count white pixels inside
            interior = cv2.bitwise_and(img_bw, mask)
            interior_pixels = cv2.countNonZero(interior)
            total_pixels = cv2.countNonZero(mask)
            
            if total_pixels == 0:
                continue
            
            fill_ratio = interior_pixels / total_pixels
            
            # Determine if filled or hollow
            if fill_ratio > 0.7:
                cls_name = "dot_filled"
            else:
                cls_name = "dot_hollow"
            
            symbols.append(Symbol(
                cls=cls_name,
                bbox=(x, y, w, h),
                conf=0.6
            ))
    
    return symbols


def detect_ticks(img_bw: np.ndarray) -> List[Symbol]:
    """
    Detect tick marks (equal length indicators) using line analysis.
    
    Args:
        img_bw: Binary image
        
    Returns:
        List of tick symbols
    """
    symbols = []
    
    # Use morphological operations to find small line segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  # Vertical lines
    vertical_ticks = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # Horizontal lines
    horizontal_ticks = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel)
    
    # Combine both orientations
    ticks = cv2.bitwise_or(vertical_ticks, horizontal_ticks)
    
    # Find tick contours
    contours, _ = cv2.findContours(ticks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Filter by area - ticks are small
        area = cv2.contourArea(contour)
        if area < 5 or area > 100:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Ticks are usually thin lines
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 2:  # Should be elongated
            # For now, classify all as tick_1 (single tick)
            # More sophisticated analysis could determine tick_2, tick_3
            symbols.append(Symbol(
                cls="tick_1",
                bbox=(x, y, w, h),
                conf=0.4
            ))
    
    return symbols


def filter_overlapping_symbols(symbols: List[Symbol], iou_threshold: float = 0.5) -> List[Symbol]:
    """
    Filter out overlapping symbols using Non-Maximum Suppression.
    
    Args:
        symbols: List of detected symbols
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Filtered list of symbols
    """
    if len(symbols) <= 1:
        return symbols
    
    # Convert to format suitable for NMS
    boxes = []
    scores = []
    
    for symbol in symbols:
        x, y, w, h = symbol.bbox
        boxes.append([x, y, x + w, y + h])  # Convert to x1,y1,x2,y2 format
        scores.append(symbol.conf)
    
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.3, iou_threshold)
    
    # Filter symbols based on NMS results
    if len(indices) > 0:
        filtered_symbols = [symbols[i] for i in indices.flatten()]
    else:
        filtered_symbols = []
    
    return filtered_symbols


def validate_symbol_context(symbols: List[Symbol], img_shape: Tuple[int, int]) -> List[Symbol]:
    """
    Validate symbols based on context and remove false positives.
    
    Args:
        symbols: List of detected symbols
        img_shape: Image shape (H, W)
        
    Returns:
        Validated list of symbols
    """
    validated = []
    
    for symbol in symbols:
        x, y, w, h = symbol.bbox
        
        # Check if symbol is within image bounds
        if x < 0 or y < 0 or x + w > img_shape[1] or y + h > img_shape[0]:
            continue
        
        # Check minimum size
        if w < 3 or h < 3:
            continue
        
        # Context-specific validation could be added here
        # For example, right angles should be near line intersections
        
        validated.append(symbol)
    
    return validated