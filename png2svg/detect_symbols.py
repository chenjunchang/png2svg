"""
Symbol detection for PNG2SVG.

Detects mathematical symbols using YOLO (when available) with rule-based fallbacks.
Handles arrows, right angles, parallel marks, tick marks, and angle arcs.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import uuid

import cv2
import numpy as np

from .config import Config
from .preprocess import PreprocessResult


@dataclass
class Symbol:
    """Detected mathematical symbol with properties."""
    cls: str                             # Symbol class name
    bbox: Tuple[int, int, int, int]     # Bounding box (x, y, w, h)
    confidence: float                   # Detection confidence [0, 1]
    center: Tuple[float, float] = None  # Symbol center point
    angle: Optional[float] = None       # Orientation angle (radians)
    properties: Dict[str, Any] = field(default_factory=dict)  # Additional properties
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def __post_init__(self):
        if self.center is None:
            x, y, w, h = self.bbox
            self.center = (x + w/2, y + h/2)


@dataclass
class SymbolsResult:
    """Result of symbol detection."""
    items: List[Symbol]
    stats: Dict[str, Any] = field(default_factory=dict)


# Symbol class definitions
SYMBOL_CLASSES = {
    'right_angle': {'name': 'Right Angle', 'fallback': True},
    'arc_1': {'name': 'Single Arc', 'fallback': True},
    'arc_2': {'name': 'Double Arc', 'fallback': False},
    'arc_3': {'name': 'Triple Arc', 'fallback': False},
    'tick_1': {'name': 'Single Tick', 'fallback': True},
    'tick_2': {'name': 'Double Tick', 'fallback': False},
    'tick_3': {'name': 'Triple Tick', 'fallback': False},
    'parallel_mark': {'name': 'Parallel Mark', 'fallback': True},
    'perp_mark': {'name': 'Perpendicular Mark', 'fallback': True},
    'arrow_head': {'name': 'Arrow Head', 'fallback': True},
    'dot_filled': {'name': 'Filled Dot', 'fallback': True},
    'dot_hollow': {'name': 'Hollow Dot', 'fallback': False},
    'hatch_region': {'name': 'Hatch Region', 'fallback': False},
    'extend_mark': {'name': 'Extension Mark', 'fallback': False}
}


def run(preprocess_result: PreprocessResult, config: Config) -> SymbolsResult:
    """
    Detect mathematical symbols in the image.
    
    Args:
        preprocess_result: Preprocessed image data
        config: Configuration object
        
    Returns:
        SymbolsResult: Detected symbols with confidence scores
    """
    logger = logging.getLogger(__name__)
    
    symbols = []
    
    # Try YOLO detection first if available and enabled
    if config.use_yolo_symbols and _check_yolo_weights(config.yolo_symbols_weights):
        logger.debug("Using YOLO for symbol detection")
        yolo_symbols = _detect_with_yolo(preprocess_result, config)
        symbols.extend(yolo_symbols)
        logger.debug(f"YOLO detected {len(yolo_symbols)} symbols")
    else:
        logger.debug("YOLO not available, using rule-based detection")
    
    # Add rule-based detection for symbols not well-covered by YOLO
    rule_symbols = _detect_with_rules(preprocess_result, config)
    symbols.extend(rule_symbols)
    logger.debug(f"Rule-based detection found {len(rule_symbols)} additional symbols")
    
    # Apply Test Time Augmentation if enabled
    if config.confidence_tta and len(symbols) < 20:  # Only for reasonable symbol counts
        logger.debug("Applying TTA for symbol detection")
        tta_symbols = _apply_tta_symbols(preprocess_result, config)
        symbols = _merge_tta_symbols(symbols, tta_symbols)
    
    # Remove duplicates and filter by confidence
    symbols = _deduplicate_symbols(symbols)
    symbols = [s for s in symbols if s.confidence > 0.1]
    
    # Calculate statistics
    stats = {
        'total_symbols': len(symbols),
        'symbol_types': {cls: sum(1 for s in symbols if s.cls == cls) for cls in SYMBOL_CLASSES},
        'avg_confidence': np.mean([s.confidence for s in symbols]) if symbols else 0,
        'detection_method': 'yolo' if config.use_yolo_symbols else 'rules'
    }
    
    logger.info(f"Detected {len(symbols)} symbols (avg confidence: {stats['avg_confidence']:.2f})")
    
    return SymbolsResult(items=symbols, stats=stats)


def _check_yolo_weights(weights_path: str) -> bool:
    """Check if YOLO weights file exists."""
    return Path(weights_path).exists()


def _detect_with_yolo(preprocess_result: PreprocessResult, config: Config) -> List[Symbol]:
    """
    Detect symbols using YOLO model.
    
    Args:
        preprocess_result: Preprocessed image data
        config: Configuration object
        
    Returns:
        List[Symbol]: Detected symbols
    """
    symbols = []
    
    try:
        # Try to load and run YOLO model
        if _is_ultralytics_available():
            symbols = _run_ultralytics_yolo(preprocess_result, config)
        elif _is_onnx_available():
            symbols = _run_onnx_yolo(preprocess_result, config)
        else:
            logging.getLogger(__name__).warning("No YOLO runtime available")
            
    except Exception as e:
        logging.getLogger(__name__).warning(f"YOLO detection failed: {e}")
    
    return symbols


def _is_ultralytics_available() -> bool:
    """Check if Ultralytics YOLO is available."""
    try:
        import ultralytics
        return True
    except ImportError:
        return False


def _is_onnx_available() -> bool:
    """Check if ONNX runtime is available."""
    try:
        import onnxruntime
        return True
    except ImportError:
        return False


def _run_ultralytics_yolo(preprocess_result: PreprocessResult, config: Config) -> List[Symbol]:
    """Run YOLO detection using Ultralytics."""
    try:
        from ultralytics import YOLO
        
        # Load model
        model = YOLO(config.yolo_symbols_weights)
        
        # Run inference
        results = model(preprocess_result.img_gray)
        
        symbols = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, score, cls_idx in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    
                    # Get class name
                    cls_name = model.names[int(cls_idx)] if int(cls_idx) < len(model.names) else f"class_{int(cls_idx)}"
                    
                    symbols.append(Symbol(
                        cls=cls_name,
                        bbox=(int(x1), int(y1), int(w), int(h)),
                        confidence=float(score)
                    ))
        
        return symbols
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Ultralytics YOLO failed: {e}")
        return []


def _run_onnx_yolo(preprocess_result: PreprocessResult, config: Config) -> List[Symbol]:
    """Run YOLO detection using ONNX runtime."""
    try:
        import onnxruntime as ort
        
        # Load ONNX model
        session = ort.InferenceSession(config.yolo_symbols_weights)
        
        # Prepare input
        img = preprocess_result.img_gray
        input_size = 640  # Standard YOLO input size
        
        # Resize and normalize
        img_resized = cv2.resize(img, (input_size, input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_transposed = img_normalized.transpose(2, 0, 1)
        img_batch = np.expand_dims(img_transposed, 0)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_batch})
        
        # Parse outputs (simplified YOLO v5/v8 format)
        predictions = outputs[0][0]  # Shape: [num_detections, 6] (x, y, w, h, conf, class)
        
        symbols = []
        h_orig, w_orig = img.shape[:2]
        scale_x, scale_y = w_orig / input_size, h_orig / input_size
        
        for pred in predictions:
            if len(pred) >= 6:
                x_center, y_center, width, height, confidence, class_id = pred[:6]
                
                if confidence > 0.1:  # Confidence threshold
                    # Convert back to original image coordinates
                    x = int((x_center - width/2) * scale_x)
                    y = int((y_center - height/2) * scale_y)
                    w = int(width * scale_x)
                    h = int(height * scale_y)
                    
                    # Map class ID to name (this would need to match training labels)
                    class_names = list(SYMBOL_CLASSES.keys())
                    cls_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f"class_{int(class_id)}"
                    
                    symbols.append(Symbol(
                        cls=cls_name,
                        bbox=(x, y, w, h),
                        confidence=float(confidence)
                    ))
        
        return symbols
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"ONNX YOLO failed: {e}")
        return []


def _detect_with_rules(preprocess_result: PreprocessResult, config: Config) -> List[Symbol]:
    """
    Detect symbols using rule-based methods.
    
    Args:
        preprocess_result: Preprocessed image data
        config: Configuration object
        
    Returns:
        List[Symbol]: Detected symbols
    """
    img_bw = preprocess_result.img_bw
    symbols = []
    
    # Detect right angle markers (small squares)
    right_angles = _detect_right_angles(img_bw)
    symbols.extend(right_angles)
    
    # Detect tick marks (short perpendicular lines)
    tick_marks = _detect_tick_marks(img_bw)
    symbols.extend(tick_marks)
    
    # Detect arrow heads (triangular shapes at line ends)
    arrow_heads = _detect_arrow_heads(img_bw)
    symbols.extend(arrow_heads)
    
    # Detect filled dots (small circular regions)
    dots = _detect_dots(img_bw)
    symbols.extend(dots)
    
    # Detect parallel marks (parallel line pairs)
    parallel_marks = _detect_parallel_marks(img_bw)
    symbols.extend(parallel_marks)
    
    # Detect simple angle arcs
    angle_arcs = _detect_angle_arcs(img_bw)
    symbols.extend(angle_arcs)
    
    return symbols


def _detect_right_angles(img_bw: np.ndarray) -> List[Symbol]:
    """Detect right angle markers (small squares)."""
    symbols = []
    
    try:
        # Find contours
        contours, _ = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (right angle markers are typically small)
            if 5 <= w <= 25 and 5 <= h <= 25:
                # Check if it's roughly square
                aspect_ratio = w / h
                if 0.7 <= aspect_ratio <= 1.3:
                    # Check if it looks like a right angle (L-shape)
                    area_ratio = cv2.contourArea(contour) / (w * h)
                    if 0.3 <= area_ratio <= 0.8:  # Not completely filled
                        symbols.append(Symbol(
                            cls='right_angle',
                            bbox=(x, y, w, h),
                            confidence=0.6
                        ))
    
    except Exception:
        pass
    
    return symbols


def _detect_tick_marks(img_bw: np.ndarray) -> List[Symbol]:
    """Detect tick marks (short perpendicular lines)."""
    symbols = []
    
    try:
        # Use morphological operations to find short line segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))  # Vertical lines
        vertical_ticks = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))  # Horizontal lines
        horizontal_ticks = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel)
        
        # Combine
        tick_candidates = cv2.bitwise_or(vertical_ticks, horizontal_ticks)
        
        # Find contours
        contours, _ = cv2.findContours(tick_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio
            if ((w > 3*h and 5 <= h <= 15) or  # Horizontal tick
                (h > 3*w and 5 <= w <= 15)):    # Vertical tick
                symbols.append(Symbol(
                    cls='tick_1',
                    bbox=(x, y, w, h),
                    confidence=0.5
                ))
    
    except Exception:
        pass
    
    return symbols


def _detect_arrow_heads(img_bw: np.ndarray) -> List[Symbol]:
    """Detect arrow heads (triangular shapes)."""
    symbols = []
    
    try:
        # Find contours
        contours, _ = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check for triangular shape
            if (len(approx) == 3 and  # Triangle
                5 <= w <= 30 and 5 <= h <= 30):  # Reasonable size
                symbols.append(Symbol(
                    cls='arrow_head',
                    bbox=(x, y, w, h),
                    confidence=0.7,
                    angle=_calculate_arrow_angle(approx)
                ))
    
    except Exception:
        pass
    
    return symbols


def _detect_dots(img_bw: np.ndarray) -> List[Symbol]:
    """Detect filled dots (points)."""
    symbols = []
    
    try:
        # Find contours
        contours, _ = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check for small, roughly circular shape
            if (3 <= w <= 15 and 3 <= h <= 15):
                aspect_ratio = w / h
                if 0.7 <= aspect_ratio <= 1.3:  # Roughly square bounding box
                    area_ratio = cv2.contourArea(contour) / (w * h)
                    if area_ratio > 0.5:  # Filled
                        symbols.append(Symbol(
                            cls='dot_filled',
                            bbox=(x, y, w, h),
                            confidence=0.8
                        ))
    
    except Exception:
        pass
    
    return symbols


def _detect_parallel_marks(img_bw: np.ndarray) -> List[Symbol]:
    """Detect parallel marks (pairs of parallel lines)."""
    symbols = []
    
    try:
        # Detect lines
        edges = cv2.Canny(img_bw, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
        
        if lines is not None:
            # Look for pairs of parallel lines that are close together
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    x1, y1, x2, y2 = lines[i][0]
                    x3, y3, x4, y4 = lines[j][0]
                    
                    # Calculate angles
                    angle1 = np.arctan2(y2 - y1, x2 - x1)
                    angle2 = np.arctan2(y4 - y3, x4 - x3)
                    
                    # Check if angles are similar (parallel)
                    angle_diff = abs(angle1 - angle2)
                    if angle_diff > np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    
                    if angle_diff < 0.1:  # ~6 degrees tolerance
                        # Check distance between lines
                        mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
                        mid2 = ((x3 + x4) / 2, (y3 + y4) / 2)
                        distance = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
                        
                        if 5 <= distance <= 25:  # Reasonable spacing for parallel marks
                            # Create bounding box around both lines
                            min_x = min(x1, x2, x3, x4)
                            max_x = max(x1, x2, x3, x4)
                            min_y = min(y1, y2, y3, y4)
                            max_y = max(y1, y2, y3, y4)
                            
                            symbols.append(Symbol(
                                cls='parallel_mark',
                                bbox=(int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)),
                                confidence=0.6
                            ))
    
    except Exception:
        pass
    
    return symbols


def _detect_angle_arcs(img_bw: np.ndarray) -> List[Symbol]:
    """Detect simple angle arcs."""
    symbols = []
    
    try:
        # Find contours
        contours, _ = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Look for arc-like shapes
            if 10 <= w <= 50 and 10 <= h <= 50:
                arc_length = cv2.arcLength(contour, closed=False)
                area = cv2.contourArea(contour)
                
                # Arcs have relatively long perimeter but small area
                if area < (w * h * 0.3) and arc_length > max(w, h) * 1.5:
                    symbols.append(Symbol(
                        cls='arc_1',
                        bbox=(x, y, w, h),
                        confidence=0.4
                    ))
    
    except Exception:
        pass
    
    return symbols


def _calculate_arrow_angle(triangle_points: np.ndarray) -> float:
    """Calculate the pointing direction of an arrow head."""
    try:
        # Find the vertex that's most isolated (the arrow tip)
        points = triangle_points.reshape(-1, 2)
        
        # Calculate distances between all point pairs
        distances = []
        for i in range(3):
            for j in range(i + 1, 3):
                dist = np.linalg.norm(points[i] - points[j])
                distances.append((dist, i, j))
        
        # The tip is opposite to the longest edge
        distances.sort(reverse=True)
        longest_edge_points = distances[0][1:]
        tip_point = 3 - sum(longest_edge_points)  # The remaining point
        
        # Calculate direction from base center to tip
        base_center = (points[longest_edge_points[0]] + points[longest_edge_points[1]]) / 2
        tip = points[tip_point]
        
        angle = np.arctan2(tip[1] - base_center[1], tip[0] - base_center[0])
        return angle
        
    except Exception:
        return 0.0


def _apply_tta_symbols(preprocess_result: PreprocessResult, config: Config) -> List[List[Symbol]]:
    """Apply Test Time Augmentation for symbol detection."""
    tta_results = []
    
    # Horizontal flip
    img_flipped = cv2.flip(preprocess_result.img_bw, 1)
    flipped_preprocess = PreprocessResult(
        img_gray=cv2.flip(preprocess_result.img_gray, 1),
        img_bw=img_flipped,
        transform=preprocess_result.transform,
        skew_angle=preprocess_result.skew_angle,
        original_size=preprocess_result.original_size
    )
    
    if config.use_yolo_symbols and _check_yolo_weights(config.yolo_symbols_weights):
        flipped_symbols = _detect_with_yolo(flipped_preprocess, config)
    else:
        flipped_symbols = _detect_with_rules(flipped_preprocess, config)
    
    # Transform coordinates back
    w = preprocess_result.original_size[0]
    for symbol in flipped_symbols:
        x, y, box_w, box_h = symbol.bbox
        symbol.bbox = (w - x - box_w, y, box_w, box_h)
        symbol.center = (w - symbol.center[0], symbol.center[1])
    
    tta_results.append(flipped_symbols)
    
    return tta_results


def _merge_tta_symbols(original: List[Symbol], tta_results: List[List[Symbol]]) -> List[Symbol]:
    """Merge TTA results with original symbols."""
    merged = original.copy()
    
    for tta_batch in tta_results:
        for tta_symbol in tta_batch:
            # Check for overlaps
            overlaps = False
            for existing in merged:
                if (_symbol_boxes_overlap(tta_symbol.bbox, existing.bbox) and
                    tta_symbol.cls == existing.cls):
                    # Boost confidence
                    existing.confidence = min(1.0, existing.confidence + 0.1)
                    overlaps = True
                    break
            
            if not overlaps and tta_symbol.confidence > 0.3:
                merged.append(tta_symbol)
    
    return merged


def _symbol_boxes_overlap(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> bool:
    """Check if two symbol bounding boxes overlap."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)


def _deduplicate_symbols(symbols: List[Symbol]) -> List[Symbol]:
    """Remove duplicate symbols based on overlap and class."""
    if not symbols:
        return symbols
    
    # Sort by confidence (highest first)
    symbols.sort(key=lambda x: x.confidence, reverse=True)
    
    deduplicated = []
    for symbol in symbols:
        # Check if this symbol overlaps with any already accepted symbol of the same class
        overlaps = False
        for existing in deduplicated:
            if (symbol.cls == existing.cls and
                _symbol_boxes_overlap(symbol.bbox, existing.bbox)):
                overlaps = True
                break
        
        if not overlaps:
            deduplicated.append(symbol)
    
    return deduplicated