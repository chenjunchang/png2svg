"""
Primitive detection module for PNG2SVG system.
Detects basic geometric primitives: lines, circles, arcs.
"""

import cv2
import numpy as np
import math
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from sklearn.cluster import DBSCAN
import uuid

from .config import Config
from .preprocess import PreprocessOut


@dataclass
class LineSeg:
    """Line segment representation with attributes."""
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    dashed: bool = False
    thickness: int = 2
    role: str = "main"       # main/aux/hidden (based on thickness/dash style)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    confidence: float = 1.0


@dataclass
class CircleArc:
    """Circle/Arc representation."""
    cx: float
    cy: float
    r: float
    theta1: float = 0.0
    theta2: float = 360.0    # Arc segment angles in degrees
    kind: str = "circle"     # circle/arc
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    confidence: float = 1.0


@dataclass
class Primitives:
    """Container for all detected primitives."""
    lines: List[LineSeg] = field(default_factory=list)
    circles: List[CircleArc] = field(default_factory=list)


def run(pre: PreprocessOut, cfg: Config) -> Primitives:
    """
    Run primitive detection on preprocessed image.
    
    - LSD first, fallback to HoughLinesP
    - Merge collinear segments
    - HoughCircles for circle detection
    - Arc fitting on edges with RANSAC/least squares
    - Dashed line detection via pixel sampling
    
    Args:
        pre: Preprocessed images and metadata
        cfg: Configuration object
        
    Returns:
        Primitives: Detected geometric primitives
    """
    logger = logging.getLogger('png2svg.detect_primitives')
    logger.info("Starting primitive detection")
    
    # Detect lines
    lines = detect_lines(pre.img_gray, pre.img_bw, cfg)
    logger.info(f"Detected {len(lines)} line segments")
    
    # Detect circles and arcs
    circles = detect_circles_and_arcs(pre.img_gray, pre.img_bw, cfg)
    logger.info(f"Detected {len(circles)} circles/arcs")
    
    # Classify dashed lines
    for line in lines:
        line.dashed = is_dashed_line(line, pre.img_bw, cfg)
        line.role = "aux" if line.dashed else "main"
    
    dashed_count = sum(1 for line in lines if line.dashed)
    logger.info(f"Classified {dashed_count} dashed lines")
    
    return Primitives(lines=lines, circles=circles)


def detect_lines(img_gray: np.ndarray, img_bw: np.ndarray, cfg: Config) -> List[LineSeg]:
    """
    Detect line segments using LSD or HoughLinesP.
    
    Args:
        img_gray: Grayscale image
        img_bw: Binary image
        cfg: Configuration object
        
    Returns:
        List of detected line segments
    """
    logger = logging.getLogger('png2svg.detect_primitives')
    lines = []
    
    # Try LSD first (Line Segment Detector)
    try:
        lines = detect_lines_lsd(img_gray, cfg)
        if lines:
            logger.debug(f"LSD detected {len(lines)} lines")
        else:
            logger.debug("LSD found no lines, trying HoughLinesP")
    except Exception as e:
        logger.warning(f"LSD failed: {e}, falling back to HoughLinesP")
    
    # Fallback to HoughLinesP
    if not lines:
        lines = detect_lines_hough(img_gray, cfg)
        logger.debug(f"HoughLinesP detected {len(lines)} lines")
    
    # Merge collinear segments
    if lines:
        lines = merge_collinear_segments(lines, cfg)
        logger.debug(f"After merging: {len(lines)} lines")
    
    return lines


def detect_lines_lsd(img_gray: np.ndarray, cfg: Config) -> List[LineSeg]:
    """
    Detect lines using Line Segment Detector (LSD).
    
    Args:
        img_gray: Grayscale image
        cfg: Configuration object
        
    Returns:
        List of line segments
    """
    try:
        # Create LSD detector with positional arguments for OpenCV 4.x compatibility
        lsd = cv2.createLineSegmentDetector(
            cv2.LSD_REFINE_STD if cfg.algorithms.lsd.refine else cv2.LSD_REFINE_NONE,
            cfg.algorithms.lsd.scale,
            cfg.algorithms.lsd.sigma,
            cfg.algorithms.lsd.quant,
            cfg.algorithms.lsd.ang_th,
            cfg.algorithms.lsd.log_eps,
            cfg.algorithms.lsd.density_th,
            cfg.algorithms.lsd.n_bins
        )
        
        # Detect lines
        lines_lsd = lsd.detect(img_gray)
        
        if lines_lsd[0] is not None:
            lines = []
            for line in lines_lsd[0]:
                x1, y1, x2, y2 = line[0]
                # Filter by minimum length
                length = math.hypot(x2 - x1, y2 - y1)
                if length >= cfg.min_line_len:
                    lines.append(LineSeg(
                        p1=(float(x1), float(y1)),
                        p2=(float(x2), float(y2)),
                        confidence=0.9  # LSD is generally high confidence
                    ))
            return lines
        
    except Exception as e:
        raise RuntimeError(f"LSD detection failed: {e}")
    
    return []


def detect_lines_hough(img_gray: np.ndarray, cfg: Config) -> List[LineSeg]:
    """
    Detect lines using Hough Line Transform.
    
    Args:
        img_gray: Grayscale image  
        cfg: Configuration object
        
    Returns:
        List of line segments
    """
    # Edge detection
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    
    # Hough line detection
    lines_hough = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=cfg.algorithms.hough_lines.threshold,
        minLineLength=cfg.algorithms.hough_lines.min_line_length,
        maxLineGap=cfg.algorithms.hough_lines.max_line_gap
    )
    
    lines = []
    if lines_hough is not None:
        for line in lines_hough:
            x1, y1, x2, y2 = line[0]
            length = math.hypot(x2 - x1, y2 - y1)
            if length >= cfg.min_line_len:
                lines.append(LineSeg(
                    p1=(float(x1), float(y1)),
                    p2=(float(x2), float(y2)),
                    confidence=0.7  # Hough is medium confidence
                ))
    
    return lines


def merge_collinear_segments(lines: List[LineSeg], cfg: Config) -> List[LineSeg]:
    """
    Merge collinear line segments that are close to each other.
    
    Args:
        lines: List of line segments
        cfg: Configuration object
        
    Returns:
        List of merged line segments
    """
    if len(lines) <= 1:
        return lines
    
    # Convert lines to features for clustering
    features = []
    for line in lines:
        x1, y1 = line.p1
        x2, y2 = line.p2
        
        # Calculate line parameters
        angle = math.atan2(y2 - y1, x2 - x1)
        length = math.hypot(x2 - x1, y2 - y1)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Normalize angle to [0, pi]
        if angle < 0:
            angle += math.pi
        
        features.append([angle, center_x / 100, center_y / 100])  # Scale positions
    
    # Cluster lines by angle and position
    features = np.array(features)
    eps = cfg.line_merge_angle_deg * math.pi / 180  # Convert to radians
    
    try:
        clustering = DBSCAN(eps=eps, min_samples=1).fit(features)
        labels = clustering.labels_
    except:
        # If clustering fails, return original lines
        return lines
    
    # Merge lines in same cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    merged_lines = []
    for cluster_indices in clusters.values():
        if len(cluster_indices) == 1:
            # Single line, keep as is
            merged_lines.append(lines[cluster_indices[0]])
        else:
            # Multiple lines, merge them
            merged_line = merge_line_cluster([lines[i] for i in cluster_indices], cfg)
            if merged_line:
                merged_lines.append(merged_line)
    
    return merged_lines


def merge_line_cluster(cluster_lines: List[LineSeg], cfg: Config) -> Optional[LineSeg]:
    """
    Merge a cluster of collinear line segments.
    
    Args:
        cluster_lines: List of lines to merge
        cfg: Configuration object
        
    Returns:
        Merged line segment or None if merge fails
    """
    if not cluster_lines:
        return None
    
    if len(cluster_lines) == 1:
        return cluster_lines[0]
    
    # Collect all endpoints
    points = []
    for line in cluster_lines:
        points.extend([line.p1, line.p2])
    
    # Find the extreme points
    points = np.array(points)
    
    # Fit a line to all points
    try:
        # Use PCA to find the best fitting line
        centroid = np.mean(points, axis=0)
        _, _, vt = np.linalg.svd(points - centroid)
        direction = vt[0]  # First principal component
        
        # Project all points onto the line
        projections = np.dot(points - centroid, direction)
        
        # Find extreme projections
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        
        # Calculate extreme points
        p1 = centroid + min_proj * direction
        p2 = centroid + max_proj * direction
        
        # Create merged line
        merged_line = LineSeg(
            p1=(float(p1[0]), float(p1[1])),
            p2=(float(p2[0]), float(p2[1])),
            confidence=np.mean([line.confidence for line in cluster_lines])
        )
        
        return merged_line
        
    except Exception:
        # If merge fails, return the longest line
        longest_line = max(cluster_lines, key=lambda l: math.hypot(
            l.p2[0] - l.p1[0], l.p2[1] - l.p1[1]
        ))
        return longest_line


def detect_circles_and_arcs(img_gray: np.ndarray, img_bw: np.ndarray, cfg: Config) -> List[CircleArc]:
    """
    Detect circles and arcs using HoughCircles and edge fitting.
    
    Args:
        img_gray: Grayscale image
        img_bw: Binary image
        cfg: Configuration object
        
    Returns:
        List of detected circles and arcs
    """
    logger = logging.getLogger('png2svg.detect_primitives')
    circles = []
    
    # Detect full circles first
    circles_hough = detect_circles_hough(img_gray, cfg)
    circles.extend(circles_hough)
    logger.debug(f"HoughCircles detected {len(circles_hough)} circles")
    
    # Add arc detection using edge fitting
    arcs = detect_arcs(img_gray, img_bw, cfg)
    circles.extend(arcs)
    logger.debug(f"Arc detection found {len(arcs)} arcs")
    
    return circles


def detect_circles_hough(img_gray: np.ndarray, cfg: Config) -> List[CircleArc]:
    """
    Detect circles using Hough Circle Transform.
    
    Args:
        img_gray: Grayscale image
        cfg: Configuration object
        
    Returns:
        List of detected circles
    """
    circles = []
    
    # Apply Hough Circle Transform
    circles_hough = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=cfg.algorithms.hough_circles.dp,
        minDist=cfg.algorithms.hough_circles.min_dist,
        param1=cfg.algorithms.hough_circles.param1,
        param2=cfg.algorithms.hough_circles.param2,
        minRadius=cfg.algorithms.hough_circles.min_radius,
        maxRadius=cfg.algorithms.hough_circles.max_radius
    )
    
    if circles_hough is not None:
        circles_hough = np.round(circles_hough[0, :]).astype("int")
        
        for (x, y, r) in circles_hough:
            circles.append(CircleArc(
                cx=float(x),
                cy=float(y), 
                r=float(r),
                kind="circle",
                confidence=0.8
            ))
    
    return circles


def is_dashed_line(line: LineSeg, img_bw: np.ndarray, cfg: Config = None) -> bool:
    """
    Determine if a line segment represents a dashed line by sampling pixels.
    Enhanced algorithm with better noise handling and pattern recognition.
    
    Args:
        line: Line segment to analyze
        img_bw: Binary image
        
    Returns:
        True if line appears to be dashed
    """
    x1, y1 = line.p1
    x2, y2 = line.p2
    
    # Calculate line length
    length = math.hypot(x2 - x1, y2 - y1)
    
    if length < 20:  # Too short to determine dash pattern
        return False
    
    # Adaptive sampling based on line length
    num_samples = max(int(length * 0.8), 20)
    num_samples = min(num_samples, 200)  # Cap to avoid excessive computation
    
    # Get pixel values along the line with better sampling
    x_samples = np.linspace(x1, x2, num_samples)
    y_samples = np.linspace(y1, y2, num_samples)
    
    # Clamp coordinates to image bounds
    h, w = img_bw.shape
    x_samples = np.clip(x_samples, 0, w - 1).astype(int)
    y_samples = np.clip(y_samples, 0, h - 1).astype(int)
    
    # Sample pixel values directly (more sensitive to gaps)
    values = []
    for x, y in zip(x_samples, y_samples):
        # Direct pixel sampling - more sensitive to dashed line gaps
        values.append(img_bw[y, x] > 0)
    
    values = np.array(values)
    
    # Apply light smoothing to reduce noise
    if len(values) >= 5:
        # Simple median filter
        smoothed = np.zeros_like(values)
        for i in range(len(values)):
            start = max(0, i-1)
            end = min(len(values), i+2)
            smoothed[i] = np.median(values[start:end]) > 0.5
        values = smoothed
    
    # Analyze run lengths
    runs = get_run_lengths(values)
    
    if not runs:
        return False
    
    # Separate ON and OFF runs
    on_runs = [run_len for is_on, run_len in runs if is_on]
    off_runs = [run_len for is_on, run_len in runs if not is_on]
    
    # Get configuration parameters or use defaults (updated for mathematical diagrams)
    min_dash_count = 1
    regularity_cv = 2.0
    on_ratio_min = 0.05
    on_ratio_max = 0.85
    min_gap_length = 1
    dash_gap_ratio_min = 0.01
    dash_gap_ratio_max = 10.0
    
    if cfg and hasattr(cfg.algorithms, 'dashed_line'):
        dl_cfg = cfg.algorithms.dashed_line
        min_dash_count = getattr(dl_cfg, 'min_dash_count', min_dash_count)
        regularity_cv = getattr(dl_cfg, 'regularity_cv', regularity_cv)
        on_ratio_min = getattr(dl_cfg, 'on_ratio_min', on_ratio_min)
        on_ratio_max = getattr(dl_cfg, 'on_ratio_max', on_ratio_max)
        min_gap_length = getattr(dl_cfg, 'min_gap_length', min_gap_length)
        dash_gap_ratio_min = getattr(dl_cfg, 'dash_gap_ratio_min', dash_gap_ratio_min)
        dash_gap_ratio_max = getattr(dl_cfg, 'dash_gap_ratio_max', dash_gap_ratio_max)

    # Enhanced criteria for dashed line detection:
    # 1. At least min_dash_count ON segments and 1 OFF segment
    if len(on_runs) < min_dash_count or len(off_runs) < 1:
        return False
    
    # 2. ON segments should be substantial (not just noise)
    mean_on_length = np.mean(on_runs)
    if mean_on_length < 1.5:  # Allow smaller dashes for fine mathematical diagrams
        return False
    
    # 3. Check for reasonable gap lengths - allow very small gaps for fine dashed lines
    mean_off_length = np.mean(off_runs)
    # For mathematical diagrams, gaps can be very small (even 1 pixel)
    if mean_off_length < 1.0:  
        return False
    
    # 4. Check ratio of ON to total length
    total_on_length = sum(on_runs)
    on_ratio = total_on_length / len(values)
    
    if on_ratio < on_ratio_min or on_ratio > on_ratio_max:
        return False
    
    # 5. Check for pattern regularity (dashes should be somewhat regular)
    if len(on_runs) >= 3:
        on_std = np.std(on_runs)
        on_cv = on_std / mean_on_length if mean_on_length > 0 else float('inf')
        if on_cv > regularity_cv:
            return False
    
    # 6. Check gap regularity
    if len(off_runs) >= 2:
        off_std = np.std(off_runs)
        off_cv = off_std / mean_off_length if mean_off_length > 0 else float('inf')
        if off_cv > regularity_cv:
            return False
    
    # 7. Final check: ensure dashes and gaps are in reasonable proportion
    dash_gap_ratio = mean_on_length / mean_off_length
    if dash_gap_ratio < dash_gap_ratio_min or dash_gap_ratio > dash_gap_ratio_max:
        return False
    
    return True


def get_run_lengths(binary_array: np.ndarray) -> List[Tuple[bool, int]]:
    """
    Get run lengths from binary array.
    
    Args:
        binary_array: Boolean array
        
    Returns:
        List of (value, length) tuples
    """
    if len(binary_array) == 0:
        return []
    
    runs = []
    current_value = binary_array[0]
    current_length = 1
    
    for value in binary_array[1:]:
        if value == current_value:
            current_length += 1
        else:
            runs.append((current_value, current_length))
            current_value = value
            current_length = 1
    
    # Add the last run
    runs.append((current_value, current_length))
    
    return runs


def detect_arcs(img_gray: np.ndarray, img_bw: np.ndarray, cfg: Config) -> List[CircleArc]:
    """
    Detect arcs using edge contour fitting.
    
    Args:
        img_gray: Grayscale image
        img_bw: Binary image
        cfg: Configuration object
        
    Returns:
        List of detected arcs
    """
    arcs = []
    
    try:
        # Find contours in binary image
        contours, _ = cv2.findContours(img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get configuration parameters or use defaults
        min_contour_points = 5
        min_contour_area = 50
        coverage_threshold = 0.9
        aspect_ratio_min = 0.6
        
        if cfg and hasattr(cfg.algorithms, 'arc_detection'):
            arc_cfg = cfg.algorithms.arc_detection
            min_contour_points = getattr(arc_cfg, 'min_contour_points', min_contour_points)
            min_contour_area = getattr(arc_cfg, 'min_contour_area', min_contour_area)
            coverage_threshold = getattr(arc_cfg, 'coverage_threshold', coverage_threshold)
            aspect_ratio_min = getattr(arc_cfg, 'aspect_ratio_min', aspect_ratio_min)

        for contour in contours:
            # Need at least min_contour_points to fit a circle/ellipse
            if len(contour) < min_contour_points:
                continue
            
            # Skip very small contours
            if cv2.contourArea(contour) < min_contour_area:
                continue
            
            # Try to fit a circle using least squares
            circle = fit_circle_to_contour(contour)
            if circle is None:
                continue
            
            cx, cy, r = circle
            
            # Filter by reasonable radius
            min_radius = getattr(cfg.algorithms.hough_circles, 'min_radius', 8)
            max_radius = getattr(cfg.algorithms.hough_circles, 'max_radius', 0)
            
            if r < min_radius:
                continue
            if max_radius > 0 and r > max_radius:
                continue
            
            # Calculate angular coverage to distinguish arcs from circles
            coverage, start_angle, end_angle = calculate_angular_coverage(contour, circle)
            
            # If coverage is less than threshold, consider it an arc
            if coverage < coverage_threshold:
                arcs.append(CircleArc(
                    cx=float(cx),
                    cy=float(cy),
                    r=float(r),
                    theta1=float(start_angle),
                    theta2=float(end_angle),
                    kind="arc",
                    confidence=0.7
                ))
            else:
                # Full circle - only add if not already detected by HoughCircles
                # This is a fallback for circles missed by Hough
                arcs.append(CircleArc(
                    cx=float(cx),
                    cy=float(cy),
                    r=float(r),
                    kind="circle",
                    confidence=0.6  # Lower confidence than Hough-detected circles
                ))
    
    except Exception as e:
        logger = logging.getLogger('png2svg.detect_primitives')
        logger.warning(f"Arc detection failed: {e}")
    
    return arcs


def fit_circle_to_contour(contour: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Fit a circle to a contour using least squares method.
    
    Args:
        contour: Contour points
        
    Returns:
        (cx, cy, r) or None if fitting fails
    """
    try:
        # Convert contour to points
        points = contour.reshape(-1, 2).astype(np.float32)
        
        if len(points) < 3:
            return None
        
        # Use cv2.fitEllipse as approximation for circle fitting
        ellipse = cv2.fitEllipse(points)
        
        # Extract center and average radius
        (cx, cy), (w, h), angle = ellipse
        r = (w + h) / 4.0  # Average of semi-axes
        
        # Only accept if it's reasonably circular (not too elliptical)
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        # Use default threshold since we don't have access to config here
        if aspect_ratio < 0.6:  # Too elliptical
            return None
        
        return float(cx), float(cy), float(r)
    
    except Exception:
        return None


def calculate_angular_coverage(contour: np.ndarray, circle: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Calculate how much of a circle is covered by a contour.
    
    Args:
        contour: Contour points
        circle: (cx, cy, r) circle parameters
        
    Returns:
        (coverage_ratio, start_angle, end_angle) in degrees
    """
    cx, cy, r = circle
    points = contour.reshape(-1, 2)
    
    # Calculate angles for all contour points
    angles = []
    for x, y in points:
        angle = math.degrees(math.atan2(y - cy, x - cx))
        if angle < 0:
            angle += 360
        angles.append(angle)
    
    if not angles:
        return 0.0, 0.0, 0.0
    
    angles = sorted(angles)
    
    # Find the largest gap between consecutive angles
    max_gap = 0
    gap_start = 0
    
    for i in range(len(angles)):
        next_i = (i + 1) % len(angles)
        gap = (angles[next_i] - angles[i]) % 360
        if gap > max_gap:
            max_gap = gap
            gap_start = angles[next_i]
    
    # Coverage is 1 minus the largest gap ratio
    coverage = 1.0 - (max_gap / 360.0)
    
    # Arc spans from end of gap to start of gap
    start_angle = gap_start % 360
    end_angle = (gap_start - max_gap) % 360
    
    return coverage, start_angle, end_angle


def filter_primitives_by_size(primitives: Primitives, cfg: Config) -> Primitives:
    """
    Filter out primitives that are too small or too large.
    
    Args:
        primitives: Input primitives
        cfg: Configuration object
        
    Returns:
        Filtered primitives
    """
    filtered_lines = []
    for line in primitives.lines:
        length = math.hypot(
            line.p2[0] - line.p1[0], 
            line.p2[1] - line.p1[1]
        )
        if length >= cfg.min_line_len:
            filtered_lines.append(line)
    
    filtered_circles = []
    for circle in primitives.circles:
        min_r = cfg.algorithms.hough_circles.min_radius
        max_r = cfg.algorithms.hough_circles.max_radius
        if max_r == 0 or (min_r <= circle.r <= max_r):
            filtered_circles.append(circle)
    
    return Primitives(lines=filtered_lines, circles=filtered_circles)