"""
Geometric primitive detection for PNG2SVG.

Detects lines, circles, and arcs using traditional computer vision techniques.
Includes dash pattern recognition and geometric element classification.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import uuid

import cv2
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from .config import Config
from .preprocess import PreprocessResult


@dataclass
class LineSeg:
    """Line segment with properties and metadata."""
    p1: Tuple[float, float]      # Start point
    p2: Tuple[float, float]      # End point
    dashed: bool = False         # Is this a dashed line?
    thickness: int = 2           # Visual thickness
    role: str = "main"           # "main", "aux", or "hidden"
    confidence: float = 1.0      # Detection confidence
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    @property
    def length(self) -> float:
        """Calculate line segment length."""
        return math.sqrt((self.p2[0] - self.p1[0])**2 + (self.p2[1] - self.p1[1])**2)
    
    @property
    def angle(self) -> float:
        """Calculate line angle in radians."""
        return math.atan2(self.p2[1] - self.p1[1], self.p2[0] - self.p1[0])
    
    @property
    def center(self) -> Tuple[float, float]:
        """Calculate line segment center point."""
        return ((self.p1[0] + self.p2[0]) / 2, (self.p1[1] + self.p2[1]) / 2)


@dataclass
class CircleArc:
    """Circle or arc with properties and metadata."""
    cx: float                    # Center x
    cy: float                    # Center y  
    r: float                     # Radius
    theta1: float = 0.0          # Start angle (radians) for arcs
    theta2: float = 2 * math.pi  # End angle (radians) for arcs
    kind: str = "circle"         # "circle" or "arc"
    confidence: float = 1.0      # Detection confidence
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point."""
        return (self.cx, self.cy)
    
    @property
    def is_full_circle(self) -> bool:
        """Check if this represents a full circle."""
        return self.kind == "circle" or abs(self.theta2 - self.theta1 - 2*math.pi) < 0.1


@dataclass
class PrimitivesResult:
    """Result of primitive detection."""
    lines: List[LineSeg]
    circles: List[CircleArc]
    stats: dict = field(default_factory=dict)


def run(preprocess_result: PreprocessResult, config: Config) -> PrimitivesResult:
    """
    Detect geometric primitives in preprocessed image.
    
    Args:
        preprocess_result: Result from preprocessing
        config: Configuration object
        
    Returns:
        PrimitivesResult: Detected lines and circles
    """
    logger = logging.getLogger(__name__)
    
    img_gray = preprocess_result.img_gray
    img_bw = preprocess_result.img_bw
    
    # Detect lines
    logger.debug("Detecting lines...")
    lines = detect_lines(img_gray, img_bw, config)
    logger.debug(f"Found {len(lines)} raw line segments")
    
    # Merge collinear lines
    lines = merge_collinear_lines(lines, config)
    logger.debug(f"After merging: {len(lines)} line segments")
    
    # Detect dashed patterns
    for line in lines:
        line.dashed = is_dashed_line(line, img_bw)
        line.role = "aux" if line.dashed else "main"
    
    # Detect circles
    logger.debug("Detecting circles...")
    circles = detect_circles(img_gray, config)
    logger.debug(f"Found {len(circles)} circles")
    
    # Detect arcs (if any circles are actually partial arcs)
    arcs = detect_arcs(img_gray, img_bw, config)
    logger.debug(f"Found {len(arcs)} arcs")
    circles.extend(arcs)
    
    # Calculate statistics
    stats = {
        'total_lines': len(lines),
        'dashed_lines': sum(1 for l in lines if l.dashed),
        'total_circles': len([c for c in circles if c.is_full_circle]),
        'total_arcs': len([c for c in circles if not c.is_full_circle]),
        'avg_line_length': np.mean([l.length for l in lines]) if lines else 0
    }
    
    logger.info(f"Detected {len(lines)} lines ({stats['dashed_lines']} dashed) "
               f"and {len(circles)} circles/arcs")
    
    return PrimitivesResult(lines=lines, circles=circles, stats=stats)


def detect_lines(img_gray: np.ndarray, img_bw: np.ndarray, config: Config) -> List[LineSeg]:
    """
    Detect line segments using LSD with HoughLinesP fallback.
    
    Args:
        img_gray: Grayscale image
        img_bw: Binary image
        config: Configuration object
        
    Returns:
        List[LineSeg]: Detected line segments
    """
    lines = []
    
    # Try LSD (Line Segment Detector) first
    try:
        lsd = cv2.createLineSegmentDetector()
        detected = lsd.detect(img_gray)[0]
        
        if detected is not None and len(detected) > 0:
            for x1, y1, x2, y2 in detected.reshape(-1, 4):
                length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length >= config.min_line_len:
                    lines.append(LineSeg(
                        p1=(float(x1), float(y1)),
                        p2=(float(x2), float(y2)),
                        confidence=0.9
                    ))
    except Exception as e:
        logging.getLogger(__name__).warning(f"LSD failed: {e}")
    
    # Fallback to HoughLinesP if LSD didn't find enough lines
    if len(lines) < 5:  # Arbitrary threshold
        edges = cv2.Canny(img_gray, 60, 120, apertureSize=3)
        hough_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=config.min_line_len,
            maxLineGap=4
        )
        
        if hough_lines is not None:
            for x1, y1, x2, y2 in hough_lines.reshape(-1, 4):
                lines.append(LineSeg(
                    p1=(float(x1), float(y1)),
                    p2=(float(x2), float(y2)),
                    confidence=0.7
                ))
    
    return lines


def detect_circles(img_gray: np.ndarray, config: Config) -> List[CircleArc]:
    """
    Detect circles using HoughCircles.
    
    Args:
        img_gray: Grayscale image
        config: Configuration object
        
    Returns:
        List[CircleArc]: Detected circles
    """
    circles = []
    
    # Apply HoughCircles
    detected = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=120,
        param2=30,
        minRadius=8,
        maxRadius=0  # No maximum radius limit
    )
    
    if detected is not None:
        detected = np.uint16(np.around(detected))
        for x, y, r in detected[0, :]:
            circles.append(CircleArc(
                cx=float(x),
                cy=float(y),
                r=float(r),
                kind="circle",
                confidence=0.8
            ))
    
    return circles


def detect_arcs(img_gray: np.ndarray, img_bw: np.ndarray, config: Config) -> List[CircleArc]:
    """
    Detect circular arcs by fitting circles to edge segments.
    
    Args:
        img_gray: Grayscale image
        img_bw: Binary image
        config: Configuration object
        
    Returns:
        List[CircleArc]: Detected arcs
    """
    arcs = []
    
    try:
        # Find contours in binary image
        contours, _ = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) < 5:  # Need at least 5 points to fit a circle
                continue
            
            # Fit circle to contour
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            
            # Check if this is a good arc candidate
            contour_length = cv2.arcLength(contour, closed=False)
            expected_circumference = 2 * math.pi * radius
            
            # If contour is much shorter than full circle, it might be an arc
            if (contour_length / expected_circumference < 0.8 and 
                contour_length > 20 and  # Minimum arc length
                radius > 10):  # Minimum radius
                
                # Estimate arc start/end angles
                points = contour.reshape(-1, 2)
                angles = [math.atan2(p[1] - cy, p[0] - cx) for p in points]
                
                # Normalize angles to [0, 2π]
                angles = [(a + 2*math.pi) % (2*math.pi) for a in angles]
                angles.sort()
                
                theta1 = angles[0]
                theta2 = angles[-1]
                
                # Handle angle wraparound
                if theta2 - theta1 > math.pi:
                    theta1, theta2 = theta2, theta1 + 2*math.pi
                
                arcs.append(CircleArc(
                    cx=float(cx),
                    cy=float(cy),
                    r=float(radius),
                    theta1=theta1,
                    theta2=theta2,
                    kind="arc",
                    confidence=0.6
                ))
    
    except Exception as e:
        logging.getLogger(__name__).warning(f"Arc detection failed: {e}")
    
    return arcs


def merge_collinear_lines(lines: List[LineSeg], config: Config) -> List[LineSeg]:
    """
    Merge collinear line segments that are close together.
    
    Args:
        lines: List of line segments
        config: Configuration object
        
    Returns:
        List[LineSeg]: Merged line segments
    """
    if not lines:
        return lines
    
    merged = []
    used = set()
    
    angle_threshold = math.radians(config.line_merge_angle_deg)
    gap_threshold = config.line_merge_gap_px
    
    for i, line1 in enumerate(lines):
        if i in used:
            continue
        
        # Start with current line
        merged_line = LineSeg(
            p1=line1.p1,
            p2=line1.p2,
            dashed=line1.dashed,
            thickness=line1.thickness,
            confidence=line1.confidence
        )
        used.add(i)
        
        # Try to merge with other lines
        changed = True
        while changed:
            changed = False
            for j, line2 in enumerate(lines):
                if j in used:
                    continue
                
                if can_merge_lines(merged_line, line2, angle_threshold, gap_threshold):
                    merged_line = merge_two_lines(merged_line, line2)
                    used.add(j)
                    changed = True
        
        merged.append(merged_line)
    
    return merged


def can_merge_lines(
    line1: LineSeg, 
    line2: LineSeg, 
    angle_threshold: float, 
    gap_threshold: float
) -> bool:
    """
    Check if two lines can be merged based on angle and proximity.
    
    Args:
        line1: First line segment
        line2: Second line segment
        angle_threshold: Maximum angle difference (radians)
        gap_threshold: Maximum gap distance (pixels)
        
    Returns:
        bool: True if lines can be merged
    """
    # Check angle similarity
    angle_diff = abs(line1.angle - line2.angle)
    angle_diff = min(angle_diff, math.pi - angle_diff)  # Handle wraparound
    
    if angle_diff > angle_threshold:
        return False
    
    # Check proximity (minimum distance between line endpoints)
    distances = [
        math.sqrt((line1.p1[0] - line2.p1[0])**2 + (line1.p1[1] - line2.p1[1])**2),
        math.sqrt((line1.p1[0] - line2.p2[0])**2 + (line1.p1[1] - line2.p2[1])**2),
        math.sqrt((line1.p2[0] - line2.p1[0])**2 + (line1.p2[1] - line2.p1[1])**2),
        math.sqrt((line1.p2[0] - line2.p2[0])**2 + (line1.p2[1] - line2.p2[1])**2),
    ]
    
    return min(distances) <= gap_threshold


def merge_two_lines(line1: LineSeg, line2: LineSeg) -> LineSeg:
    """
    Merge two line segments into one.
    
    Args:
        line1: First line segment
        line2: Second line segment
        
    Returns:
        LineSeg: Merged line segment
    """
    # Find all endpoints
    points = [line1.p1, line1.p2, line2.p1, line2.p2]
    
    # Find the two points that are farthest apart
    max_distance = 0
    best_endpoints = (points[0], points[1])
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = math.sqrt(
                (points[i][0] - points[j][0])**2 + 
                (points[i][1] - points[j][1])**2
            )
            if distance > max_distance:
                max_distance = distance
                best_endpoints = (points[i], points[j])
    
    return LineSeg(
        p1=best_endpoints[0],
        p2=best_endpoints[1],
        dashed=line1.dashed or line2.dashed,
        thickness=max(line1.thickness, line2.thickness),
        confidence=(line1.confidence + line2.confidence) / 2
    )


def is_dashed_line(line: LineSeg, img_bw: np.ndarray) -> bool:
    """
    Determine if a line segment represents a dashed line.
    
    Args:
        line: Line segment to analyze
        img_bw: Binary image
        
    Returns:
        bool: True if line appears to be dashed
    """
    try:
        x1, y1 = int(line.p1[0]), int(line.p1[1])
        x2, y2 = int(line.p2[0]), int(line.p2[1])
        
        length = int(line.length)
        if length < 20:
            return False
        
        # Sample points along the line
        num_samples = min(length, 100)
        xs = np.linspace(x1, x2, num_samples, dtype=int)
        ys = np.linspace(y1, y2, num_samples, dtype=int)
        
        # Clamp coordinates to image bounds
        h, w = img_bw.shape
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)
        
        # Get pixel values along the line
        values = img_bw[ys, xs]
        
        # Analyze run-length encoding to detect dash pattern
        runs = []
        current_val = values[0]
        current_run = 1
        
        for val in values[1:]:
            if val == current_val:
                current_run += 1
            else:
                runs.append((current_val, current_run))
                current_val = val
                current_run = 1
        runs.append((current_val, current_run))
        
        # Analyze runs for dash pattern
        on_runs = [length for val, length in runs if val > 0]  # Foreground runs
        off_runs = [length for val, length in runs if val == 0]  # Background runs
        
        # Criteria for dashed line:
        # 1. At least 2 "on" segments
        # 2. At least 1 "off" segment 
        # 3. Average "on" segment length > threshold
        if len(on_runs) >= 2 and len(off_runs) >= 1:
            avg_on = np.mean(on_runs)
            if avg_on > 3:  # Minimum dash length
                return True
        
        return False
        
    except Exception:
        return False


def classify_line_thickness(line: LineSeg, img_bw: np.ndarray) -> int:
    """
    Estimate line thickness by sampling perpendicular to line direction.
    
    Args:
        line: Line segment
        img_bw: Binary image
        
    Returns:
        int: Estimated thickness in pixels
    """
    try:
        # Get line center and direction
        cx, cy = line.center
        angle = line.angle
        
        # Sample perpendicular to line direction
        perp_angle = angle + math.pi / 2
        
        # Sample along perpendicular direction
        max_thickness = 20  # Maximum expected thickness
        thicknesses = []
        
        for offset in [-0.25, 0, 0.25]:  # Sample at multiple points along line
            sample_x = cx + offset * (line.p2[0] - line.p1[0])
            sample_y = cy + offset * (line.p2[1] - line.p1[1])
            
            thickness = 0
            for r in range(1, max_thickness):
                x = int(sample_x + r * math.cos(perp_angle))
                y = int(sample_y + r * math.sin(perp_angle))
                
                if (0 <= x < img_bw.shape[1] and 0 <= y < img_bw.shape[0]):
                    if img_bw[y, x] > 0:
                        thickness = r
                    else:
                        break
                else:
                    break
            
            if thickness > 0:
                thicknesses.append(thickness * 2)  # Account for both sides
        
        return int(np.median(thicknesses)) if thicknesses else 2
        
    except Exception:
        return 2  # Default thickness