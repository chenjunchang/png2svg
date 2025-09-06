"""
Main processing pipeline for PNG2SVG.

Orchestrates the complete workflow from PNG input to SVG+GeoJSON output,
coordinating preprocessing, detection, analysis, and output generation.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np

from .config import Config
from .io_utils import read_image, get_output_paths, ensure_output_dir
from .preprocess import run as preprocess_run
from .detect_primitives import run as detect_primitives_run
from .detect_symbols import run as detect_symbols_run
from .ocr_text import run as ocr_run
from .topology import build as topology_build
from .constraints import solve as constraints_solve
from .svg_writer import write as svg_write
from .geojson_writer import write as geojson_write


@dataclass
class ProcessingResult:
    """Result of processing a single PNG image."""
    success: bool
    input_path: str
    svg_path: Optional[str] = None
    geo_path: Optional[str] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    stats: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.stats is None:
            self.stats = {}


def process_image(image_path: str, config: Config) -> ProcessingResult:
    """
    Process a single PNG image through the complete PNG2SVG pipeline.
    
    This is the main entry point for processing individual images. It coordinates
    all processing steps from input to output generation.
    
    Args:
        image_path: Path to PNG image file
        config: Configuration object
        
    Returns:
        ProcessingResult: Processing outcome with paths and statistics
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    try:
        logger.debug(f"Starting processing: {Path(image_path).name}")
        
        # Ensure output directory exists
        ensure_output_dir(config.output_dir)
        
        # Get output file paths
        svg_path, geo_path = get_output_paths(image_path, config.output_dir)
        
        # Step 1: Load image
        logger.debug("Loading image...")
        img_bgr = read_image(image_path)
        h, w = img_bgr.shape[:2]
        logger.debug(f"Image size: {w}x{h}")
        
        # Step 2: Preprocessing
        logger.debug("Preprocessing image...")
        preprocess_result = preprocess_run(img_bgr, config)
        
        # Step 3: Primitive detection (lines, circles, arcs)
        logger.debug("Detecting geometric primitives...")
        primitives_result = detect_primitives_run(preprocess_result, config)
        
        # Step 4: Symbol detection (optional YOLO or rule-based)
        logger.debug("Detecting symbols...")
        symbols_result = detect_symbols_run(preprocess_result, config)
        
        # Step 5: OCR text recognition
        logger.debug("Running OCR...")
        ocr_result = ocr_run(img_bgr, preprocess_result, config)
        
        # Step 6: Build topological relationships
        logger.debug("Building topology...")
        topology_graph = topology_build(
            primitives_result, 
            symbols_result, 
            ocr_result, 
            config
        )
        
        # Step 7: Optional constraint solving
        if config.apply_constraint_solver:
            logger.debug("Applying constraint solver...")
            topology_graph = constraints_solve(topology_graph, config)
        
        # Step 8: Generate outputs
        logger.debug("Writing SVG output...")
        actual_svg_path = svg_write(image_path, topology_graph, config)
        
        logger.debug("Writing GeoJSON output...")
        actual_geo_path = geojson_write(image_path, topology_graph, config)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Collect statistics
        stats = {
            'image_size': (w, h),
            'processing_time': processing_time,
            'skew_angle': preprocess_result.skew_angle,
            'primitives': primitives_result.stats,
            'symbols': symbols_result.stats,
            'ocr': ocr_result.stats,
            'topology': {
                'nodes': len(topology_graph.nodes),
                'edges': len(topology_graph.edges),
                'relations': len(topology_graph.relations)
            }
        }
        
        logger.info(f"Successfully processed {Path(image_path).name} in {processing_time:.2f}s")
        
        return ProcessingResult(
            success=True,
            input_path=image_path,
            svg_path=actual_svg_path,
            geo_path=actual_geo_path,
            processing_time=processing_time,
            stats=stats
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Processing failed: {type(e).__name__}: {str(e)}"
        
        logger.error(f"Error processing {Path(image_path).name}: {error_msg}")
        logger.debug(f"Processing failed after {processing_time:.2f}s")
        
        return ProcessingResult(
            success=False,
            input_path=image_path,
            processing_time=processing_time,
            error_message=error_msg
        )


def validate_image(image_path: str) -> bool:
    """
    Validate that an image file can be processed.
    
    Args:
        image_path: Path to image file
        
    Returns:
        bool: True if image is valid for processing
    """
    try:
        # Check file exists
        if not Path(image_path).exists():
            return False
        
        # Try to load image with Unicode path support
        img = read_image(image_path)
        
        # Check minimum size
        h, w = img.shape[:2]
        if w < 50 or h < 50:
            return False
        
        # Check maximum size (to prevent memory issues)
        if w > 5000 or h > 5000:
            return False
        
        return True
        
    except Exception:
        return False


def estimate_processing_time(image_path: str, config: Config) -> float:
    """
    Estimate processing time for an image based on size and configuration.
    
    Args:
        image_path: Path to image file
        config: Configuration object
        
    Returns:
        float: Estimated processing time in seconds
    """
    try:
        img = read_image(image_path)
        
        h, w = img.shape[:2]
        pixels = w * h
        
        # Base time estimate based on image size
        base_time = pixels / 1000000  # 1 second per megapixel
        
        # Add time for optional features
        if config.use_yolo_symbols:
            base_time += 2.0  # YOLO inference time
        
        if config.use_paddle_ocr:
            base_time += 1.0  # PaddleOCR time
        else:
            base_time += 0.5  # Tesseract time
        
        if config.apply_constraint_solver:
            base_time += 0.5  # Constraint solving time
        
        if config.confidence_tta:
            base_time *= 1.5  # TTA multiplier
        
        return max(1.0, base_time)  # Minimum 1 second
        
    except Exception:
        return 5.0  # Default estimate


def get_pipeline_info() -> Dict[str, Any]:
    """
    Get information about the processing pipeline capabilities.
    
    Returns:
        Dict[str, Any]: Pipeline information and capabilities
    """
    info = {
        'version': '0.1.0',
        'capabilities': {
            'geometric_primitives': ['lines', 'circles', 'arcs', 'dashed_lines'],
            'symbols': ['right_angles', 'parallel_marks', 'arrows', 'tick_marks', 'angle_arcs'],
            'text_recognition': ['labels', 'numbers', 'angles', 'variables'],
            'output_formats': ['svg', 'geojson']
        },
        'optional_features': {
            'yolo_symbols': _check_yolo_availability(),
            'paddle_ocr': _check_paddleocr_availability(),
            'constraint_solver': _check_scipy_availability(),
            'advanced_preprocessing': _check_skimage_availability()
        },
        'processing_steps': [
            'preprocessing',
            'primitive_detection', 
            'symbol_detection',
            'text_recognition',
            'topology_building',
            'constraint_solving',
            'output_generation'
        ]
    }
    
    return info


def _check_yolo_availability() -> bool:
    """Check if YOLO dependencies are available."""
    try:
        import ultralytics
        return True
    except ImportError:
        try:
            import onnxruntime
            return True
        except ImportError:
            return False


def _check_paddleocr_availability() -> bool:
    """Check if PaddleOCR is available."""
    try:
        import paddleocr
        return True
    except ImportError:
        return False


def _check_scipy_availability() -> bool:
    """Check if SciPy (for constraint solving) is available."""
    try:
        import scipy.optimize
        return True
    except ImportError:
        return False


def _check_skimage_availability() -> bool:
    """Check if scikit-image is available."""
    try:
        import skimage
        return True
    except ImportError:
        return False


def create_processing_summary(results: list) -> Dict[str, Any]:
    """
    Create a summary of batch processing results.
    
    Args:
        results: List of ProcessingResult objects
        
    Returns:
        Dict[str, Any]: Processing summary statistics
    """
    if not results:
        return {'total': 0}
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    summary = {
        'total': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'success_rate': len(successful) / len(results) if results else 0,
        'total_time': sum(r.processing_time for r in results),
        'avg_time': sum(r.processing_time for r in results) / len(results),
    }
    
    if successful:
        # Aggregate statistics from successful processings
        all_stats = [r.stats for r in successful if r.stats]
        
        if all_stats:
            summary['avg_lines'] = np.mean([
                s.get('primitives', {}).get('total_lines', 0) 
                for s in all_stats
            ])
            summary['avg_circles'] = np.mean([
                s.get('primitives', {}).get('total_circles', 0) 
                for s in all_stats
            ])
            summary['avg_text_regions'] = np.mean([
                s.get('ocr', {}).get('filtered_detections', 0) 
                for s in all_stats
            ])
    
    if failed:
        # Common error types
        error_types = {}
        for r in failed:
            if r.error_message:
                error_type = r.error_message.split(':')[0]
                error_types[error_type] = error_types.get(error_type, 0) + 1
        summary['error_types'] = error_types
    
    return summary