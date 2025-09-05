"""
Main processing pipeline for PNG2SVG system.
Orchestrates all processing steps from input image to final SVG/GeoJSON output.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

from .config import Config, validate_dependencies
from .io_utils import read_image
from . import preprocess
from . import detect_primitives
from . import detect_symbols
from . import ocr_text
from . import topology
from . import constraints
from . import svg_writer
from . import geojson_writer


def process_image(image_path: str, cfg: Config) -> Dict[str, Any]:
    """
    Process a single image through the complete PNG2SVG pipeline.
    
    Pipeline steps:
    1. Preprocessing: grayscale + noise reduction + adaptive thresholding + optional deskewing
    2. Primitive detection: lines/circles/arcs using LSD/Hough + dash detection
    3. Symbol detection: YOLO small targets (right angles, ticks, arcs, parallel marks, arrows) or rule-based fallback
    4. Text detection: PaddleOCR/Tesseract for labels (A/B/C, angles 30°, lengths x+1 etc)
    5. Topology building: merge co-linear segments, detect intersections/endpoints, bind symbols to geometry, build constraint relationships
    6. Constraint solving: soft constraints for parallel/perpendicular/collinear/equal-length/point-on relationships with optimization
    7. SVG generation: layered output with semantic data-* attributes
    8. GeoJSON export: structured semantic metadata
    
    Args:
        image_path: Path to input image
        cfg: Configuration object
        
    Returns:
        Dictionary with processing results:
        - svg_path: Path to generated SVG
        - geo_path: Path to generated GeoJSON
        - success: Boolean indicating success
        - error: Error message if failed
        - stats: Processing statistics
    """
    logger = logging.getLogger('png2svg.pipeline')
    start_time = time.time()
    
    try:
        logger.info(f"Starting pipeline for: {Path(image_path).name}")
        
        # Validate dependencies
        validate_dependencies(cfg)
        
        # Step 1: Load and preprocess image
        logger.info("Step 1: Loading and preprocessing image")
        img_bgr = read_image(image_path)
        preprocess_result = preprocess.run(img_bgr, cfg)
        
        if not preprocess.validate_preprocessing_result(preprocess_result):
            raise ValueError("Preprocessing failed - invalid output")
        
        logger.info(f"Preprocessing complete. Skew angle: {preprocess_result.skew_angle:.2f}°")
        
        # Step 2: Detect geometric primitives
        logger.info("Step 2: Detecting geometric primitives")
        primitives = detect_primitives.run(preprocess_result, cfg)
        
        logger.info(f"Detected {len(primitives.lines)} lines, {len(primitives.circles)} circles")
        
        # Step 3: Detect symbols (optional YOLO + rule-based fallback)
        logger.info("Step 3: Detecting mathematical symbols")
        symbols = detect_symbols.run(preprocess_result, cfg)
        
        logger.info(f"Detected {len(symbols.items)} symbols")
        
        # Step 4: OCR text detection
        logger.info("Step 4: Detecting text and labels")
        ocr_results = ocr_text.run(img_bgr, preprocess_result, cfg)
        
        logger.info(f"Detected {len(ocr_results)} text items")
        
        # Step 5: Build topology and relationships
        logger.info("Step 5: Building topology and relationships")
        graph = topology.build(primitives, symbols, ocr_results, cfg)
        
        logger.info(f"Built graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges, {len(graph.relations)} relations")
        
        # Step 6: Apply constraint solving (optional)
        if cfg.apply_constraint_solver:
            logger.info("Step 6: Applying constraint optimization")
            graph = constraints.solve(graph, cfg)
            logger.info("Constraint optimization complete")
        else:
            logger.info("Step 6: Skipped constraint solving (disabled)")
        
        # Step 7: Generate SVG output
        logger.info("Step 7: Generating SVG output")
        svg_path = None
        if cfg.export.write_svg:
            svg_path = svg_writer.write(image_path, graph, cfg)
            logger.info(f"SVG generated: {svg_path}")
        
        # Step 8: Generate GeoJSON output
        logger.info("Step 8: Generating GeoJSON output")
        geo_path = None
        if cfg.export.write_geojson:
            geo_path = geojson_writer.write(image_path, graph, cfg)
            logger.info(f"GeoJSON generated: {geo_path}")
        
        # Calculate processing time and statistics
        processing_time = time.time() - start_time
        stats = calculate_processing_stats(graph, processing_time, cfg)
        
        logger.info(f"Pipeline complete in {processing_time:.2f}s")
        
        return {
            'svg_path': svg_path,
            'geo_path': geo_path,
            'success': True,
            'error': None,
            'stats': stats
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Pipeline failed for {Path(image_path).name}: {str(e)}"
        logger.error(error_msg)
        
        return {
            'svg_path': None,
            'geo_path': None,
            'success': False,
            'error': error_msg,
            'stats': {
                'processing_time': processing_time,
                'failed_step': get_current_step_from_error(e)
            }
        }


def calculate_processing_stats(graph: topology.Graph, processing_time: float, cfg: Config) -> Dict[str, Any]:
    """
    Calculate comprehensive processing statistics.
    
    Args:
        graph: Final topology graph
        processing_time: Total processing time
        cfg: Configuration object
        
    Returns:
        Dictionary of processing statistics
    """
    # Count different element types
    line_count = sum(1 for e in graph.edges if isinstance(e.geom, detect_primitives.LineSeg))
    circle_count = sum(1 for e in graph.edges if isinstance(e.geom, detect_primitives.CircleArc) and e.geom.kind == "circle")
    arc_count = sum(1 for e in graph.edges if isinstance(e.geom, detect_primitives.CircleArc) and e.geom.kind == "arc")
    dashed_count = sum(1 for e in graph.edges if isinstance(e.geom, detect_primitives.LineSeg) and e.geom.dashed)
    
    # Count relationship types
    parallel_count = sum(1 for r in graph.relations if r.type == "parallel")
    perp_count = sum(1 for r in graph.relations if r.type == "perpendicular")
    equal_len_count = sum(1 for r in graph.relations if r.type == "equal_length")
    point_on_count = sum(1 for r in graph.relations if r.type.startswith("point_on"))
    
    # Count labeled elements
    labeled_nodes = sum(1 for n in graph.nodes if n.tag)
    
    # Calculate average confidence
    all_confidences = []
    for node in graph.nodes:
        all_confidences.append(node.confidence)
    for edge in graph.edges:
        if hasattr(edge.geom, 'confidence'):
            all_confidences.append(edge.geom.confidence)
    for relation in graph.relations:
        all_confidences.append(relation.conf)
    
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    
    stats = {
        'processing_time': round(processing_time, 3),
        'total_nodes': len(graph.nodes),
        'total_edges': len(graph.edges),
        'total_relations': len(graph.relations),
        'geometry_counts': {
            'lines': line_count,
            'circles': circle_count,
            'arcs': arc_count,
            'dashed_lines': dashed_count
        },
        'relationship_counts': {
            'parallel': parallel_count,
            'perpendicular': perp_count,
            'equal_length': equal_len_count,
            'point_on': point_on_count
        },
        'semantic_counts': {
            'labeled_nodes': labeled_nodes
        },
        'quality_metrics': {
            'average_confidence': round(avg_confidence, 3)
        },
        'processing_config': {
            'deskew_applied': cfg.deskew,
            'constraint_solver_used': cfg.apply_constraint_solver,
            'yolo_symbols_used': cfg.use_yolo_symbols,
            'paddle_ocr_used': cfg.use_paddle_ocr,
            'confidence_tta_used': cfg.confidence_tta
        }
    }
    
    return stats


def get_current_step_from_error(error: Exception) -> str:
    """
    Determine which processing step failed based on error type and message.
    
    Args:
        error: Exception that occurred
        
    Returns:
        Step name where error occurred
    """
    error_msg = str(error).lower()
    
    if any(keyword in error_msg for keyword in ['read', 'load', 'file not found']):
        return "image_loading"
    elif any(keyword in error_msg for keyword in ['preprocess', 'threshold', 'gray']):
        return "preprocessing"
    elif any(keyword in error_msg for keyword in ['line', 'circle', 'hough', 'lsd']):
        return "primitive_detection"
    elif any(keyword in error_msg for keyword in ['symbol', 'yolo', 'detect']):
        return "symbol_detection"
    elif any(keyword in error_msg for keyword in ['ocr', 'text', 'tesseract', 'paddle']):
        return "text_detection"
    elif any(keyword in error_msg for keyword in ['topology', 'graph', 'relation']):
        return "topology_building"
    elif any(keyword in error_msg for keyword in ['constraint', 'optim', 'solve']):
        return "constraint_solving"
    elif any(keyword in error_msg for keyword in ['svg', 'write', 'output']):
        return "output_generation"
    else:
        return "unknown"


def validate_pipeline_result(result: Dict[str, Any]) -> bool:
    """
    Validate that pipeline result is complete and valid.
    
    Args:
        result: Pipeline result dictionary
        
    Returns:
        True if result is valid
    """
    if not isinstance(result, dict):
        return False
    
    required_keys = ['success', 'error', 'stats']
    if not all(key in result for key in required_keys):
        return False
    
    if result['success']:
        # For successful results, check that at least one output was generated
        if not result.get('svg_path') and not result.get('geo_path'):
            return False
    else:
        # For failed results, check that error message is present
        if not result.get('error'):
            return False
    
    # Validate stats structure
    stats = result.get('stats', {})
    if not isinstance(stats, dict) or 'processing_time' not in stats:
        return False
    
    return True


def create_fallback_result(image_path: str, error_msg: str) -> Dict[str, Any]:
    """
    Create a fallback result when pipeline fails completely.
    
    Args:
        image_path: Path to input image
        error_msg: Error message
        
    Returns:
        Fallback result dictionary
    """
    return {
        'svg_path': None,
        'geo_path': None,
        'success': False,
        'error': f"Complete pipeline failure for {Path(image_path).name}: {error_msg}",
        'stats': {
            'processing_time': 0.0,
            'failed_step': 'initialization',
            'total_nodes': 0,
            'total_edges': 0,
            'total_relations': 0
        }
    }


def run_pipeline_with_recovery(image_path: str, cfg: Config) -> Dict[str, Any]:
    """
    Run pipeline with error recovery and fallback mechanisms.
    
    Args:
        image_path: Path to input image
        cfg: Configuration object
        
    Returns:
        Processing result with recovery information
    """
    logger = logging.getLogger('png2svg.pipeline')
    
    try:
        # Attempt full pipeline
        result = process_image(image_path, cfg)
        
        # Validate result
        if not validate_pipeline_result(result):
            logger.warning(f"Invalid pipeline result for {Path(image_path).name}")
            result = create_fallback_result(image_path, "Invalid pipeline result")
        
        return result
        
    except Exception as e:
        logger.error(f"Critical pipeline error for {Path(image_path).name}: {e}")
        return create_fallback_result(image_path, str(e))


def get_processing_summary(results: list) -> Dict[str, Any]:
    """
    Generate summary statistics for batch processing results.
    
    Args:
        results: List of processing results
        
    Returns:
        Summary statistics
    """
    if not results:
        return {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'success_rate': 0.0
        }
    
    # Handle ProcessResult objects (from parallel processing)
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    total_time = sum(r.processing_time for r in results)
    avg_time = total_time / len(results) if results else 0.0
    
    # Collect failure reasons
    failure_reasons = {}
    for r in results:
        if not r.success:
            # For ProcessResult, error messages give us clues about failure type
            if r.error and 'read' in r.error.lower():
                step = 'image_loading'
            elif r.error and 'process' in r.error.lower():
                step = 'processing'
            else:
                step = 'unknown'
            failure_reasons[step] = failure_reasons.get(step, 0) + 1
    
    # For ProcessResult objects, we don't have detailed geometry stats
    # This is a limitation of the current structure
    total_lines = 0
    total_circles = 0 
    total_relations = 0
    
    summary = {
        'total_processed': len(results),
        'successful': successful,
        'failed': failed,
        'success_rate': round(successful / len(results) * 100, 1),
        'timing': {
            'total_time': round(total_time, 2),
            'average_time': round(avg_time, 2)
        },
        'failure_analysis': failure_reasons,
        'geometry_totals': {
            'lines': total_lines,
            'circles': total_circles,
            'relations': total_relations
        }
    }
    
    return summary