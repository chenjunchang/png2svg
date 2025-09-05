"""
Configuration management for PNG2SVG system.
Handles YAML configuration loading and validation using Pydantic.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, validator, Field


@dataclass
class AlgorithmLSDConfig:
    """LSD (Line Segment Detector) algorithm configuration."""
    refine: bool = True
    scale: float = 0.8
    sigma: float = 0.6
    quant: float = 2.0
    ang_th: float = 22.5
    log_eps: float = 0.0
    density_th: float = 0.7
    n_bins: int = 1024


@dataclass
class AlgorithmHoughLinesConfig:
    """Hough Lines algorithm configuration."""
    threshold: int = 50
    min_line_length: int = 25
    max_line_gap: int = 4


@dataclass
class AlgorithmHoughCirclesConfig:
    """Hough Circles algorithm configuration."""
    dp: float = 1.2
    min_dist: int = 20
    param1: int = 120
    param2: int = 30
    min_radius: int = 8
    max_radius: int = 0


@dataclass
class AlgorithmPreprocessConfig:
    """Preprocessing algorithm configuration."""
    gaussian_blur_ksize: int = 3
    adaptive_thresh_blocksize: int = 35
    adaptive_thresh_c: int = 15


@dataclass
class AlgorithmConstraintsConfig:
    """Constraint solver configuration."""
    max_iterations: int = 100
    tolerance: float = 1e-6
    huber_delta: float = 1.0
    lambda_parallel: float = 1.0
    lambda_perpendicular: float = 1.0
    lambda_collinear: float = 0.5
    lambda_equal_length: float = 0.8
    lambda_point_on_circle: float = 1.0


@dataclass
class AlgorithmsConfig:
    """All algorithm configurations."""
    lsd: AlgorithmLSDConfig = field(default_factory=AlgorithmLSDConfig)
    hough_lines: AlgorithmHoughLinesConfig = field(default_factory=AlgorithmHoughLinesConfig)
    hough_circles: AlgorithmHoughCirclesConfig = field(default_factory=AlgorithmHoughCirclesConfig)
    preprocess: AlgorithmPreprocessConfig = field(default_factory=AlgorithmPreprocessConfig)
    constraints: AlgorithmConstraintsConfig = field(default_factory=AlgorithmConstraintsConfig)


@dataclass
class SVGConfig:
    """SVG output configuration."""
    scale: float = 1.0
    stroke_main: int = 2
    stroke_aux: int = 1
    dash_pattern: str = "6,6"


@dataclass
class ExportConfig:
    """Export settings configuration."""
    write_geojson: bool = True
    write_svg: bool = True


@dataclass
class Config:
    """Main configuration class for PNG2SVG system."""
    
    # Basic I/O
    input_dir: str = "./pngs"
    output_dir: str = "./out"
    
    # Processing
    jobs: int = 4
    deskew: bool = True
    min_line_len: int = 25
    line_merge_angle_deg: int = 3
    line_merge_gap_px: int = 6
    
    # Model settings
    use_yolo_symbols: bool = True
    yolo_symbols_weights: str = "models/symbols_yolo.onnx"
    use_paddle_ocr: bool = True
    
    # Advanced processing
    confidence_tta: bool = True
    apply_constraint_solver: bool = True
    
    # Output configuration
    svg: SVGConfig = field(default_factory=SVGConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # Logging
    log_level: str = "INFO"
    
    # Algorithm parameters
    algorithms: AlgorithmsConfig = field(default_factory=AlgorithmsConfig)
    
    def __post_init__(self):
        """Post-initialization validation and path resolution."""
        # Convert relative paths to absolute
        self.input_dir = str(Path(self.input_dir).resolve())
        self.output_dir = str(Path(self.output_dir).resolve())
        
        # Check if YOLO weights exist, disable if not found
        if self.use_yolo_symbols:
            weights_path = Path(self.yolo_symbols_weights)
            if not weights_path.exists():
                self.use_yolo_symbols = False
                print(f"[WARNING] YOLO weights not found at {weights_path}, disabling YOLO symbols detection")
        
        # Validate log level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR"}
        if self.log_level.upper() not in valid_levels:
            self.log_level = "INFO"
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[Union[str, Path]] = None, 
                cli_overrides: Optional[Dict[str, Any]] = None) -> Config:
    """
    Load configuration from YAML file with optional CLI overrides.
    
    Args:
        config_path: Path to YAML configuration file
        cli_overrides: Dictionary of CLI parameter overrides
        
    Returns:
        Config: Loaded and validated configuration
    """
    # Default configuration
    config_dict = {}
    
    # Load from YAML file if provided
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f) or {}
                print(f"[INFO] Loaded configuration from {config_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load config from {config_path}: {e}")
                raise
        else:
            print(f"[WARNING] Config file not found: {config_path}")
    
    # Apply CLI overrides
    if cli_overrides:
        config_dict.update(cli_overrides)
        print(f"[INFO] Applied CLI overrides: {list(cli_overrides.keys())}")
    
    # Create config object from dictionary
    try:
        config = _dict_to_config(config_dict)
        print(f"[INFO] Configuration loaded successfully")
        print(f"[INFO] Input: {config.input_dir}")
        print(f"[INFO] Output: {config.output_dir}")
        print(f"[INFO] Jobs: {config.jobs}")
        print(f"[INFO] YOLO symbols: {config.use_yolo_symbols}")
        print(f"[INFO] PaddleOCR: {config.use_paddle_ocr}")
        print(f"[INFO] Constraint solver: {config.apply_constraint_solver}")
        return config
    except Exception as e:
        print(f"[ERROR] Configuration validation failed: {e}")
        raise


def _dict_to_config(config_dict: Dict[str, Any]) -> Config:
    """Convert nested dictionary to Config dataclass."""
    
    # Handle nested SVG config
    svg_dict = config_dict.pop('svg', {})
    svg_config = SVGConfig(
        scale=svg_dict.get('scale', 1.0),
        stroke_main=svg_dict.get('stroke_main', 2),
        stroke_aux=svg_dict.get('stroke_aux', 1),
        dash_pattern=svg_dict.get('dash_pattern', "6,6")
    )
    
    # Handle nested export config
    export_dict = config_dict.pop('export', {})
    export_config = ExportConfig(
        write_geojson=export_dict.get('write_geojson', True),
        write_svg=export_dict.get('write_svg', True)
    )
    
    # Handle nested algorithms config
    algorithms_dict = config_dict.pop('algorithms', {})
    
    lsd_dict = algorithms_dict.get('lsd', {})
    lsd_config = AlgorithmLSDConfig(
        refine=lsd_dict.get('refine', True),
        scale=lsd_dict.get('scale', 0.8),
        sigma=lsd_dict.get('sigma', 0.6),
        quant=lsd_dict.get('quant', 2.0),
        ang_th=lsd_dict.get('ang_th', 22.5),
        log_eps=lsd_dict.get('log_eps', 0.0),
        density_th=lsd_dict.get('density_th', 0.7),
        n_bins=lsd_dict.get('n_bins', 1024)
    )
    
    hough_lines_dict = algorithms_dict.get('hough_lines', {})
    hough_lines_config = AlgorithmHoughLinesConfig(
        threshold=hough_lines_dict.get('threshold', 50),
        min_line_length=hough_lines_dict.get('min_line_length', 25),
        max_line_gap=hough_lines_dict.get('max_line_gap', 4)
    )
    
    hough_circles_dict = algorithms_dict.get('hough_circles', {})
    hough_circles_config = AlgorithmHoughCirclesConfig(
        dp=hough_circles_dict.get('dp', 1.2),
        min_dist=hough_circles_dict.get('min_dist', 20),
        param1=hough_circles_dict.get('param1', 120),
        param2=hough_circles_dict.get('param2', 30),
        min_radius=hough_circles_dict.get('min_radius', 8),
        max_radius=hough_circles_dict.get('max_radius', 0)
    )
    
    preprocess_dict = algorithms_dict.get('preprocess', {})
    preprocess_config = AlgorithmPreprocessConfig(
        gaussian_blur_ksize=preprocess_dict.get('gaussian_blur_ksize', 3),
        adaptive_thresh_blocksize=preprocess_dict.get('adaptive_thresh_blocksize', 35),
        adaptive_thresh_c=preprocess_dict.get('adaptive_thresh_c', 15)
    )
    
    constraints_dict = algorithms_dict.get('constraints', {})
    constraints_config = AlgorithmConstraintsConfig(
        max_iterations=constraints_dict.get('max_iterations', 100),
        tolerance=constraints_dict.get('tolerance', 1e-6),
        huber_delta=constraints_dict.get('huber_delta', 1.0),
        lambda_parallel=constraints_dict.get('lambda_parallel', 1.0),
        lambda_perpendicular=constraints_dict.get('lambda_perpendicular', 1.0),
        lambda_collinear=constraints_dict.get('lambda_collinear', 0.5),
        lambda_equal_length=constraints_dict.get('lambda_equal_length', 0.8),
        lambda_point_on_circle=constraints_dict.get('lambda_point_on_circle', 1.0)
    )
    
    algorithms_config = AlgorithmsConfig(
        lsd=lsd_config,
        hough_lines=hough_lines_config,
        hough_circles=hough_circles_config,
        preprocess=preprocess_config,
        constraints=constraints_config
    )
    
    # Create main config
    config = Config(
        svg=svg_config,
        export=export_config,
        algorithms=algorithms_config,
        **config_dict
    )
    
    return config


def validate_dependencies(config: Config) -> None:
    """
    Validate that required dependencies are available based on configuration.
    Issues warnings and updates config if optional dependencies are missing.
    """
    # Check PaddleOCR availability
    if config.use_paddle_ocr:
        try:
            import paddleocr
            print("[INFO] PaddleOCR is available")
        except ImportError:
            print("[WARNING] PaddleOCR not available, will use pytesseract as fallback")
            config.use_paddle_ocr = False
    
    # Check YOLO/ONNX availability
    if config.use_yolo_symbols:
        try:
            import onnxruntime
            print("[INFO] ONNX Runtime is available for YOLO inference")
        except ImportError:
            try:
                import ultralytics
                print("[INFO] Ultralytics YOLO is available")
            except ImportError:
                print("[WARNING] Neither ONNX Runtime nor Ultralytics available, disabling YOLO symbols")
                config.use_yolo_symbols = False
    
    # Check constraint solver dependencies
    if config.apply_constraint_solver:
        try:
            import scipy
            print("[INFO] SciPy is available for constraint solving")
        except ImportError:
            print("[WARNING] SciPy not available, disabling constraint solver")
            config.apply_constraint_solver = False