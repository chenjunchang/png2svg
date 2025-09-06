"""
Configuration management for PNG2SVG.

Provides Pydantic models for configuration validation and loading from YAML files.
Supports command-line argument overrides and graceful handling of optional dependencies.
"""

import os
import argparse
from pathlib import Path
from typing import Optional, List, Union
from dataclasses import dataclass

import yaml
from pydantic import BaseModel, Field, validator, ConfigDict


class SVGConfig(BaseModel):
    """SVG output configuration."""
    scale: float = Field(default=1.0, description="Scale factor for SVG output")
    stroke_main: int = Field(default=2, description="Main stroke width")
    stroke_aux: int = Field(default=1, description="Auxiliary stroke width")
    dash_pattern: str = Field(default="6,6", description="Dash pattern for dashed lines")


class ExportConfig(BaseModel):
    """Export configuration."""
    write_geojson: bool = Field(default=True, description="Whether to write GeoJSON output")
    write_svg: bool = Field(default=True, description="Whether to write SVG output")


class Config(BaseModel):
    """Main configuration for PNG2SVG processing."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Input/Output
    input_dir: str = Field(default="./inputs", description="Input directory containing PNG files")
    output_dir: str = Field(default="./out", description="Output directory for results")
    
    # Processing
    jobs: int = Field(default=4, description="Number of parallel processing jobs")
    deskew: bool = Field(default=True, description="Enable automatic deskewing")
    
    # Line detection parameters
    min_line_len: int = Field(default=25, description="Minimum line length for detection")
    line_merge_angle_deg: float = Field(default=3.0, description="Angle threshold for merging collinear lines")
    line_merge_gap_px: int = Field(default=6, description="Gap threshold for merging lines")
    
    # Model configuration
    use_yolo_symbols: bool = Field(default=True, description="Use YOLO for symbol detection")
    yolo_symbols_weights: str = Field(default="models/symbols_yolo.onnx", description="Path to YOLO weights")
    use_paddle_ocr: bool = Field(default=True, description="Use PaddleOCR for text recognition")
    
    # Processing options
    confidence_tta: bool = Field(default=True, description="Use Test Time Augmentation for confidence")
    apply_constraint_solver: bool = Field(default=True, description="Apply geometric constraint solving")
    
    # Sub-configurations
    svg: SVGConfig = Field(default_factory=SVGConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()
    
    @validator('jobs')
    def validate_jobs(cls, v):
        if v < 1:
            raise ValueError("jobs must be at least 1")
        return v
    
    @validator('input_dir', 'output_dir')
    def validate_directories(cls, v):
        return os.path.expanduser(v)


def load_config(args: Optional[argparse.Namespace] = None) -> Config:
    """
    Load configuration from YAML file and apply command-line overrides.
    
    Args:
        args: Command-line arguments from argparse
        
    Returns:
        Config: Validated configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_data = {}
    
    # Load from config file if specified
    if args and hasattr(args, 'config') and args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
    
    # Apply command-line overrides
    if args:
        overrides = {}
        if hasattr(args, 'input') and args.input:
            overrides['input_dir'] = args.input
        if hasattr(args, 'output') and args.output:
            overrides['output_dir'] = args.output
        if hasattr(args, 'jobs') and args.jobs:
            overrides['jobs'] = args.jobs
        if hasattr(args, 'no_yolo') and args.no_yolo:
            overrides['use_yolo_symbols'] = False
        if hasattr(args, 'no_paddleocr') and args.no_paddleocr:
            overrides['use_paddle_ocr'] = False
        if hasattr(args, 'no_constraints') and args.no_constraints:
            overrides['apply_constraint_solver'] = False
        
        # Merge overrides into config_data
        config_data.update(overrides)
    
    # Create and validate config
    config = Config(**config_data)
    
    # Post-validation checks for file dependencies
    _check_optional_dependencies(config)
    
    return config


def _check_optional_dependencies(config: Config) -> None:
    """
    Check for optional dependencies and automatically disable features if not available.
    
    Args:
        config: Configuration object to modify
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Check YOLO weights file
    if config.use_yolo_symbols:
        weights_path = Path(config.yolo_symbols_weights)
        if not weights_path.exists():
            logger.warning(f"YOLO weights file not found: {weights_path}. Disabling YOLO symbol detection.")
            config.use_yolo_symbols = False
    
    # Check PaddleOCR availability
    if config.use_paddle_ocr:
        try:
            import paddleocr
        except ImportError:
            logger.warning("PaddleOCR not available. Falling back to Tesseract.")
            config.use_paddle_ocr = False
    
    # Check constraint solver dependencies
    if config.apply_constraint_solver:
        try:
            import scipy.optimize
        except ImportError:
            logger.warning("SciPy not available. Disabling constraint solver.")
            config.apply_constraint_solver = False


def create_default_config_file(output_path: str = "config.example.yaml") -> None:
    """
    Create a default configuration file with all available options.
    
    Args:
        output_path: Path where to write the config file
    """
    default_config = {
        'input_dir': './inputs',
        'output_dir': './out',
        'jobs': 4,
        'deskew': True,
        'min_line_len': 25,
        'line_merge_angle_deg': 3.0,
        'line_merge_gap_px': 6,
        'use_yolo_symbols': True,
        'yolo_symbols_weights': 'models/symbols_yolo.onnx',
        'use_paddle_ocr': True,
        'confidence_tta': True,
        'apply_constraint_solver': True,
        'svg': {
            'scale': 1.0,
            'stroke_main': 2,
            'stroke_aux': 1,
            'dash_pattern': '6,6'
        },
        'export': {
            'write_geojson': True,
            'write_svg': True
        },
        'log_level': 'INFO'
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True, indent=2)