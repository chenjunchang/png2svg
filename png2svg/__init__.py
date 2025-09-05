"""
PNG2SVG - Mathematical Diagram Vectorization System

A comprehensive system for converting PNG mathematical diagrams to semantic SVG format.
Combines traditional computer vision techniques with modern ML approaches for robust
geometric analysis and intelligent vectorization.

Main Features:
- Geometric primitive detection (lines, circles, arcs)
- Mathematical symbol recognition (angles, tick marks, arrows)
- OCR for text labels and measurements
- Topological relationship inference
- Constraint-based geometric optimization
- Layered SVG output with semantic annotations
- Structured GeoJSON metadata export

Example Usage:
    ```python
    from png2svg import Config, process_image
    from png2svg.config import load_config
    
    # Load configuration
    cfg = load_config('config.yaml')
    
    # Process single image
    result = process_image('diagram.png', cfg)
    
    if result['success']:
        print(f"SVG: {result['svg_path']}")
        print(f"GeoJSON: {result['geo_path']}")
    ```

CLI Usage:
    ```bash
    python -m png2svg.cli --input ./pngs --output ./out --config config.yaml
    ```
"""

import sys
import logging
from pathlib import Path

# Version information
__version__ = "1.0.0"
__author__ = "PNG2SVG Team"
__email__ = "png2svg@example.com"
__description__ = "Mathematical Diagram Vectorization System"
__url__ = "https://github.com/example/png2svg"

# Set up package-level logger
logger = logging.getLogger(__name__)

# Compatibility check
if sys.version_info < (3, 8):
    raise RuntimeError("PNG2SVG requires Python 3.8 or higher")

# Import main components for easy access
try:
    from .config import Config, load_config, validate_dependencies
    from .pipeline import process_image, run_pipeline_with_recovery, get_processing_summary
    from .io_utils import setup_logging, list_png_files
    
    # Import key data structures
    from .topology import Graph, Node, Edge, Relation
    from .detect_primitives import LineSeg, CircleArc, Primitives
    from .detect_symbols import Symbol, Symbols
    from .ocr_text import OCRItem
    
    # Mark successful imports
    _imports_successful = True
    
except ImportError as e:
    logger.warning(f"Some PNG2SVG components could not be imported: {e}")
    _imports_successful = False

# Public API - only export main components
__all__ = [
    # Version info
    '__version__',
    '__author__', 
    '__description__',
    
    # Main API
    'Config',
    'load_config',
    'validate_dependencies',
    'process_image',
    'run_pipeline_with_recovery',
    'get_processing_summary',
    
    # Utilities
    'setup_logging',
    'list_png_files',
    
    # Data structures
    'Graph',
    'Node', 
    'Edge',
    'Relation',
    'LineSeg',
    'CircleArc',
    'Primitives',
    'Symbol',
    'Symbols',
    'OCRItem',
    
    # Helper functions
    'check_dependencies',
    'get_version_info'
]


def check_dependencies(verbose: bool = False) -> dict:
    """
    Check availability of optional dependencies.
    
    Args:
        verbose: Print detailed information about dependencies
        
    Returns:
        Dictionary with dependency status
    """
    dependencies = {
        'core': {
            'opencv-python': False,
            'numpy': False,
            'scipy': False,
            'shapely': False,
            'networkx': False,
            'svgwrite': False,
            'pydantic': False,
            'PyYAML': False,
            'tqdm': False
        },
        'ocr': {
            'pytesseract': False,
            'paddleocr': False
        },
        'ml': {
            'ultralytics': False,
            'onnxruntime': False
        },
        'advanced': {
            'scikit-image': False,
            'scikit-learn': False
        }
    }
    
    # Check core dependencies
    for dep in dependencies['core']:
        try:
            __import__(dep.replace('-', '_'))
            dependencies['core'][dep] = True
        except ImportError:
            pass
    
    # Check OCR dependencies
    for dep in dependencies['ocr']:
        try:
            __import__(dep)
            dependencies['ocr'][dep] = True
        except ImportError:
            pass
    
    # Check ML dependencies
    for dep in dependencies['ml']:
        try:
            __import__(dep)
            dependencies['ml'][dep] = True
        except ImportError:
            pass
    
    # Check advanced dependencies
    for dep in dependencies['advanced']:
        try:
            __import__(dep.replace('-', '_'))
            dependencies['advanced'][dep] = True
        except ImportError:
            pass
    
    if verbose:
        print("PNG2SVG dependency status:")
        print("=" * 40)
        
        for category, deps in dependencies.items():
            print(f"\n{category.upper()} dependencies:")
            for dep, available in deps.items():
                status = "[+]" if available else "[X]"
                print(f"  {status} {dep}")
        
        # Check minimum requirements
        core_available = sum(dependencies['core'].values())
        total_core = len(dependencies['core'])
        
        print(f"\nCore dependencies: {core_available}/{total_core}")
        
        if core_available < total_core:
            print("[!] Some core dependencies are missing. Please install with:")
            print("   pip install -r requirements.txt")
        else:
            print("[OK] All core dependencies are available")
        
        # Optional features
        ocr_available = any(dependencies['ocr'].values())
        ml_available = any(dependencies['ml'].values())
        
        print(f"\nOptional features:")
        print(f"  OCR support: {'[+]' if ocr_available else '[X]'}")
        print(f"  ML symbols: {'[+]' if ml_available else '[X]'}")
    
    return dependencies


def get_version_info() -> dict:
    """
    Get comprehensive version and system information.
    
    Returns:
        Dictionary with version information
    """
    import platform
    
    info = {
        'png2svg_version': __version__,
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.architecture(),
        'dependencies': check_dependencies(verbose=False)
    }
    
    # Add key dependency versions if available
    dep_versions = {}
    
    version_checks = [
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('shapely', 'shapely'),
        ('svgwrite', 'svgwrite'),
    ]
    
    for pkg_name, import_name in version_checks:
        try:
            module = __import__(import_name)
            if hasattr(module, '__version__'):
                dep_versions[pkg_name] = module.__version__
        except ImportError:
            dep_versions[pkg_name] = "Not available"
    
    info['dependency_versions'] = dep_versions
    
    return info


def print_version_info():
    """Print formatted version information."""
    info = get_version_info()
    
    print(f"PNG2SVG v{info['png2svg_version']}")
    print(f"Python {info['python_version']}")
    print(f"Platform: {info['platform']}")
    print()
    
    print("Key dependencies:")
    for pkg, version in info['dependency_versions'].items():
        print(f"  {pkg}: {version}")


# Initialize package
def _initialize_package():
    """Initialize package-level settings and checks."""
    # Set up basic logging for the package
    logging.getLogger(__name__).addHandler(logging.NullHandler())
    
    # Check if we're in a development environment
    pkg_path = Path(__file__).parent
    if (pkg_path.parent / '.git').exists():
        logger.info("PNG2SVG running in development mode")
    
    # Warn about missing dependencies
    if not _imports_successful:
        logger.warning(
            "PNG2SVG initialization incomplete due to missing dependencies. "
            "Run check_dependencies(verbose=True) for details."
        )


# Run initialization
_initialize_package()


# Convenience functions for backwards compatibility and easy access
def convert_image(image_path: str, config_path: str = None, **kwargs) -> dict:
    """
    Convenience function to convert a single image.
    
    Args:
        image_path: Path to input image
        config_path: Path to configuration file
        **kwargs: Additional configuration overrides
        
    Returns:
        Processing result dictionary
    """
    if not _imports_successful:
        raise RuntimeError("PNG2SVG not properly initialized due to missing dependencies")
    
    # Load configuration
    cfg = load_config(config_path, kwargs)
    
    # Process image
    return process_image(image_path, cfg)


def batch_convert(input_dir: str, output_dir: str = None, config_path: str = None, **kwargs) -> list:
    """
    Convenience function to batch convert images.
    
    Args:
        input_dir: Input directory containing PNG files
        output_dir: Output directory for results
        config_path: Path to configuration file
        **kwargs: Additional configuration overrides
        
    Returns:
        List of processing results
    """
    if not _imports_successful:
        raise RuntimeError("PNG2SVG not properly initialized due to missing dependencies")
    
    from .io_utils import run_sequential_processing
    
    # Build config overrides
    config_overrides = dict(kwargs)
    config_overrides['input_dir'] = input_dir
    if output_dir:
        config_overrides['output_dir'] = output_dir
    
    # Load configuration
    cfg = load_config(config_path, config_overrides)
    
    # Find files and process
    image_files = list_png_files(cfg.input_dir)
    
    return run_sequential_processing(
        image_files,
        run_pipeline_with_recovery,
        cfg,
        progress_bar=True
    )