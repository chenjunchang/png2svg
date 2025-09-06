"""
PNG2SVG: Convert PNG mathematical diagrams to structured SVG with semantic metadata.

This package provides tools for converting PNG images of mathematical diagrams
into structured SVG files with semantic annotations, including:
- Geometric primitives (lines, circles, arcs)
- Mathematical symbols (arrows, right angles, tick marks)
- Text labels and annotations
- Topological relationships and constraints
"""

__version__ = "0.1.0"
__author__ = "PNG2SVG Team"
__email__ = "support@png2svg.org"

from .pipeline import process_image
from .config import Config

__all__ = ["process_image", "Config", "__version__"]