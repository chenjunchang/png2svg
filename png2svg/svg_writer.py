"""
SVG output generation for PNG2SVG.

Creates layered SVG files with semantic metadata, including proper styling,
markers, and coordinate transformation from pixel to SVG space.
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import svgwrite
from svgwrite.container import Group
from svgwrite.shapes import Line, Circle, Ellipse
from svgwrite.text import Text
from svgwrite.path import Path as SvgPath

from .config import Config
from .topology import TopologyGraph, Node, Edge
from .detect_primitives import LineSeg, CircleArc


def write(image_path: str, graph: TopologyGraph, config: Config) -> str:
    """
    Generate SVG file from topology graph.
    
    Args:
        image_path: Original image path (for output naming)
        graph: Topology graph with geometric elements
        config: Configuration object
        
    Returns:
        str: Path to generated SVG file
    """
    logger = logging.getLogger(__name__)
    
    # Determine output path
    base_name = Path(image_path).stem
    svg_path = Path(config.output_dir) / f"{base_name}.svg"
    
    # Calculate SVG dimensions and viewBox
    svg_size, viewbox = _calculate_svg_dimensions(graph, config)
    
    # Create SVG document
    dwg = svgwrite.Drawing(
        str(svg_path),
        size=svg_size,
        viewBox=f"0 0 {viewbox[0]} {viewbox[1]}"
    )
    
    # Add metadata using proper svgwrite API
    # Note: svgwrite.Drawing doesn't have title/desc methods, we'll add them as elements later if needed
    
    # Add styles and definitions
    _add_svg_definitions(dwg, config)
    
    # Create layered groups
    groups = _create_svg_groups(dwg)
    
    # Add geometric elements
    _add_geometric_elements(dwg, groups, graph, config)
    
    # Add symbols and markers
    _add_symbol_elements(dwg, groups, graph, config)
    
    # Add text elements
    _add_text_elements(dwg, groups, graph, config)
    
    # Add groups to document
    for group in groups.values():
        dwg.add(group)
    
    # Save SVG
    dwg.save()
    
    logger.debug(f"SVG saved to: {svg_path}")
    return str(svg_path)


def _calculate_svg_dimensions(graph: TopologyGraph, config: Config) -> Tuple[Tuple[str, str], Tuple[float, float]]:
    """Calculate SVG size and viewBox from graph bounds."""
    if not graph.nodes:
        return ("400px", "300px"), (400, 300)
    
    # Find bounds of all nodes
    x_coords = [node.x for node in graph.nodes]
    y_coords = [node.y for node in graph.nodes]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # Add padding
    padding = 20
    width = max_x - min_x + 2 * padding
    height = max_y - min_y + 2 * padding
    
    # Apply scaling
    width *= config.svg.scale
    height *= config.svg.scale
    
    # Create size strings
    size = (f"{width:.0f}px", f"{height:.0f}px")
    viewbox = (width, height)
    
    return size, viewbox


def _add_svg_definitions(dwg: svgwrite.Drawing, config: Config) -> None:
    """Add SVG definitions for markers, patterns, and styles."""
    defs = dwg.defs
    
    # Arrow marker
    arrow_marker = defs.add(
        dwg.marker(
            id='arrow',
            insert=(4, 2),
            size=(4, 4),
            orient='auto',
            markerUnits='userSpaceOnUse'
        )
    )
    arrow_marker.add(
        dwg.path(
            d="M0,0 L4,2 L0,4 z",
            fill='black',
            stroke='none'
        )
    )
    
    # Right angle marker
    right_angle_marker = defs.add(
        dwg.marker(
            id='right_angle',
            insert=(5, 5),
            size=(10, 10),
            orient='auto',
            markerUnits='userSpaceOnUse'
        )
    )
    right_angle_marker.add(
        dwg.path(
            d="M0,5 L5,5 L5,0",
            fill='none',
            stroke='black',
            stroke_width=1
        )
    )
    
    # Dot markers
    dot_marker = defs.add(
        dwg.marker(
            id='dot',
            insert=(2, 2),
            size=(4, 4),
            markerUnits='userSpaceOnUse'
        )
    )
    dot_marker.add(
        dwg.circle(
            center=(2, 2),
            r=2,
            fill='black'
        )
    )
    
    # Tick mark pattern
    tick_pattern = defs.add(
        dwg.pattern(
            id='ticks',
            patternUnits='userSpaceOnUse',
            size=(10, 10)
        )
    )
    tick_pattern.add(
        dwg.line(
            start=(5, 0),
            end=(5, 10),
            stroke='black',
            stroke_width=1
        )
    )


def _create_svg_groups(dwg: svgwrite.Drawing) -> Dict[str, Group]:
    """Create layered groups for different element types."""
    groups = {
        'background': dwg.g(id='background', class_='background-layer'),
        'auxiliary': dwg.g(id='auxiliary', class_='auxiliary-layer'),
        'main': dwg.g(id='main', class_='main-layer'),
        'symbols': dwg.g(id='symbols', class_='symbols-layer'),
        'text': dwg.g(id='text', class_='text-layer')
    }
    
    # Set default styles for groups
    groups['main'].attribs.update({
        'stroke': 'black',
        'stroke-width': '2',
        'fill': 'none',
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round'
    })
    
    groups['auxiliary'].attribs.update({
        'stroke': 'black',
        'stroke-width': '1',
        'fill': 'none',
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
        'opacity': '0.7'
    })
    
    groups['symbols'].attribs.update({
        'stroke': 'black',
        'stroke-width': '1',
        'fill': 'none'
    })
    
    groups['text'].attribs.update({
        'font-family': 'Arial, sans-serif',
        'font-size': '14',
        'text-anchor': 'middle',
        'dominant-baseline': 'central'
    })
    
    return groups


def _add_geometric_elements(dwg: svgwrite.Drawing, groups: Dict[str, Group], graph: TopologyGraph, config: Config) -> None:
    """Add geometric elements (lines, circles, arcs) to SVG."""
    
    for edge in graph.edges:
        if isinstance(edge.geometry, LineSeg):
            _add_line_element(dwg, groups, edge, config)
        elif isinstance(edge.geometry, CircleArc):
            _add_circle_arc_element(dwg, groups, edge, config)


def _add_line_element(dwg: svgwrite.Drawing, groups: Dict[str, Group], edge: Edge, config: Config) -> None:
    """Add a line segment to SVG."""
    line_geom = edge.geometry
    
    # Determine target group
    group = groups['auxiliary'] if line_geom.dashed else groups['main']
    
    # Create line element
    line = dwg.line(
        start=(line_geom.p1[0] * config.svg.scale, line_geom.p1[1] * config.svg.scale),
        end=(line_geom.p2[0] * config.svg.scale, line_geom.p2[1] * config.svg.scale)
    )
    
    # Set line properties
    if line_geom.dashed:
        line.attribs['stroke-dasharray'] = config.svg.dash_pattern
    
    if line_geom.thickness != 2:
        line.attribs['stroke-width'] = str(line_geom.thickness)
    
    # Add semantic metadata using class
    line.attribs['class'] = f"line-element role-{line_geom.role}"
    
    # Add arrow markers if this line has arrow symbols
    # (This would require symbol binding information)
    
    group.add(line)


def _add_circle_arc_element(dwg: svgwrite.Drawing, groups: Dict[str, Group], edge: Edge, config: Config) -> None:
    """Add a circle or arc to SVG."""
    circle_geom = edge.geometry
    group = groups['main']
    
    center_x = circle_geom.cx * config.svg.scale
    center_y = circle_geom.cy * config.svg.scale
    radius = circle_geom.r * config.svg.scale
    
    if circle_geom.is_full_circle:
        # Full circle
        circle = dwg.circle(
            center=(center_x, center_y),
            r=radius
        )
        
        # Add metadata using class
        circle.attribs['class'] = 'circle-element'
        
        group.add(circle)
    else:
        # Arc - create using path
        start_angle = circle_geom.theta1
        end_angle = circle_geom.theta2
        
        # Calculate arc endpoints
        start_x = center_x + radius * math.cos(start_angle)
        start_y = center_y + radius * math.sin(start_angle)
        end_x = center_x + radius * math.cos(end_angle)
        end_y = center_y + radius * math.sin(end_angle)
        
        # Determine if this is a large arc
        angle_diff = end_angle - start_angle
        if angle_diff < 0:
            angle_diff += 2 * math.pi
        large_arc = 1 if angle_diff > math.pi else 0
        
        # Create arc path
        path_data = f"M {start_x:.2f},{start_y:.2f} A {radius:.2f},{radius:.2f} 0 {large_arc},1 {end_x:.2f},{end_y:.2f}"
        
        arc_path = dwg.path(d=path_data)
        
        # Add metadata using class
        arc_path.attribs['class'] = 'arc-element'
        
        group.add(arc_path)


def _add_symbol_elements(dwg: svgwrite.Drawing, groups: Dict[str, Group], graph: TopologyGraph, config: Config) -> None:
    """Add symbol elements to SVG."""
    group = groups['symbols']
    
    # Add symbols based on symbol bindings
    for symbol_id, element_ids in graph.symbol_bindings.items():
        # This would require access to the original symbol data
        # For now, we'll add placeholder symbols
        
        if 'right_angle' in symbol_id and len(element_ids) >= 2:
            # Find intersection point of the two elements
            edge1 = graph.get_edge_by_id(element_ids[0])
            edge2 = graph.get_edge_by_id(element_ids[1])
            
            if edge1 and edge2:
                intersection = _find_edge_intersection(edge1, edge2)
                if intersection:
                    _add_right_angle_symbol(dwg, group, intersection, config)


def _add_right_angle_symbol(dwg: svgwrite.Drawing, group: Group, position: Tuple[float, float], config: Config) -> None:
    """Add a right angle symbol at the specified position."""
    x, y = position
    x *= config.svg.scale
    y *= config.svg.scale
    
    # Create small square for right angle
    size = 8
    square = dwg.rect(
        insert=(x - size/2, y - size/2),
        size=(size, size),
        fill='none',
        stroke='black',
        stroke_width=1
    )
    
    square.attribs['class'] = 'right-angle-symbol'
    
    group.add(square)


def _add_text_elements(dwg: svgwrite.Drawing, groups: Dict[str, Group], graph: TopologyGraph, config: Config) -> None:
    """Add text elements to SVG."""
    group = groups['text']
    
    # Add node labels
    for node in graph.nodes:
        if node.label:
            text_elem = dwg.text(
                node.label,
                insert=(node.x * config.svg.scale, node.y * config.svg.scale)
            )
            
            # Add metadata using class
            text_elem.attribs['class'] = f"node-label {node.node_type}-node"
            
            group.add(text_elem)
    
    # Add text from text bindings (OCR results bound to elements)
    for text_id, element_id in graph.text_bindings.items():
        node = graph.get_node_by_id(element_id)
        if node:
            # The text content would need to be passed from the OCR results
            # For now, we'll skip adding text content since we don't have access to it
            pass


def _find_edge_intersection(edge1: Edge, edge2: Edge) -> Optional[Tuple[float, float]]:
    """Find intersection point between two edges."""
    if isinstance(edge1.geometry, LineSeg) and isinstance(edge2.geometry, LineSeg):
        line1 = edge1.geometry
        line2 = edge2.geometry
        
        # Line intersection calculation
        x1, y1, x2, y2 = line1.p1[0], line1.p1[1], line1.p2[0], line1.p2[1]
        x3, y3, x4, y4 = line2.p1[0], line2.p1[1], line2.p2[0], line2.p2[1]
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None  # Lines are parallel
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        # Calculate intersection point
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        
        return (ix, iy)
    
    return None


def _add_construction_elements(dwg: svgwrite.Drawing, groups: Dict[str, Group], graph: TopologyGraph, config: Config) -> None:
    """Add construction elements like guidelines and dimensions."""
    # This could be extended to add construction lines, dimension annotations, etc.
    pass


def create_svg_style_definitions(config: Config) -> str:
    """Create CSS style definitions for SVG."""
    styles = f"""
    <defs>
        <style type="text/css">
        <![CDATA[
            .main-layer {{
                stroke: black;
                stroke-width: {config.svg.stroke_main}px;
                fill: none;
                stroke-linecap: round;
                stroke-linejoin: round;
            }}
            .auxiliary-layer {{
                stroke: black;
                stroke-width: {config.svg.stroke_aux}px;
                fill: none;
                stroke-linecap: round;
                stroke-linejoin: round;
                opacity: 0.7;
                stroke-dasharray: {config.svg.dash_pattern};
            }}
            .symbols-layer {{
                stroke: black;
                stroke-width: 1px;
                fill: none;
            }}
            .text-layer {{
                font-family: Arial, sans-serif;
                font-size: 14px;
                text-anchor: middle;
                dominant-baseline: central;
                fill: black;
            }}
            .node-label {{
                font-weight: bold;
                font-size: 12px;
            }}
            .dimension {{
                font-size: 10px;
                fill: blue;
            }}
        ]]>
        </style>
    </defs>
    """
    return styles


def add_svg_metadata(dwg: svgwrite.Drawing, image_path: str, graph: TopologyGraph) -> None:
    """Add metadata to SVG document."""
    # Add title
    dwg.add(dwg.title(f"Mathematical diagram converted from {Path(image_path).name}"))
    
    # Add description with statistics
    description = f"""
    Converted by PNG2SVG
    Original image: {Path(image_path).name}
    Elements: {len(graph.edges)} geometric elements, {len(graph.nodes)} nodes, {len(graph.relations)} relationships
    Generated with semantic annotations and layered structure
    """
    dwg.add(dwg.desc(description.strip()))


def optimize_svg_output(svg_content: str) -> str:
    """Optimize SVG content for smaller file size and better compatibility."""
    # This could implement various SVG optimizations:
    # - Remove redundant attributes
    # - Optimize path data
    # - Combine similar elements
    # - Minimize precision of coordinates
    
    # For now, just return the original content
    return svg_content


def validate_svg_output(svg_path: str) -> bool:
    """Validate that the generated SVG is well-formed."""
    try:
        # Simple validation - check if file exists and has content
        svg_file = Path(svg_path)
        if not svg_file.exists():
            return False
        
        content = svg_file.read_text(encoding='utf-8')
        
        # Basic checks
        if not content.strip():
            return False
        
        if not content.startswith('<?xml') and not content.startswith('<svg'):
            return False
        
        if '</svg>' not in content:
            return False
        
        return True
        
    except Exception:
        return False