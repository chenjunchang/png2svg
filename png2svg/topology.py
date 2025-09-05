"""
Topology and relationship building module for PNG2SVG system.
Builds geometric relationships and semantic graph from detected primitives.
"""

import math
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict, Set
import uuid
import networkx as nx
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

from .config import Config
from .detect_primitives import Primitives, LineSeg, CircleArc
from .detect_symbols import Symbols, Symbol
from .ocr_text import OCRItem


@dataclass
class Node:
    """Geometric node (point) in the topology."""
    x: float
    y: float
    tag: str = ""                # A/B/C/O/H/M... (from OCR)
    kind: str = "point"          # point/mid/foot/center/vertex/intersection
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    confidence: float = 1.0


@dataclass 
class Edge:
    """Geometric edge (line/arc) with semantic attributes."""
    geom: Union[LineSeg, CircleArc]
    role: str = "main"           # main/aux/hidden
    attrs: Dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class Relation:
    """Geometric relationship between elements."""
    type: str                    # parallel/perp/equal_len/point_on/angle_group
    members: List[str]           # IDs of related elements
    conf: float
    attrs: Dict = field(default_factory=dict)


@dataclass
class Graph:
    """Complete geometric topology graph."""
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list) 
    relations: List[Relation] = field(default_factory=list)
    nx_graph: nx.Graph = field(default_factory=nx.Graph)


def build(prim: Primitives, sym: Symbols, ocr: List[OCRItem], cfg: Config) -> Graph:
    """
    Build topology graph from detected elements.
    
    - Calculate all intersections/endpoints using Shapely
    - Bind symbols to nearest line segments/angles
    - Bind OCR text to nearest elements
    - Generate initial relations (parallel/perp/point_on/equal_len/angle_group)
    
    Args:
        prim: Detected geometric primitives
        sym: Detected symbols
        ocr: OCR text detections
        cfg: Configuration object
        
    Returns:
        Graph: Complete topology with nodes, edges, and relations
    """
    logger = logging.getLogger('png2svg.topology')
    logger.info("Building topology graph")
    
    graph = Graph()
    
    # Step 1: Extract nodes from lines and circles
    nodes = extract_nodes(prim, cfg)
    graph.nodes = nodes
    logger.info(f"Extracted {len(nodes)} nodes")
    
    # Step 2: Create edges from primitives
    edges = create_edges(prim)
    graph.edges = edges
    logger.info(f"Created {len(edges)} edges")
    
    # Step 3: Build NetworkX graph for analysis
    build_nx_graph(graph)
    
    # Step 4: Bind symbols to elements
    bind_symbols_to_elements(graph, sym, cfg)
    logger.info(f"Bound {len(sym.items)} symbols to elements")
    
    # Step 5: Bind OCR text to elements
    bind_ocr_to_elements(graph, ocr, cfg)
    logger.info(f"Bound {len(ocr)} OCR items to elements")
    
    # Step 6: Detect geometric relationships
    relations = detect_relationships(graph, sym, cfg)
    graph.relations = relations
    logger.info(f"Detected {len(relations)} relationships")
    
    return graph


def extract_nodes(prim: Primitives, cfg: Config) -> List[Node]:
    """
    Extract nodes (points) from geometric primitives.
    
    Args:
        prim: Detected primitives
        cfg: Configuration object
        
    Returns:
        List of nodes
    """
    nodes = []
    node_positions = {}  # Track unique positions
    
    # Extract endpoints from lines
    for line in prim.lines:
        # Add start point
        p1_key = (round(line.p1[0], 1), round(line.p1[1], 1))
        if p1_key not in node_positions:
            node = Node(
                x=line.p1[0],
                y=line.p1[1],
                kind="endpoint",
                confidence=line.confidence
            )
            nodes.append(node)
            node_positions[p1_key] = node
        
        # Add end point
        p2_key = (round(line.p2[0], 1), round(line.p2[1], 1))
        if p2_key not in node_positions:
            node = Node(
                x=line.p2[0],
                y=line.p2[1],
                kind="endpoint",
                confidence=line.confidence
            )
            nodes.append(node)
            node_positions[p2_key] = node
    
    # Find line intersections
    intersections = find_line_intersections(prim.lines)
    for point in intersections:
        p_key = (round(point[0], 1), round(point[1], 1))
        if p_key not in node_positions:
            node = Node(
                x=point[0],
                y=point[1],
                kind="intersection",
                confidence=0.8
            )
            nodes.append(node)
            node_positions[p_key] = node
    
    # Extract circle centers
    for circle in prim.circles:
        center_key = (round(circle.cx, 1), round(circle.cy, 1))
        if center_key not in node_positions:
            node = Node(
                x=circle.cx,
                y=circle.cy,
                kind="center",
                confidence=circle.confidence
            )
            nodes.append(node)
            node_positions[center_key] = node
    
    return nodes


def find_line_intersections(lines: List[LineSeg]) -> List[Tuple[float, float]]:
    """
    Find all intersection points between line segments.
    
    Args:
        lines: List of line segments
        
    Returns:
        List of intersection points
    """
    intersections = []
    
    for i, line1 in enumerate(lines):
        geom1 = LineString([line1.p1, line1.p2])
        
        for j, line2 in enumerate(lines[i+1:], i+1):
            geom2 = LineString([line2.p1, line2.p2])
            
            if geom1.intersects(geom2):
                intersection = geom1.intersection(geom2)
                
                if intersection.geom_type == 'Point':
                    intersections.append((intersection.x, intersection.y))
                elif intersection.geom_type == 'MultiPoint':
                    for point in intersection.geoms:
                        intersections.append((point.x, point.y))
    
    return intersections


def create_edges(prim: Primitives) -> List[Edge]:
    """
    Create edges from geometric primitives.
    
    Args:
        prim: Detected primitives
        
    Returns:
        List of edges
    """
    edges = []
    
    # Create edges from lines
    for line in prim.lines:
        edge = Edge(
            geom=line,
            role=line.role,
            attrs={
                'dashed': line.dashed,
                'thickness': line.thickness
            }
        )
        edges.append(edge)
    
    # Create edges from circles/arcs
    for circle in prim.circles:
        edge = Edge(
            geom=circle,
            role="main",
            attrs={
                'kind': circle.kind
            }
        )
        edges.append(edge)
    
    return edges


def build_nx_graph(graph: Graph) -> None:
    """
    Build NetworkX graph for topological analysis.
    
    Args:
        graph: Topology graph
    """
    G = nx.Graph()
    
    # Add nodes
    for node in graph.nodes:
        G.add_node(node.id, x=node.x, y=node.y, kind=node.kind, tag=node.tag)
    
    # Add edges for line segments
    for edge in graph.edges:
        if isinstance(edge.geom, LineSeg):
            # Find nodes at line endpoints
            p1_node = find_nearest_node(graph.nodes, edge.geom.p1)
            p2_node = find_nearest_node(graph.nodes, edge.geom.p2)
            
            if p1_node and p2_node:
                G.add_edge(p1_node.id, p2_node.id, 
                          edge_id=edge.id,
                          role=edge.role,
                          attrs=edge.attrs)
    
    graph.nx_graph = G


def find_nearest_node(nodes: List[Node], point: Tuple[float, float], 
                      threshold: float = 5.0) -> Optional[Node]:
    """
    Find the nearest node to a point within threshold.
    
    Args:
        nodes: List of nodes
        point: Target point (x, y)
        threshold: Maximum distance threshold
        
    Returns:
        Nearest node or None if none within threshold
    """
    min_dist = float('inf')
    nearest = None
    
    for node in nodes:
        dist = math.hypot(node.x - point[0], node.y - point[1])
        if dist < min_dist and dist <= threshold:
            min_dist = dist
            nearest = node
    
    return nearest


def bind_symbols_to_elements(graph: Graph, sym: Symbols, cfg: Config) -> None:
    """
    Bind detected symbols to their corresponding geometric elements.
    
    Args:
        graph: Topology graph
        sym: Detected symbols
        cfg: Configuration object
    """
    for symbol in sym.items:
        # Get symbol center
        sx = symbol.bbox[0] + symbol.bbox[2] / 2
        sy = symbol.bbox[1] + symbol.bbox[3] / 2
        
        if symbol.cls == "right_angle":
            bind_right_angle_symbol(graph, symbol, (sx, sy))
        elif symbol.cls.startswith("arc_"):
            bind_arc_symbol(graph, symbol, (sx, sy))
        elif symbol.cls.startswith("tick_"):
            bind_tick_symbol(graph, symbol, (sx, sy))
        elif symbol.cls == "parallel_mark":
            bind_parallel_symbol(graph, symbol, (sx, sy))
        elif symbol.cls == "arrow_head":
            bind_arrow_symbol(graph, symbol, (sx, sy))


def bind_right_angle_symbol(graph: Graph, symbol: Symbol, center: Tuple[float, float]) -> None:
    """
    Bind right angle symbol to two perpendicular lines.
    
    Args:
        graph: Topology graph
        symbol: Right angle symbol
        center: Symbol center position
    """
    # Find two nearest lines
    nearest_lines = find_nearest_lines(graph.edges, center, max_count=2)
    
    if len(nearest_lines) >= 2:
        line1, line2 = nearest_lines[:2]
        
        # Check if lines are approximately perpendicular
        angle = calculate_angle_between_lines(line1.geom, line2.geom)
        if 80 <= angle <= 100:  # Within 10 degrees of perpendicular
            # Mark these lines as perpendicular
            line1.attrs['perpendicular_to'] = line2.id
            line2.attrs['perpendicular_to'] = line1.id
            line1.attrs['has_right_angle_symbol'] = True
            line2.attrs['has_right_angle_symbol'] = True


def bind_arc_symbol(graph: Graph, symbol: Symbol, center: Tuple[float, float]) -> None:
    """
    Bind arc symbol (angle marker) to angle formed by two lines.
    
    Args:
        graph: Topology graph
        symbol: Arc symbol
        center: Symbol center position
    """
    # Extract arc count (arc_1, arc_2, arc_3)
    arc_count = int(symbol.cls.split('_')[1])
    
    # Find vertex node nearest to symbol
    nearest_vertex = find_nearest_node(graph.nodes, center, threshold=30)
    
    if nearest_vertex:
        # Find lines connected to this vertex
        connected_lines = find_lines_connected_to_node(graph, nearest_vertex)
        
        if len(connected_lines) >= 2:
            # Mark angle with arc count
            for line in connected_lines[:2]:
                line.attrs['angle_arcs'] = arc_count
                line.attrs['angle_vertex'] = nearest_vertex.id


def bind_tick_symbol(graph: Graph, symbol: Symbol, center: Tuple[float, float]) -> None:
    """
    Bind tick marks (equal length indicators) to lines.
    
    Args:
        graph: Topology graph
        symbol: Tick symbol
        center: Symbol center position
    """
    # Extract tick count
    tick_count = int(symbol.cls.split('_')[1])
    
    # Find nearest line
    nearest_lines = find_nearest_lines(graph.edges, center, max_count=1)
    
    if nearest_lines:
        line = nearest_lines[0]
        line.attrs['tick_marks'] = tick_count
        line.attrs['has_equal_length'] = True


def bind_parallel_symbol(graph: Graph, symbol: Symbol, center: Tuple[float, float]) -> None:
    """
    Bind parallel marks to lines.
    
    Args:
        graph: Topology graph
        symbol: Parallel symbol
        center: Symbol center position
    """
    # Find nearest line
    nearest_lines = find_nearest_lines(graph.edges, center, max_count=1)
    
    if nearest_lines:
        line = nearest_lines[0]
        line.attrs['has_parallel_mark'] = True


def bind_arrow_symbol(graph: Graph, symbol: Symbol, center: Tuple[float, float]) -> None:
    """
    Bind arrow head to line endpoints.
    
    Args:
        graph: Topology graph
        symbol: Arrow symbol
        center: Symbol center position
    """
    # Find nearest line endpoint
    min_dist = float('inf')
    nearest_line = None
    is_p1 = True
    
    for edge in graph.edges:
        if isinstance(edge.geom, LineSeg):
            dist_p1 = math.hypot(edge.geom.p1[0] - center[0], edge.geom.p1[1] - center[1])
            dist_p2 = math.hypot(edge.geom.p2[0] - center[0], edge.geom.p2[1] - center[1])
            
            if dist_p1 < min_dist:
                min_dist = dist_p1
                nearest_line = edge
                is_p1 = True
            
            if dist_p2 < min_dist:
                min_dist = dist_p2
                nearest_line = edge
                is_p1 = False
    
    if nearest_line and min_dist < 20:  # Within 20 pixels
        if is_p1:
            nearest_line.attrs['arrow_at_p1'] = True
        else:
            nearest_line.attrs['arrow_at_p2'] = True


def bind_ocr_to_elements(graph: Graph, ocr: List[OCRItem], cfg: Config) -> None:
    """
    Bind OCR text to nearest geometric elements.
    
    Args:
        graph: Topology graph
        ocr: OCR detections
        cfg: Configuration object
    """
    for ocr_item in ocr:
        # Get OCR center
        ox = ocr_item.bbox[0] + ocr_item.bbox[2] / 2
        oy = ocr_item.bbox[1] + ocr_item.bbox[3] / 2
        
        if ocr_item.category == "label":
            # Point labels (A, B, C, etc.)
            bind_label_to_node(graph, ocr_item, (ox, oy))
        elif ocr_item.category == "angle":
            # Angle measurements (30°, 45°, etc.)
            bind_angle_to_vertex(graph, ocr_item, (ox, oy))
        elif ocr_item.category == "length":
            # Length measurements
            bind_length_to_line(graph, ocr_item, (ox, oy))


def bind_label_to_node(graph: Graph, ocr_item: OCRItem, center: Tuple[float, float]) -> None:
    """
    Bind label text to nearest node.
    
    Args:
        graph: Topology graph
        ocr_item: OCR text item
        center: Text center position
    """
    nearest_node = find_nearest_node(graph.nodes, center, threshold=30)
    
    if nearest_node:
        nearest_node.tag = ocr_item.text
        
        # Update node kind based on label
        if ocr_item.text == 'O':
            nearest_node.kind = "center"  # Likely circle center
        elif ocr_item.text == 'M':
            nearest_node.kind = "midpoint"
        elif ocr_item.text == 'H':
            nearest_node.kind = "foot"  # Perpendicular foot


def bind_angle_to_vertex(graph: Graph, ocr_item: OCRItem, center: Tuple[float, float]) -> None:
    """
    Bind angle measurement to vertex.
    
    Args:
        graph: Topology graph
        ocr_item: OCR angle text
        center: Text center position
    """
    nearest_node = find_nearest_node(graph.nodes, center, threshold=50)
    
    if nearest_node:
        # Store angle value at vertex
        if 'angle_value' not in nearest_node.__dict__:
            nearest_node.__dict__['angle_value'] = ocr_item.text


def bind_length_to_line(graph: Graph, ocr_item: OCRItem, center: Tuple[float, float]) -> None:
    """
    Bind length measurement to line.
    
    Args:
        graph: Topology graph
        ocr_item: OCR length text
        center: Text center position
    """
    nearest_lines = find_nearest_lines(graph.edges, center, max_count=1)
    
    if nearest_lines:
        line = nearest_lines[0]
        line.attrs['length_label'] = ocr_item.text


def detect_relationships(graph: Graph, sym: Symbols, cfg: Config) -> List[Relation]:
    """
    Detect geometric relationships between elements.
    
    Args:
        graph: Topology graph
        sym: Detected symbols
        cfg: Configuration object
        
    Returns:
        List of detected relationships
    """
    relations = []
    
    # Detect parallel lines
    parallel_rels = detect_parallel_lines(graph)
    relations.extend(parallel_rels)
    
    # Detect perpendicular lines
    perp_rels = detect_perpendicular_lines(graph)
    relations.extend(perp_rels)
    
    # Detect equal length lines
    equal_len_rels = detect_equal_length_lines(graph)
    relations.extend(equal_len_rels)
    
    # Detect point-on relationships
    point_on_rels = detect_point_on_relationships(graph)
    relations.extend(point_on_rels)
    
    # Detect angle groups (equal angles)
    angle_group_rels = detect_angle_groups(graph)
    relations.extend(angle_group_rels)
    
    return relations


def detect_parallel_lines(graph: Graph) -> List[Relation]:
    """
    Detect parallel line relationships.
    
    Args:
        graph: Topology graph
        
    Returns:
        List of parallel relationships
    """
    relations = []
    lines = [edge for edge in graph.edges if isinstance(edge.geom, LineSeg)]
    
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines[i+1:], i+1):
            # Check if lines have parallel marks
            if (line1.attrs.get('has_parallel_mark') and 
                line2.attrs.get('has_parallel_mark')):
                relations.append(Relation(
                    type="parallel",
                    members=[line1.id, line2.id],
                    conf=0.9
                ))
                continue
            
            # Check geometric parallelism
            angle = calculate_angle_between_lines(line1.geom, line2.geom)
            if angle < 5 or angle > 175:  # Within 5 degrees of parallel
                relations.append(Relation(
                    type="parallel",
                    members=[line1.id, line2.id],
                    conf=0.7
                ))
    
    return relations


def detect_perpendicular_lines(graph: Graph) -> List[Relation]:
    """
    Detect perpendicular line relationships.
    
    Args:
        graph: Topology graph
        
    Returns:
        List of perpendicular relationships
    """
    relations = []
    lines = [edge for edge in graph.edges if isinstance(edge.geom, LineSeg)]
    
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines[i+1:], i+1):
            # Check if lines have right angle symbol
            if (line1.attrs.get('has_right_angle_symbol') and 
                line2.attrs.get('has_right_angle_symbol')):
                relations.append(Relation(
                    type="perpendicular",
                    members=[line1.id, line2.id],
                    conf=0.95
                ))
                continue
            
            # Check geometric perpendicularity
            angle = calculate_angle_between_lines(line1.geom, line2.geom)
            if 85 <= angle <= 95:  # Within 5 degrees of perpendicular
                relations.append(Relation(
                    type="perpendicular",
                    members=[line1.id, line2.id],
                    conf=0.7
                ))
    
    return relations


def detect_equal_length_lines(graph: Graph) -> List[Relation]:
    """
    Detect lines with equal length markings.
    
    Args:
        graph: Topology graph
        
    Returns:
        List of equal length relationships
    """
    relations = []
    
    # Group lines by tick mark count
    tick_groups = {}
    for edge in graph.edges:
        if isinstance(edge.geom, LineSeg):
            tick_count = edge.attrs.get('tick_marks', 0)
            if tick_count > 0:
                if tick_count not in tick_groups:
                    tick_groups[tick_count] = []
                tick_groups[tick_count].append(edge.id)
    
    # Create relationships for each group
    for tick_count, line_ids in tick_groups.items():
        if len(line_ids) > 1:
            relations.append(Relation(
                type="equal_length",
                members=line_ids,
                conf=0.85,
                attrs={'tick_count': tick_count}
            ))
    
    return relations


def detect_point_on_relationships(graph: Graph) -> List[Relation]:
    """
    Detect point-on-line and point-on-circle relationships.
    
    Args:
        graph: Topology graph
        
    Returns:
        List of point-on relationships
    """
    relations = []
    
    # Check points on lines
    for node in graph.nodes:
        for edge in graph.edges:
            if isinstance(edge.geom, LineSeg):
                # Check if point is on line segment
                line = LineString([edge.geom.p1, edge.geom.p2])
                point = Point(node.x, node.y)
                
                if line.distance(point) < 2.0:  # Within 2 pixels
                    # Check it's not an endpoint
                    is_endpoint = (
                        math.hypot(node.x - edge.geom.p1[0], node.y - edge.geom.p1[1]) < 2 or
                        math.hypot(node.x - edge.geom.p2[0], node.y - edge.geom.p2[1]) < 2
                    )
                    
                    if not is_endpoint:
                        relations.append(Relation(
                            type="point_on_line",
                            members=[node.id, edge.id],
                            conf=0.8
                        ))
            
            elif isinstance(edge.geom, CircleArc):
                # Check if point is on circle
                dist_to_center = math.hypot(
                    node.x - edge.geom.cx,
                    node.y - edge.geom.cy
                )
                
                if abs(dist_to_center - edge.geom.r) < 3.0:  # Within 3 pixels of radius
                    relations.append(Relation(
                        type="point_on_circle",
                        members=[node.id, edge.id],
                        conf=0.8
                    ))
    
    return relations


def detect_angle_groups(graph: Graph) -> List[Relation]:
    """
    Detect groups of equal angles based on arc markings.
    
    Args:
        graph: Topology graph
        
    Returns:
        List of angle group relationships
    """
    relations = []
    
    # Group vertices by arc count
    arc_groups = {}
    for edge in graph.edges:
        if isinstance(edge.geom, LineSeg):
            arc_count = edge.attrs.get('angle_arcs', 0)
            vertex_id = edge.attrs.get('angle_vertex')
            
            if arc_count > 0 and vertex_id:
                if arc_count not in arc_groups:
                    arc_groups[arc_count] = set()
                arc_groups[arc_count].add(vertex_id)
    
    # Create relationships for each group
    for arc_count, vertex_ids in arc_groups.items():
        if len(vertex_ids) > 1:
            relations.append(Relation(
                type="angle_group",
                members=list(vertex_ids),
                conf=0.85,
                attrs={'arc_count': arc_count}
            ))
    
    return relations


def calculate_angle_between_lines(line1: LineSeg, line2: LineSeg) -> float:
    """
    Calculate angle between two line segments in degrees.
    
    Args:
        line1: First line segment
        line2: Second line segment
        
    Returns:
        Angle in degrees (0-180)
    """
    # Calculate direction vectors
    v1 = np.array([line1.p2[0] - line1.p1[0], line1.p2[1] - line1.p1[1]])
    v2 = np.array([line2.p2[0] - line2.p1[0], line2.p2[1] - line2.p1[1]])
    
    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calculate angle using dot product
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = math.degrees(math.acos(abs(cos_angle)))
    
    return angle


def find_nearest_lines(edges: List[Edge], point: Tuple[float, float], 
                       max_count: int = 1) -> List[Edge]:
    """
    Find nearest line segments to a point.
    
    Args:
        edges: List of edges
        point: Target point
        max_count: Maximum number of lines to return
        
    Returns:
        List of nearest edges
    """
    line_distances = []
    
    for edge in edges:
        if isinstance(edge.geom, LineSeg):
            # Calculate distance from point to line segment
            line = LineString([edge.geom.p1, edge.geom.p2])
            p = Point(point)
            dist = line.distance(p)
            line_distances.append((dist, edge))
    
    # Sort by distance and return top N
    line_distances.sort(key=lambda x: x[0])
    return [edge for _, edge in line_distances[:max_count]]


def find_lines_connected_to_node(graph: Graph, node: Node) -> List[Edge]:
    """
    Find all lines connected to a given node.
    
    Args:
        graph: Topology graph
        node: Target node
        
    Returns:
        List of connected edges
    """
    connected = []
    threshold = 5.0  # Distance threshold
    
    for edge in graph.edges:
        if isinstance(edge.geom, LineSeg):
            dist_p1 = math.hypot(node.x - edge.geom.p1[0], node.y - edge.geom.p1[1])
            dist_p2 = math.hypot(node.x - edge.geom.p2[0], node.y - edge.geom.p2[1])
            
            if dist_p1 < threshold or dist_p2 < threshold:
                connected.append(edge)
    
    return connected