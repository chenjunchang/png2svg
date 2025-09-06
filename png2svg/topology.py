"""
Topological analysis and relationship building for PNG2SVG.

Builds a graph representation of geometric elements and their relationships,
including intersections, parallel/perpendicular constraints, and semantic bindings.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
import uuid

import numpy as np
from shapely.geometry import LineString, Point as ShapelyPoint, GeometryCollection
from shapely.ops import unary_union
import networkx as nx

from .config import Config
from .detect_primitives import PrimitivesResult, LineSeg, CircleArc
from .detect_symbols import SymbolsResult, Symbol
from .ocr_text import OCRResults, OCRResult


@dataclass
class Node:
    """Graph node representing a geometric point."""
    x: float
    y: float
    node_type: str = "point"          # "point", "intersection", "endpoint", "center", "vertex"
    label: str = ""                   # Text label (A, B, C, etc.)
    semantic_role: str = ""           # "vertex", "midpoint", "foot", "center", etc.
    confidence: float = 1.0           # Confidence in node position
    properties: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    @property
    def position(self) -> Tuple[float, float]:
        """Get node position as tuple."""
        return (self.x, self.y)


@dataclass
class Edge:
    """Graph edge representing a geometric element."""
    geometry: Union[LineSeg, CircleArc]  # Original geometric element
    start_node_id: Optional[str] = None  # Start node ID
    end_node_id: Optional[str] = None    # End node ID
    edge_type: str = "geometric"         # "geometric", "auxiliary", "construction"
    properties: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class Relation:
    """Geometric relationship between elements."""
    relation_type: str                   # "parallel", "perpendicular", "equal_length", "point_on", "angle", etc.
    members: List[str]                   # IDs of participating elements
    confidence: float = 1.0              # Confidence in relationship
    properties: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class TopologyGraph:
    """Complete topological representation of the diagram."""
    nodes: List[Node]
    edges: List[Edge]
    relations: List[Relation]
    symbol_bindings: Dict[str, List[str]] = field(default_factory=dict)  # symbol_id -> element_ids
    text_bindings: Dict[str, str] = field(default_factory=dict)          # text_id -> element_id
    
    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_edge_by_id(self, edge_id: str) -> Optional[Edge]:
        """Get edge by ID."""
        for edge in self.edges:
            if edge.id == edge_id:
                return edge
        return None


def build(
    primitives: PrimitivesResult,
    symbols: SymbolsResult, 
    ocr: OCRResults,
    config: Config
) -> TopologyGraph:
    """
    Build topological graph from detected elements.
    
    Args:
        primitives: Detected geometric primitives
        symbols: Detected symbols
        ocr: OCR text results
        config: Configuration object
        
    Returns:
        TopologyGraph: Complete topological representation
    """
    logger = logging.getLogger(__name__)
    
    logger.debug("Building topological graph...")
    
    # Initialize graph
    graph = TopologyGraph(nodes=[], edges=[], relations=[])
    
    # Step 1: Create nodes and edges from primitives
    _create_primitive_elements(graph, primitives)
    logger.debug(f"Created {len(graph.nodes)} nodes and {len(graph.edges)} edges from primitives")
    
    # Step 2: Find intersections and add intersection nodes
    _find_and_add_intersections(graph)
    logger.debug(f"Found intersections, now have {len(graph.nodes)} nodes")
    
    # Step 3: Associate symbols with geometric elements
    _associate_symbols(graph, symbols, config)
    logger.debug(f"Associated {len(symbols.items)} symbols")
    
    # Step 4: Associate text with geometric elements
    _associate_text(graph, ocr, config)
    logger.debug(f"Associated {len(ocr.items)} text regions")
    
    # Step 5: Infer geometric relationships
    _infer_relationships(graph, config)
    logger.debug(f"Inferred {len(graph.relations)} relationships")
    
    # Step 6: Enhance with symbol-based relationships
    _enhance_with_symbol_relationships(graph)
    logger.debug("Enhanced with symbol-based relationships")
    
    logger.info(f"Built topology: {len(graph.nodes)} nodes, {len(graph.edges)} edges, {len(graph.relations)} relations")
    
    return graph


def _create_primitive_elements(graph: TopologyGraph, primitives: PrimitivesResult) -> None:
    """Create nodes and edges from geometric primitives."""
    
    # Create nodes and edges for line segments
    for line in primitives.lines:
        # Create endpoint nodes
        start_node = Node(
            x=line.p1[0],
            y=line.p1[1],
            node_type="endpoint",
            confidence=line.confidence
        )
        end_node = Node(
            x=line.p2[0],
            y=line.p2[1],
            node_type="endpoint", 
            confidence=line.confidence
        )
        
        # Add nodes (will merge close ones later)
        graph.nodes.extend([start_node, end_node])
        
        # Create edge
        edge = Edge(
            geometry=line,
            start_node_id=start_node.id,
            end_node_id=end_node.id,
            edge_type="auxiliary" if line.dashed else "geometric"
        )
        graph.edges.append(edge)
    
    # Create nodes and edges for circles/arcs
    for circle in primitives.circles:
        # Create center node
        center_node = Node(
            x=circle.cx,
            y=circle.cy,
            node_type="center",
            semantic_role="center",
            confidence=circle.confidence
        )
        graph.nodes.append(center_node)
        
        # Create edge (self-loop for circles)
        edge = Edge(
            geometry=circle,
            start_node_id=center_node.id,
            end_node_id=center_node.id if circle.is_full_circle else None,
            edge_type="geometric"
        )
        graph.edges.append(edge)


def _find_and_add_intersections(graph: TopologyGraph) -> None:
    """Find intersections between geometric elements and add intersection nodes."""
    
    # Merge nearby nodes first (within 5 pixels)
    _merge_nearby_nodes(graph, threshold=5.0)
    
    # Find intersections between line segments
    line_edges = [e for e in graph.edges if isinstance(e.geometry, LineSeg)]
    
    for i in range(len(line_edges)):
        for j in range(i + 1, len(line_edges)):
            edge1, edge2 = line_edges[i], line_edges[j]
            line1, line2 = edge1.geometry, edge2.geometry
            
            # Create Shapely LineStrings
            shapely_line1 = LineString([line1.p1, line1.p2])
            shapely_line2 = LineString([line2.p1, line2.p2])
            
            # Find intersection
            intersection = shapely_line1.intersection(shapely_line2)
            
            if intersection and hasattr(intersection, 'x') and hasattr(intersection, 'y'):
                # Add intersection node
                intersection_node = Node(
                    x=intersection.x,
                    y=intersection.y,
                    node_type="intersection",
                    confidence=min(line1.confidence, line2.confidence) * 0.9
                )
                graph.nodes.append(intersection_node)


def _merge_nearby_nodes(graph: TopologyGraph, threshold: float = 5.0) -> None:
    """Merge nodes that are very close together."""
    merged_nodes = []
    node_mapping = {}  # old_id -> new_id
    
    for node in graph.nodes:
        # Check if this node is close to any already processed node
        merged_with = None
        for merged_node in merged_nodes:
            distance = math.sqrt((node.x - merged_node.x)**2 + (node.y - merged_node.y)**2)
            if distance <= threshold:
                merged_with = merged_node
                break
        
        if merged_with:
            # Map this node to the merged node
            node_mapping[node.id] = merged_with.id
            # Update merged node position (weighted average)
            weight = merged_with.confidence / (merged_with.confidence + node.confidence)
            merged_with.x = merged_with.x * weight + node.x * (1 - weight)
            merged_with.y = merged_with.y * weight + node.y * (1 - weight)
            merged_with.confidence = max(merged_with.confidence, node.confidence)
        else:
            # Keep this node
            merged_nodes.append(node)
            node_mapping[node.id] = node.id
    
    # Update graph nodes
    graph.nodes = merged_nodes
    
    # Update edge references
    for edge in graph.edges:
        if edge.start_node_id and edge.start_node_id in node_mapping:
            edge.start_node_id = node_mapping[edge.start_node_id]
        if edge.end_node_id and edge.end_node_id in node_mapping:
            edge.end_node_id = node_mapping[edge.end_node_id]


def _associate_symbols(graph: TopologyGraph, symbols: SymbolsResult, config: Config) -> None:
    """Associate symbols with nearby geometric elements."""
    
    for symbol in symbols.items:
        symbol_center = symbol.center
        associated_elements = []
        
        # Find nearby geometric elements
        for edge in graph.edges:
            if isinstance(edge.geometry, LineSeg):
                line = edge.geometry
                # Calculate distance from symbol to line
                distance = _point_to_line_distance(symbol_center, line.p1, line.p2)
                if distance <= 30:  # Within 30 pixels
                    associated_elements.append(edge.id)
            elif isinstance(edge.geometry, CircleArc):
                circle = edge.geometry
                # Calculate distance from symbol to circle center
                distance = math.sqrt(
                    (symbol_center[0] - circle.cx)**2 + 
                    (symbol_center[1] - circle.cy)**2
                )
                if abs(distance - circle.r) <= 20:  # Near the circle
                    associated_elements.append(edge.id)
        
        if associated_elements:
            graph.symbol_bindings[symbol.id] = associated_elements


def _associate_text(graph: TopologyGraph, ocr: OCRResults, config: Config) -> None:
    """Associate text with nearby geometric elements."""
    
    for text_item in ocr.items:
        text_center = (
            text_item.bbox[0] + text_item.bbox[2] / 2,
            text_item.bbox[1] + text_item.bbox[3] / 2
        )
        
        # Find closest node for labels
        if text_item.text_type in ["label", "variable"]:
            closest_node = None
            min_distance = float('inf')
            
            for node in graph.nodes:
                distance = math.sqrt(
                    (text_center[0] - node.x)**2 + 
                    (text_center[1] - node.y)**2
                )
                if distance < min_distance and distance <= 50:  # Within 50 pixels
                    min_distance = distance
                    closest_node = node
            
            if closest_node:
                closest_node.label = text_item.cleaned_text
                graph.text_bindings[text_item.id] = closest_node.id
        
        # Associate angle measurements with nearby intersection nodes
        elif text_item.text_type == "angle":
            for node in graph.nodes:
                if node.node_type == "intersection":
                    distance = math.sqrt(
                        (text_center[0] - node.x)**2 + 
                        (text_center[1] - node.y)**2
                    )
                    if distance <= 40:  # Within 40 pixels of intersection
                        graph.text_bindings[text_item.id] = node.id
                        break


def _infer_relationships(graph: TopologyGraph, config: Config) -> None:
    """Infer geometric relationships between elements."""
    
    # Find parallel lines
    _find_parallel_lines(graph)
    
    # Find perpendicular lines
    _find_perpendicular_lines(graph)
    
    # Find equal length segments
    _find_equal_length_segments(graph)
    
    # Find point-on relationships
    _find_point_on_relationships(graph)


def _find_parallel_lines(graph: TopologyGraph) -> None:
    """Find parallel line segments."""
    line_edges = [e for e in graph.edges if isinstance(e.geometry, LineSeg)]
    
    for i in range(len(line_edges)):
        for j in range(i + 1, len(line_edges)):
            edge1, edge2 = line_edges[i], line_edges[j]
            line1, line2 = edge1.geometry, edge2.geometry
            
            # Calculate angles
            angle1 = line1.angle
            angle2 = line2.angle
            
            # Check if angles are similar (parallel)
            angle_diff = abs(angle1 - angle2)
            angle_diff = min(angle_diff, math.pi - angle_diff)  # Handle wraparound
            
            if angle_diff < math.radians(5):  # Within 5 degrees
                relation = Relation(
                    relation_type="parallel",
                    members=[edge1.id, edge2.id],
                    confidence=0.8,
                    properties={'angle_diff': angle_diff}
                )
                graph.relations.append(relation)


def _find_perpendicular_lines(graph: TopologyGraph) -> None:
    """Find perpendicular line segments."""
    line_edges = [e for e in graph.edges if isinstance(e.geometry, LineSeg)]
    
    for i in range(len(line_edges)):
        for j in range(i + 1, len(line_edges)):
            edge1, edge2 = line_edges[i], line_edges[j]
            line1, line2 = edge1.geometry, edge2.geometry
            
            # Calculate angles
            angle1 = line1.angle
            angle2 = line2.angle
            
            # Check if angles are perpendicular
            angle_diff = abs(angle1 - angle2)
            perp_diff = min(abs(angle_diff - math.pi/2), abs(angle_diff - 3*math.pi/2))
            
            if perp_diff < math.radians(5):  # Within 5 degrees of 90°
                relation = Relation(
                    relation_type="perpendicular",
                    members=[edge1.id, edge2.id],
                    confidence=0.8,
                    properties={'angle_diff': angle_diff}
                )
                graph.relations.append(relation)


def _find_equal_length_segments(graph: TopologyGraph) -> None:
    """Find line segments with equal lengths."""
    line_edges = [e for e in graph.edges if isinstance(e.geometry, LineSeg)]
    
    for i in range(len(line_edges)):
        for j in range(i + 1, len(line_edges)):
            edge1, edge2 = line_edges[i], line_edges[j]
            line1, line2 = edge1.geometry, edge2.geometry
            
            # Compare lengths
            length1 = line1.length
            length2 = line2.length
            
            if length1 > 0 and length2 > 0:
                length_ratio = min(length1, length2) / max(length1, length2)
                if length_ratio > 0.95:  # Within 5% of each other
                    relation = Relation(
                        relation_type="equal_length",
                        members=[edge1.id, edge2.id],
                        confidence=0.7,
                        properties={'length1': length1, 'length2': length2}
                    )
                    graph.relations.append(relation)


def _find_point_on_relationships(graph: TopologyGraph) -> None:
    """Find points that lie on lines or circles."""
    
    for node in graph.nodes:
        for edge in graph.edges:
            # Skip if node is already an endpoint of this edge
            if node.id in [edge.start_node_id, edge.end_node_id]:
                continue
            
            if isinstance(edge.geometry, LineSeg):
                line = edge.geometry
                distance = _point_to_line_distance((node.x, node.y), line.p1, line.p2)
                
                if distance <= 3:  # Within 3 pixels of line
                    relation = Relation(
                        relation_type="point_on_line",
                        members=[node.id, edge.id],
                        confidence=0.8,
                        properties={'distance': distance}
                    )
                    graph.relations.append(relation)
            
            elif isinstance(edge.geometry, CircleArc):
                circle = edge.geometry
                distance_to_center = math.sqrt(
                    (node.x - circle.cx)**2 + (node.y - circle.cy)**2
                )
                
                if abs(distance_to_center - circle.r) <= 3:  # Within 3 pixels of circle
                    relation = Relation(
                        relation_type="point_on_circle",
                        members=[node.id, edge.id],
                        confidence=0.8,
                        properties={'distance_error': abs(distance_to_center - circle.r)}
                    )
                    graph.relations.append(relation)


def _enhance_with_symbol_relationships(graph: TopologyGraph) -> None:
    """Add relationships based on detected symbols."""
    
    for symbol_id, element_ids in graph.symbol_bindings.items():
        # Find the symbol
        symbol = None
        # Note: We'd need to pass the symbols list to access symbol details
        # For now, we'll work with the bindings
        
        if len(element_ids) >= 2:
            # Right angle symbols create perpendicular relationships
            if "right_angle" in symbol_id:  # This is a simplification
                relation = Relation(
                    relation_type="perpendicular",
                    members=element_ids[:2],
                    confidence=0.9,
                    properties={'detected_by': 'right_angle_symbol'}
                )
                graph.relations.append(relation)
            
            # Parallel marks create parallel relationships
            elif "parallel" in symbol_id:
                relation = Relation(
                    relation_type="parallel", 
                    members=element_ids[:2],
                    confidence=0.9,
                    properties={'detected_by': 'parallel_symbol'}
                )
                graph.relations.append(relation)
            
            # Tick marks suggest equal length
            elif "tick" in symbol_id:
                relation = Relation(
                    relation_type="equal_length",
                    members=element_ids,
                    confidence=0.8,
                    properties={'detected_by': 'tick_marks'}
                )
                graph.relations.append(relation)


def _point_to_line_distance(point: Tuple[float, float], line_start: Tuple[float, float], line_end: Tuple[float, float]) -> float:
    """Calculate perpendicular distance from point to line segment."""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Vector from line start to end
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        # Line is a point
        return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    
    # Parameter t for projection of point onto line
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
    
    # Clamp t to line segment
    t = max(0, min(1, t))
    
    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance from point to closest point
    return math.sqrt((x0 - closest_x)**2 + (y0 - closest_y)**2)


def get_topology_statistics(graph: TopologyGraph) -> Dict[str, Any]:
    """Calculate statistics about the topology graph."""
    
    node_types = {}
    for node in graph.nodes:
        node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
    
    edge_types = {}
    for edge in graph.edges:
        edge_type = type(edge.geometry).__name__
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    relation_types = {}
    for relation in graph.relations:
        relation_types[relation.relation_type] = relation_types.get(relation.relation_type, 0) + 1
    
    return {
        'node_count': len(graph.nodes),
        'edge_count': len(graph.edges),
        'relation_count': len(graph.relations),
        'node_types': node_types,
        'edge_types': edge_types,
        'relation_types': relation_types,
        'symbol_bindings': len(graph.symbol_bindings),
        'text_bindings': len(graph.text_bindings)
    }