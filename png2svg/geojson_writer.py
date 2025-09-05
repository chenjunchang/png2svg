"""
GeoJSON writer module for PNG2SVG system.
Exports structured metadata about the geometric elements and relationships.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from .config import Config
from .topology import Graph, Node, Edge, Relation, LineSeg, CircleArc


def write(img_path: str, graph: Graph, cfg: Config) -> str:
    """
    Export topology graph to GeoJSON format with metadata.
    
    Export structure:
    - nodes: [{id, x, y, tag, kind}]
    - edges: [{id, type: 'line'|'arc'|'circle', endpoints|center+radius, role}]
    - relations: [{type, members, conf}]
    - ocr: [{text, bbox, conf, bind_to}]
    
    Args:
        img_path: Original image path (for naming output)
        graph: Topology graph with optimized geometry
        cfg: Configuration object
        
    Returns:
        Path to generated GeoJSON file
    """
    logger = logging.getLogger('png2svg.geojson_writer')
    
    # Prepare output path
    input_name = Path(img_path).stem
    output_path = Path(cfg.output_dir) / f"{input_name}.geo.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build GeoJSON structure
    geojson = build_geojson(graph, img_path, cfg)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)
    
    logger.info(f"GeoJSON saved to: {output_path}")
    return str(output_path)


def build_geojson(graph: Graph, img_path: str, cfg: Config) -> Dict[str, Any]:
    """
    Build GeoJSON structure from topology graph.
    
    Args:
        graph: Topology graph
        img_path: Original image path
        cfg: Configuration object
        
    Returns:
        GeoJSON dictionary
    """
    # Build metadata
    metadata = {
        "source": Path(img_path).name,
        "generated": datetime.now().isoformat(),
        "generator": "PNG2SVG v1.0",
        "config": {
            "deskew": cfg.deskew,
            "constraint_solver": cfg.apply_constraint_solver,
            "yolo_symbols": cfg.use_yolo_symbols,
            "paddle_ocr": cfg.use_paddle_ocr,
            "confidence_tta": cfg.confidence_tta
        }
    }
    
    # Extract nodes
    nodes = export_nodes(graph)
    
    # Extract edges
    edges = export_edges(graph)
    
    # Extract relations
    relations = export_relations(graph)
    
    # Build statistics
    statistics = build_statistics(graph)
    
    # Assemble GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "metadata": metadata,
        "statistics": statistics,
        "features": [],
        "geometry": {
            "nodes": nodes,
            "edges": edges,
            "relations": relations
        }
    }
    
    # Add features (nodes as point features)
    for node_data in nodes:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [node_data["x"], node_data["y"]]
            },
            "properties": {
                "id": node_data["id"],
                "kind": node_data["kind"],
                "tag": node_data.get("tag", ""),
                "confidence": node_data.get("confidence", 1.0)
            }
        }
        geojson["features"].append(feature)
    
    # Add features (edges as line/curve features)
    for edge_data in edges:
        if edge_data["type"] == "line":
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        edge_data["endpoints"]["p1"],
                        edge_data["endpoints"]["p2"]
                    ]
                },
                "properties": {
                    "id": edge_data["id"],
                    "role": edge_data["role"],
                    "attributes": edge_data.get("attributes", {})
                }
            }
            geojson["features"].append(feature)
        elif edge_data["type"] in ["circle", "arc"]:
            # Represent circle/arc as a polygon approximation for GeoJSON
            coords = approximate_circle_arc(
                edge_data["center"],
                edge_data["radius"],
                edge_data.get("theta1", 0),
                edge_data.get("theta2", 360)
            )
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                },
                "properties": {
                    "id": edge_data["id"],
                    "type": edge_data["type"],
                    "center": edge_data["center"],
                    "radius": edge_data["radius"],
                    "attributes": edge_data.get("attributes", {})
                }
            }
            geojson["features"].append(feature)
    
    return geojson


def export_nodes(graph: Graph) -> List[Dict[str, Any]]:
    """
    Export nodes to dictionary format.
    
    Args:
        graph: Topology graph
        
    Returns:
        List of node dictionaries
    """
    nodes = []
    
    for node in graph.nodes:
        node_data = {
            "id": node.id,
            "x": round(node.x, 2),
            "y": round(node.y, 2),
            "kind": node.kind,
            "confidence": round(node.confidence, 3)
        }
        
        if node.tag:
            node_data["tag"] = node.tag
        
        # Add any additional attributes
        if hasattr(node, 'angle_value'):
            node_data["angle_value"] = node.angle_value
        
        nodes.append(node_data)
    
    return nodes


def export_edges(graph: Graph) -> List[Dict[str, Any]]:
    """
    Export edges to dictionary format.
    
    Args:
        graph: Topology graph
        
    Returns:
        List of edge dictionaries
    """
    edges = []
    
    for edge in graph.edges:
        edge_data = {
            "id": edge.id,
            "role": edge.role
        }
        
        if isinstance(edge.geom, LineSeg):
            edge_data.update({
                "type": "line",
                "endpoints": {
                    "p1": [round(edge.geom.p1[0], 2), round(edge.geom.p1[1], 2)],
                    "p2": [round(edge.geom.p2[0], 2), round(edge.geom.p2[1], 2)]
                },
                "attributes": {
                    "dashed": edge.geom.dashed,
                    "thickness": edge.geom.thickness,
                    "confidence": round(edge.geom.confidence, 3)
                }
            })
        
        elif isinstance(edge.geom, CircleArc):
            edge_data.update({
                "type": edge.geom.kind,  # "circle" or "arc"
                "center": [round(edge.geom.cx, 2), round(edge.geom.cy, 2)],
                "radius": round(edge.geom.r, 2),
                "theta1": round(edge.geom.theta1, 2),
                "theta2": round(edge.geom.theta2, 2),
                "attributes": {
                    "confidence": round(edge.geom.confidence, 3)
                }
            })
        
        # Add edge attributes
        if edge.attrs:
            for key, value in edge.attrs.items():
                if key not in ['dashed', 'thickness']:  # Avoid duplicates
                    edge_data["attributes"][key] = value
        
        edges.append(edge_data)
    
    return edges


def export_relations(graph: Graph) -> List[Dict[str, Any]]:
    """
    Export relationships to dictionary format.
    
    Args:
        graph: Topology graph
        
    Returns:
        List of relation dictionaries
    """
    relations = []
    
    for relation in graph.relations:
        rel_data = {
            "type": relation.type,
            "members": relation.members,
            "confidence": round(relation.conf, 3)
        }
        
        # Add relation attributes
        if relation.attrs:
            rel_data["attributes"] = relation.attrs
        
        relations.append(rel_data)
    
    return relations


def build_statistics(graph: Graph) -> Dict[str, Any]:
    """
    Build statistics about the geometric content.
    
    Args:
        graph: Topology graph
        
    Returns:
        Statistics dictionary
    """
    # Count different node types
    node_kinds = {}
    for node in graph.nodes:
        kind = node.kind
        node_kinds[kind] = node_kinds.get(kind, 0) + 1
    
    # Count different edge types
    line_count = sum(1 for e in graph.edges if isinstance(e.geom, LineSeg))
    circle_count = sum(1 for e in graph.edges if isinstance(e.geom, CircleArc) and e.geom.kind == "circle")
    arc_count = sum(1 for e in graph.edges if isinstance(e.geom, CircleArc) and e.geom.kind == "arc")
    dashed_count = sum(1 for e in graph.edges if isinstance(e.geom, LineSeg) and e.geom.dashed)
    
    # Count different relation types
    relation_types = {}
    for relation in graph.relations:
        rel_type = relation.type
        relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
    
    # Count labeled nodes
    labeled_count = sum(1 for node in graph.nodes if node.tag)
    
    # Build statistics
    stats = {
        "total_nodes": len(graph.nodes),
        "total_edges": len(graph.edges),
        "total_relations": len(graph.relations),
        "node_types": node_kinds,
        "edge_types": {
            "lines": line_count,
            "circles": circle_count,
            "arcs": arc_count,
            "dashed_lines": dashed_count
        },
        "relation_types": relation_types,
        "labeled_nodes": labeled_count,
        "average_confidence": calculate_average_confidence(graph)
    }
    
    return stats


def calculate_average_confidence(graph: Graph) -> float:
    """
    Calculate average confidence across all elements.
    
    Args:
        graph: Topology graph
        
    Returns:
        Average confidence value
    """
    confidences = []
    
    # Collect node confidences
    for node in graph.nodes:
        confidences.append(node.confidence)
    
    # Collect edge confidences
    for edge in graph.edges:
        if isinstance(edge.geom, LineSeg):
            confidences.append(edge.geom.confidence)
        elif isinstance(edge.geom, CircleArc):
            confidences.append(edge.geom.confidence)
    
    # Collect relation confidences
    for relation in graph.relations:
        confidences.append(relation.conf)
    
    if confidences:
        return round(sum(confidences) / len(confidences), 3)
    else:
        return 0.0


def approximate_circle_arc(center: List[float], radius: float, 
                           theta1: float, theta2: float, 
                           num_points: int = 32) -> List[List[float]]:
    """
    Approximate a circle or arc with line segments.
    
    Args:
        center: Center point [x, y]
        radius: Circle radius
        theta1: Start angle in degrees
        theta2: End angle in degrees
        num_points: Number of points for approximation
        
    Returns:
        List of coordinates for LineString
    """
    import math
    
    coords = []
    
    # Convert to radians
    theta1_rad = math.radians(theta1)
    theta2_rad = math.radians(theta2)
    
    # Handle full circle
    if abs(theta2 - theta1) >= 360:
        theta2_rad = theta1_rad + 2 * math.pi
    
    # Generate points
    for i in range(num_points + 1):
        t = i / num_points
        angle = theta1_rad + t * (theta2_rad - theta1_rad)
        
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        
        coords.append([round(x, 2), round(y, 2)])
    
    return coords


def export_semantic_annotations(graph: Graph) -> Dict[str, Any]:
    """
    Export high-level semantic annotations.
    
    Args:
        graph: Topology graph
        
    Returns:
        Dictionary of semantic annotations
    """
    annotations = {
        "triangles": find_triangles(graph),
        "quadrilaterals": find_quadrilaterals(graph),
        "right_angles": find_right_angles(graph),
        "parallel_pairs": find_parallel_pairs(graph),
        "perpendicular_pairs": find_perpendicular_pairs(graph),
        "equal_segments": find_equal_segments(graph),
        "angle_bisectors": find_angle_bisectors(graph),
        "medians": find_medians(graph),
        "altitudes": find_altitudes(graph)
    }
    
    return annotations


def find_triangles(graph: Graph) -> List[Dict[str, Any]]:
    """
    Find triangle structures in the graph.
    
    Args:
        graph: Topology graph
        
    Returns:
        List of triangle definitions
    """
    triangles = []
    
    # Use NetworkX to find triangles (3-cycles)
    if hasattr(graph, 'nx_graph'):
        import networkx as nx
        
        cycles = nx.cycle_basis(graph.nx_graph)
        for cycle in cycles:
            if len(cycle) == 3:
                # Get node tags if available
                vertices = []
                for node_id in cycle:
                    node = next((n for n in graph.nodes if n.id == node_id), None)
                    if node:
                        vertices.append({
                            "id": node.id,
                            "tag": node.tag if node.tag else None,
                            "position": [node.x, node.y]
                        })
                
                if len(vertices) == 3:
                    triangles.append({
                        "type": "triangle",
                        "vertices": vertices
                    })
    
    return triangles


def find_quadrilaterals(graph: Graph) -> List[Dict[str, Any]]:
    """
    Find quadrilateral structures in the graph.
    
    Args:
        graph: Topology graph
        
    Returns:
        List of quadrilateral definitions
    """
    quadrilaterals = []
    
    # Use NetworkX to find 4-cycles
    if hasattr(graph, 'nx_graph'):
        import networkx as nx
        
        cycles = nx.cycle_basis(graph.nx_graph)
        for cycle in cycles:
            if len(cycle) == 4:
                # Get node tags if available
                vertices = []
                for node_id in cycle:
                    node = next((n for n in graph.nodes if n.id == node_id), None)
                    if node:
                        vertices.append({
                            "id": node.id,
                            "tag": node.tag if node.tag else None,
                            "position": [node.x, node.y]
                        })
                
                if len(vertices) == 4:
                    quadrilaterals.append({
                        "type": "quadrilateral",
                        "vertices": vertices
                    })
    
    return quadrilaterals


def find_right_angles(graph: Graph) -> List[Dict[str, str]]:
    """
    Find right angles from perpendicular relations.
    
    Args:
        graph: Topology graph
        
    Returns:
        List of right angle pairs
    """
    right_angles = []
    
    for relation in graph.relations:
        if relation.type == "perpendicular" and relation.conf > 0.7:
            right_angles.append({
                "line1": relation.members[0] if len(relation.members) > 0 else None,
                "line2": relation.members[1] if len(relation.members) > 1 else None,
                "confidence": relation.conf
            })
    
    return right_angles


def find_parallel_pairs(graph: Graph) -> List[Dict[str, str]]:
    """
    Find parallel line pairs.
    
    Args:
        graph: Topology graph
        
    Returns:
        List of parallel pairs
    """
    parallel_pairs = []
    
    for relation in graph.relations:
        if relation.type == "parallel":
            parallel_pairs.append({
                "line1": relation.members[0] if len(relation.members) > 0 else None,
                "line2": relation.members[1] if len(relation.members) > 1 else None,
                "confidence": relation.conf
            })
    
    return parallel_pairs


def find_perpendicular_pairs(graph: Graph) -> List[Dict[str, str]]:
    """
    Find perpendicular line pairs.
    
    Args:
        graph: Topology graph
        
    Returns:
        List of perpendicular pairs
    """
    perp_pairs = []
    
    for relation in graph.relations:
        if relation.type == "perpendicular":
            perp_pairs.append({
                "line1": relation.members[0] if len(relation.members) > 0 else None,
                "line2": relation.members[1] if len(relation.members) > 1 else None,
                "confidence": relation.conf
            })
    
    return perp_pairs


def find_equal_segments(graph: Graph) -> List[List[str]]:
    """
    Find groups of equal length segments.
    
    Args:
        graph: Topology graph
        
    Returns:
        List of equal segment groups
    """
    equal_groups = []
    
    for relation in graph.relations:
        if relation.type == "equal_length":
            equal_groups.append(relation.members)
    
    return equal_groups


def find_angle_bisectors(graph: Graph) -> List[Dict[str, Any]]:
    """
    Find potential angle bisectors.
    
    Args:
        graph: Topology graph
        
    Returns:
        List of angle bisector candidates
    """
    # This would require more sophisticated analysis
    # Placeholder for future implementation
    return []


def find_medians(graph: Graph) -> List[Dict[str, Any]]:
    """
    Find triangle medians (lines from vertex to opposite midpoint).
    
    Args:
        graph: Topology graph
        
    Returns:
        List of median definitions
    """
    medians = []
    
    # Find nodes marked as midpoints
    midpoints = [n for n in graph.nodes if n.kind == "midpoint"]
    
    # For each midpoint, check if there's a line to a vertex
    # This would require more sophisticated analysis
    # Placeholder for future implementation
    
    return medians


def find_altitudes(graph: Graph) -> List[Dict[str, Any]]:
    """
    Find triangle altitudes (perpendiculars from vertex to opposite side).
    
    Args:
        graph: Topology graph
        
    Returns:
        List of altitude definitions
    """
    altitudes = []
    
    # Find nodes marked as perpendicular feet
    feet = [n for n in graph.nodes if n.kind == "foot"]
    
    # For each foot, trace back to find the altitude
    # This would require more sophisticated analysis
    # Placeholder for future implementation
    
    return altitudes