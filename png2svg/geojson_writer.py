"""
GeoJSON output generation for PNG2SVG.

Exports structured geometric data as GeoJSON FeatureCollection with semantic
properties, relationships, and metadata for programmatic analysis.
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .config import Config
from .topology import TopologyGraph, Node, Edge, Relation
from .detect_primitives import LineSeg, CircleArc


def write(image_path: str, graph: TopologyGraph, config: Config) -> str:
    """
    Generate GeoJSON file from topology graph.
    
    Args:
        image_path: Original image path (for output naming)
        graph: Topology graph with geometric elements
        config: Configuration object
        
    Returns:
        str: Path to generated GeoJSON file
    """
    logger = logging.getLogger(__name__)
    
    # Determine output path
    base_name = Path(image_path).stem
    geojson_path = Path(config.output_dir) / f"{base_name}.geo.json"
    
    # Create GeoJSON structure
    geojson_data = _create_geojson_structure(image_path, graph, config)
    
    # Write to file
    with open(geojson_path, 'w', encoding='utf-8') as f:
        json.dump(geojson_data, f, ensure_ascii=False, indent=2)
    
    logger.debug(f"GeoJSON saved to: {geojson_path}")
    return str(geojson_path)


def _create_geojson_structure(image_path: str, graph: TopologyGraph, config: Config) -> Dict[str, Any]:
    """Create the complete GeoJSON structure."""
    
    # Create feature collections for different element types
    features = []
    
    # Add node features (points)
    features.extend(_create_node_features(graph))
    
    # Add edge features (lines, circles)
    features.extend(_create_edge_features(graph))
    
    # Add relationship features (virtual geometries)
    features.extend(_create_relationship_features(graph))
    
    # Create main GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "metadata": _create_metadata(image_path, graph, config),
        "features": features
    }
    
    return geojson


def _create_metadata(image_path: str, graph: TopologyGraph, config: Config) -> Dict[str, Any]:
    """Create metadata section for GeoJSON."""
    
    return {
        "version": "1.0",
        "generator": "PNG2SVG",
        "generated_at": datetime.now().isoformat(),
        "source_image": Path(image_path).name,
        "coordinate_system": "image_pixels",
        "statistics": {
            "total_nodes": len(graph.nodes),
            "total_edges": len(graph.edges),
            "total_relations": len(graph.relations),
            "node_types": _count_by_attribute(graph.nodes, 'node_type'),
            "edge_types": _count_edge_types(graph.edges),
            "relation_types": _count_by_attribute(graph.relations, 'relation_type')
        },
        "bindings": {
            "symbol_bindings": len(graph.symbol_bindings),
            "text_bindings": len(graph.text_bindings)
        },
        "config": {
            "constraint_solver_used": config.apply_constraint_solver,
            "yolo_symbols_used": config.use_yolo_symbols,
            "paddle_ocr_used": config.use_paddle_ocr,
            "tta_used": config.confidence_tta
        }
    }


def _create_node_features(graph: TopologyGraph) -> List[Dict[str, Any]]:
    """Create GeoJSON features for nodes (points)."""
    features = []
    
    for node in graph.nodes:
        feature = {
            "type": "Feature",
            "id": node.id,
            "geometry": {
                "type": "Point",
                "coordinates": [node.x, node.y]
            },
            "properties": {
                "element_type": "node",
                "node_type": node.node_type,
                "label": node.label,
                "semantic_role": node.semantic_role,
                "confidence": node.confidence,
                **node.properties
            }
        }
        
        features.append(feature)
    
    return features


def _create_edge_features(graph: TopologyGraph) -> List[Dict[str, Any]]:
    """Create GeoJSON features for edges (lines, circles, arcs)."""
    features = []
    
    for edge in graph.edges:
        if isinstance(edge.geometry, LineSeg):
            feature = _create_line_feature(edge, graph)
        elif isinstance(edge.geometry, CircleArc):
            feature = _create_circle_arc_feature(edge, graph)
        else:
            continue  # Skip unknown geometry types
        
        if feature:
            features.append(feature)
    
    return features


def _create_line_feature(edge: Edge, graph: TopologyGraph) -> Dict[str, Any]:
    """Create GeoJSON feature for a line segment."""
    line_geom = edge.geometry
    
    # Get node coordinates if available
    start_node = graph.get_node_by_id(edge.start_node_id) if edge.start_node_id else None
    end_node = graph.get_node_by_id(edge.end_node_id) if edge.end_node_id else None
    
    if start_node and end_node:
        coordinates = [[start_node.x, start_node.y], [end_node.x, end_node.y]]
    else:
        coordinates = [list(line_geom.p1), list(line_geom.p2)]
    
    feature = {
        "type": "Feature",
        "id": edge.id,
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates
        },
        "properties": {
            "element_type": "line_segment",
            "edge_type": edge.edge_type,
            "dashed": line_geom.dashed,
            "thickness": line_geom.thickness,
            "role": line_geom.role,
            "confidence": line_geom.confidence,
            "length": line_geom.length,
            "angle_degrees": math.degrees(line_geom.angle),
            "start_node_id": edge.start_node_id,
            "end_node_id": edge.end_node_id,
            **edge.properties
        }
    }
    
    return feature


def _create_circle_arc_feature(edge: Edge, graph: TopologyGraph) -> Dict[str, Any]:
    """Create GeoJSON feature for a circle or arc."""
    circle_geom = edge.geometry
    
    if circle_geom.is_full_circle:
        # Create a polygon approximation of the circle
        coordinates = _create_circle_polygon(
            circle_geom.cx, circle_geom.cy, circle_geom.r
        )
        geometry_type = "Polygon"
    else:
        # Create a LineString approximation of the arc
        coordinates = _create_arc_linestring(
            circle_geom.cx, circle_geom.cy, circle_geom.r,
            circle_geom.theta1, circle_geom.theta2
        )
        geometry_type = "LineString"
    
    feature = {
        "type": "Feature",
        "id": edge.id,
        "geometry": {
            "type": geometry_type,
            "coordinates": coordinates
        },
        "properties": {
            "element_type": "circle" if circle_geom.is_full_circle else "arc",
            "edge_type": edge.edge_type,
            "center": [circle_geom.cx, circle_geom.cy],
            "radius": circle_geom.r,
            "confidence": circle_geom.confidence,
            "center_node_id": edge.start_node_id,
            **edge.properties
        }
    }
    
    # Add arc-specific properties
    if not circle_geom.is_full_circle:
        feature["properties"].update({
            "start_angle_degrees": math.degrees(circle_geom.theta1),
            "end_angle_degrees": math.degrees(circle_geom.theta2),
            "arc_length": circle_geom.r * abs(circle_geom.theta2 - circle_geom.theta1)
        })
    
    return feature


def _create_relationship_features(graph: TopologyGraph) -> List[Dict[str, Any]]:
    """Create GeoJSON features for relationships (as virtual geometries)."""
    features = []
    
    for relation in graph.relations:
        # Create a point feature at the centroid of participating elements
        centroid = _calculate_relation_centroid(relation, graph)
        
        if centroid:
            feature = {
                "type": "Feature",
                "id": relation.id,
                "geometry": {
                    "type": "Point",
                    "coordinates": list(centroid)
                },
                "properties": {
                    "element_type": "relationship",
                    "relation_type": relation.relation_type,
                    "member_ids": relation.members,
                    "member_count": len(relation.members),
                    "confidence": relation.confidence,
                    **relation.properties
                }
            }
            
            features.append(feature)
    
    return features


def _calculate_relation_centroid(relation: Relation, graph: TopologyGraph) -> Optional[Tuple[float, float]]:
    """Calculate the centroid of elements participating in a relationship."""
    points = []
    
    for member_id in relation.members:
        # Try as node first
        node = graph.get_node_by_id(member_id)
        if node:
            points.append((node.x, node.y))
            continue
        
        # Try as edge
        edge = graph.get_edge_by_id(member_id)
        if edge:
            if isinstance(edge.geometry, LineSeg):
                # Use line midpoint
                line = edge.geometry
                mid_x = (line.p1[0] + line.p2[0]) / 2
                mid_y = (line.p1[1] + line.p2[1]) / 2
                points.append((mid_x, mid_y))
            elif isinstance(edge.geometry, CircleArc):
                # Use circle center
                circle = edge.geometry
                points.append((circle.cx, circle.cy))
    
    if not points:
        return None
    
    # Calculate centroid
    centroid_x = sum(p[0] for p in points) / len(points)
    centroid_y = sum(p[1] for p in points) / len(points)
    
    return (centroid_x, centroid_y)


def _create_circle_polygon(cx: float, cy: float, r: float, num_points: int = 32) -> List[List[List[float]]]:
    """Create a polygon approximation of a circle for GeoJSON."""
    coordinates = []
    
    for i in range(num_points + 1):  # +1 to close the polygon
        angle = 2 * math.pi * i / num_points
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        coordinates.append([x, y])
    
    return [coordinates]  # GeoJSON Polygon format requires array of rings


def _create_arc_linestring(cx: float, cy: float, r: float, theta1: float, theta2: float, num_points: int = 16) -> List[List[float]]:
    """Create a LineString approximation of an arc for GeoJSON."""
    coordinates = []
    
    # Handle angle wraparound
    if theta2 < theta1:
        theta2 += 2 * math.pi
    
    angle_step = (theta2 - theta1) / num_points
    
    for i in range(num_points + 1):
        angle = theta1 + i * angle_step
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        coordinates.append([x, y])
    
    return coordinates


def _count_by_attribute(items: List, attribute: str) -> Dict[str, int]:
    """Count items by attribute value."""
    counts = {}
    for item in items:
        value = getattr(item, attribute, 'unknown')
        counts[value] = counts.get(value, 0) + 1
    return counts


def _count_edge_types(edges: List[Edge]) -> Dict[str, int]:
    """Count edges by geometry type."""
    counts = {}
    for edge in edges:
        geom_type = type(edge.geometry).__name__
        counts[geom_type] = counts.get(geom_type, 0) + 1
    return counts


def create_geojson_schema() -> Dict[str, Any]:
    """Create a JSON schema for the GeoJSON output format."""
    
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "PNG2SVG GeoJSON Output",
        "description": "Structured geometric data exported from PNG mathematical diagrams",
        "type": "object",
        "required": ["type", "features"],
        "properties": {
            "type": {
                "type": "string",
                "const": "FeatureCollection"
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "version": {"type": "string"},
                    "generator": {"type": "string"},
                    "generated_at": {"type": "string", "format": "date-time"},
                    "source_image": {"type": "string"},
                    "coordinate_system": {"type": "string"},
                    "statistics": {"type": "object"},
                    "bindings": {"type": "object"},
                    "config": {"type": "object"}
                }
            },
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["type", "geometry", "properties"],
                    "properties": {
                        "type": {"const": "Feature"},
                        "id": {"type": "string"},
                        "geometry": {
                            "type": "object",
                            "oneOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": {"const": "Point"},
                                        "coordinates": {
                                            "type": "array",
                                            "minItems": 2,
                                            "maxItems": 2,
                                            "items": {"type": "number"}
                                        }
                                    }
                                },
                                {
                                    "type": "object", 
                                    "properties": {
                                        "type": {"const": "LineString"},
                                        "coordinates": {
                                            "type": "array",
                                            "minItems": 2,
                                            "items": {
                                                "type": "array",
                                                "minItems": 2,
                                                "maxItems": 2,
                                                "items": {"type": "number"}
                                            }
                                        }
                                    }
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": {"const": "Polygon"},
                                        "coordinates": {
                                            "type": "array",
                                            "items": {
                                                "type": "array",
                                                "minItems": 4,
                                                "items": {
                                                    "type": "array",
                                                    "minItems": 2,
                                                    "maxItems": 2,
                                                    "items": {"type": "number"}
                                                }
                                            }
                                        }
                                    }
                                }
                            ]
                        },
                        "properties": {
                            "type": "object",
                            "required": ["element_type"],
                            "properties": {
                                "element_type": {
                                    "type": "string",
                                    "enum": ["node", "line_segment", "circle", "arc", "relationship"]
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return schema


def validate_geojson_output(geojson_path: str) -> bool:
    """Validate that the generated GeoJSON is well-formed."""
    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Basic structure validation
        if not isinstance(data, dict):
            return False
        
        if data.get('type') != 'FeatureCollection':
            return False
        
        if 'features' not in data or not isinstance(data['features'], list):
            return False
        
        # Validate features
        for feature in data['features']:
            if not isinstance(feature, dict):
                return False
            
            if feature.get('type') != 'Feature':
                return False
            
            if 'geometry' not in feature or 'properties' not in feature:
                return False
            
            # Validate geometry
            geometry = feature['geometry']
            if not isinstance(geometry, dict) or 'type' not in geometry:
                return False
            
            geom_type = geometry['type']
            if geom_type not in ['Point', 'LineString', 'Polygon']:
                return False
            
            if 'coordinates' not in geometry:
                return False
        
        return True
        
    except Exception:
        return False


def export_geojson_summary(geojson_path: str) -> Dict[str, Any]:
    """Extract summary statistics from generated GeoJSON."""
    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        summary = {
            'total_features': len(data.get('features', [])),
            'feature_types': {},
            'geometry_types': {},
            'has_metadata': 'metadata' in data
        }
        
        # Count feature types
        for feature in data.get('features', []):
            properties = feature.get('properties', {})
            element_type = properties.get('element_type', 'unknown')
            summary['feature_types'][element_type] = summary['feature_types'].get(element_type, 0) + 1
            
            geometry = feature.get('geometry', {})
            geom_type = geometry.get('type', 'unknown')
            summary['geometry_types'][geom_type] = summary['geometry_types'].get(geom_type, 0) + 1
        
        # Extract metadata statistics if available
        if 'metadata' in data:
            metadata = data['metadata']
            if 'statistics' in metadata:
                summary['metadata_statistics'] = metadata['statistics']
        
        return summary
        
    except Exception as e:
        return {'error': str(e)}