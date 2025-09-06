"""
Geometric constraint solving for PNG2SVG.

Optimizes node positions to satisfy geometric constraints while staying close
to detected positions. Uses soft constraints with SciPy optimization.
"""

import logging
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from .config import Config
from .topology import TopologyGraph, Node, Edge, Relation
from .detect_primitives import LineSeg, CircleArc


def solve(graph: TopologyGraph, config: Config) -> TopologyGraph:
    """
    Solve geometric constraints to optimize node positions.
    
    This function formulates and solves a constrained optimization problem where:
    - Variables: x, y coordinates of all nodes
    - Objective: Minimize deviation from detected positions + constraint violations
    - Constraints: Parallel/perpendicular lines, equal lengths, point-on relationships
    
    Args:
        graph: Topology graph with detected elements
        config: Configuration object
        
    Returns:
        TopologyGraph: Graph with optimized node positions
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check if SciPy is available
        import scipy.optimize
    except ImportError:
        logger.warning("SciPy not available, skipping constraint solving")
        return graph
    
    if not graph.nodes:
        logger.debug("No nodes to optimize")
        return graph
    
    logger.debug(f"Starting constraint optimization with {len(graph.nodes)} nodes and {len(graph.relations)} constraints")
    
    # Set up optimization problem
    solver = ConstraintSolver(graph, config)
    
    # Solve the optimization problem
    optimized_graph = solver.solve()
    
    if optimized_graph:
        logger.info("Constraint solving completed successfully")
        return optimized_graph
    else:
        logger.warning("Constraint solving failed, returning original graph")
        return graph


class ConstraintSolver:
    """Handles the constraint optimization process."""
    
    def __init__(self, graph: TopologyGraph, config: Config):
        self.graph = graph
        self.config = config
        self.node_indices = {node.id: i for i, node in enumerate(graph.nodes)}
        self.n_nodes = len(graph.nodes)
        self.n_vars = 2 * self.n_nodes  # x, y for each node
        
        # Weights for different terms in objective function
        self.w_data = 1.0          # Data fitting weight
        self.w_parallel = 0.5      # Parallel constraint weight
        self.w_perpendicular = 0.5 # Perpendicular constraint weight
        self.w_equal_length = 0.3  # Equal length constraint weight
        self.w_point_on = 0.7      # Point-on constraint weight
        
    def solve(self) -> Optional[TopologyGraph]:
        """Solve the constraint optimization problem."""
        try:
            import scipy.optimize
            
            # Initial guess: current node positions
            x0 = self._get_initial_positions()
            
            # Set up bounds (keep nodes reasonably close to original positions)
            bounds = self._get_position_bounds()
            
            # Solve optimization
            result = scipy.optimize.minimize(
                fun=self._objective_function,
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-6
                }
            )
            
            if result.success:
                # Update graph with optimized positions
                return self._update_graph_positions(result.x)
            else:
                logging.getLogger(__name__).warning(f"Optimization failed: {result.message}")
                return None
                
        except Exception as e:
            logging.getLogger(__name__).warning(f"Constraint solving failed: {e}")
            return None
    
    def _get_initial_positions(self) -> np.ndarray:
        """Get initial positions as flat array [x1, y1, x2, y2, ...]."""
        positions = np.zeros(self.n_vars)
        for i, node in enumerate(self.graph.nodes):
            positions[2*i] = node.x
            positions[2*i + 1] = node.y
        return positions
    
    def _get_position_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for node positions (stay within reasonable range of original)."""
        bounds = []
        max_movement = 50.0  # Maximum pixels a node can move
        
        for node in self.graph.nodes:
            bounds.append((node.x - max_movement, node.x + max_movement))  # x bounds
            bounds.append((node.y - max_movement, node.y + max_movement))  # y bounds
        
        return bounds
    
    def _objective_function(self, positions: np.ndarray) -> float:
        """
        Objective function to minimize.
        
        Args:
            positions: Flat array of node positions [x1, y1, x2, y2, ...]
            
        Returns:
            float: Total objective value (lower is better)
        """
        total_cost = 0.0
        
        # Data fitting term: keep nodes close to detected positions
        total_cost += self.w_data * self._data_fitting_cost(positions)
        
        # Constraint terms
        total_cost += self.w_parallel * self._parallel_constraint_cost(positions)
        total_cost += self.w_perpendicular * self._perpendicular_constraint_cost(positions)
        total_cost += self.w_equal_length * self._equal_length_constraint_cost(positions)
        total_cost += self.w_point_on * self._point_on_constraint_cost(positions)
        
        return total_cost
    
    def _data_fitting_cost(self, positions: np.ndarray) -> float:
        """Cost for deviation from detected positions."""
        cost = 0.0
        for i, node in enumerate(self.graph.nodes):
            dx = positions[2*i] - node.x
            dy = positions[2*i + 1] - node.y
            # Weight by node confidence
            cost += node.confidence * (dx**2 + dy**2)
        return cost
    
    def _parallel_constraint_cost(self, positions: np.ndarray) -> float:
        """Cost for violating parallel constraints."""
        cost = 0.0
        
        for relation in self.graph.relations:
            if relation.relation_type == "parallel" and len(relation.members) >= 2:
                edge1_id, edge2_id = relation.members[:2]
                edge1 = self.graph.get_edge_by_id(edge1_id)
                edge2 = self.graph.get_edge_by_id(edge2_id)
                
                if edge1 and edge2 and isinstance(edge1.geometry, LineSeg) and isinstance(edge2.geometry, LineSeg):
                    # Get node positions
                    n1_idx = self.node_indices.get(edge1.start_node_id)
                    n2_idx = self.node_indices.get(edge1.end_node_id)
                    n3_idx = self.node_indices.get(edge2.start_node_id)
                    n4_idx = self.node_indices.get(edge2.end_node_id)
                    
                    if all(idx is not None for idx in [n1_idx, n2_idx, n3_idx, n4_idx]):
                        # Calculate direction vectors
                        v1 = np.array([positions[2*n2_idx] - positions[2*n1_idx],
                                      positions[2*n2_idx + 1] - positions[2*n1_idx + 1]])
                        v2 = np.array([positions[2*n4_idx] - positions[2*n3_idx],
                                      positions[2*n4_idx + 1] - positions[2*n3_idx + 1]])
                        
                        # Normalize vectors
                        v1_norm = np.linalg.norm(v1)
                        v2_norm = np.linalg.norm(v2)
                        
                        if v1_norm > 1e-6 and v2_norm > 1e-6:
                            v1 = v1 / v1_norm
                            v2 = v2 / v2_norm
                            
                            # Cross product magnitude (should be 0 for parallel lines)
                            cross_product = abs(v1[0] * v2[1] - v1[1] * v2[0])
                            cost += relation.confidence * cross_product**2
        
        return cost
    
    def _perpendicular_constraint_cost(self, positions: np.ndarray) -> float:
        """Cost for violating perpendicular constraints."""
        cost = 0.0
        
        for relation in self.graph.relations:
            if relation.relation_type == "perpendicular" and len(relation.members) >= 2:
                edge1_id, edge2_id = relation.members[:2]
                edge1 = self.graph.get_edge_by_id(edge1_id)
                edge2 = self.graph.get_edge_by_id(edge2_id)
                
                if edge1 and edge2 and isinstance(edge1.geometry, LineSeg) and isinstance(edge2.geometry, LineSeg):
                    # Get node positions
                    n1_idx = self.node_indices.get(edge1.start_node_id)
                    n2_idx = self.node_indices.get(edge1.end_node_id)
                    n3_idx = self.node_indices.get(edge2.start_node_id)
                    n4_idx = self.node_indices.get(edge2.end_node_id)
                    
                    if all(idx is not None for idx in [n1_idx, n2_idx, n3_idx, n4_idx]):
                        # Calculate direction vectors
                        v1 = np.array([positions[2*n2_idx] - positions[2*n1_idx],
                                      positions[2*n2_idx + 1] - positions[2*n1_idx + 1]])
                        v2 = np.array([positions[2*n4_idx] - positions[2*n3_idx],
                                      positions[2*n4_idx + 1] - positions[2*n3_idx + 1]])
                        
                        # Normalize vectors
                        v1_norm = np.linalg.norm(v1)
                        v2_norm = np.linalg.norm(v2)
                        
                        if v1_norm > 1e-6 and v2_norm > 1e-6:
                            v1 = v1 / v1_norm
                            v2 = v2 / v2_norm
                            
                            # Dot product (should be 0 for perpendicular lines)
                            dot_product = np.dot(v1, v2)
                            cost += relation.confidence * dot_product**2
        
        return cost
    
    def _equal_length_constraint_cost(self, positions: np.ndarray) -> float:
        """Cost for violating equal length constraints."""
        cost = 0.0
        
        for relation in self.graph.relations:
            if relation.relation_type == "equal_length" and len(relation.members) >= 2:
                lengths = []
                valid_edges = []
                
                for edge_id in relation.members:
                    edge = self.graph.get_edge_by_id(edge_id)
                    if edge and isinstance(edge.geometry, LineSeg):
                        n1_idx = self.node_indices.get(edge.start_node_id)
                        n2_idx = self.node_indices.get(edge.end_node_id)
                        
                        if n1_idx is not None and n2_idx is not None:
                            dx = positions[2*n2_idx] - positions[2*n1_idx]
                            dy = positions[2*n2_idx + 1] - positions[2*n1_idx + 1]
                            length = math.sqrt(dx**2 + dy**2)
                            lengths.append(length)
                            valid_edges.append(edge)
                
                if len(lengths) >= 2:
                    # Penalize deviations from mean length
                    mean_length = np.mean(lengths)
                    for length in lengths:
                        deviation = (length - mean_length) / max(mean_length, 1.0)
                        cost += relation.confidence * deviation**2
        
        return cost
    
    def _point_on_constraint_cost(self, positions: np.ndarray) -> float:
        """Cost for violating point-on constraints."""
        cost = 0.0
        
        for relation in self.graph.relations:
            if relation.relation_type in ["point_on_line", "point_on_circle"] and len(relation.members) >= 2:
                node_id = relation.members[0]
                edge_id = relation.members[1]
                
                node = self.graph.get_node_by_id(node_id)
                edge = self.graph.get_edge_by_id(edge_id)
                
                if node and edge:
                    node_idx = self.node_indices.get(node_id)
                    if node_idx is not None:
                        point_x = positions[2*node_idx]
                        point_y = positions[2*node_idx + 1]
                        
                        if relation.relation_type == "point_on_line" and isinstance(edge.geometry, LineSeg):
                            # Calculate distance from point to line
                            n1_idx = self.node_indices.get(edge.start_node_id)
                            n2_idx = self.node_indices.get(edge.end_node_id)
                            
                            if n1_idx is not None and n2_idx is not None:
                                line_x1, line_y1 = positions[2*n1_idx], positions[2*n1_idx + 1]
                                line_x2, line_y2 = positions[2*n2_idx], positions[2*n2_idx + 1]
                                
                                distance = self._point_to_line_distance(
                                    (point_x, point_y), 
                                    (line_x1, line_y1), 
                                    (line_x2, line_y2)
                                )
                                cost += relation.confidence * distance**2
                        
                        elif relation.relation_type == "point_on_circle" and isinstance(edge.geometry, CircleArc):
                            # Calculate distance from point to circle
                            circle = edge.geometry
                            distance_to_center = math.sqrt(
                                (point_x - circle.cx)**2 + (point_y - circle.cy)**2
                            )
                            distance_error = abs(distance_to_center - circle.r)
                            cost += relation.confidence * distance_error**2
        
        return cost
    
    def _point_to_line_distance(self, point: Tuple[float, float], line_start: Tuple[float, float], line_end: Tuple[float, float]) -> float:
        """Calculate perpendicular distance from point to line segment."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return math.sqrt((x0 - closest_x)**2 + (y0 - closest_y)**2)
    
    def _update_graph_positions(self, optimized_positions: np.ndarray) -> TopologyGraph:
        """Update graph with optimized node positions."""
        # Create a copy of the graph
        optimized_graph = TopologyGraph(
            nodes=[],
            edges=self.graph.edges.copy(),
            relations=self.graph.relations.copy(),
            symbol_bindings=self.graph.symbol_bindings.copy(),
            text_bindings=self.graph.text_bindings.copy()
        )
        
        # Update node positions
        for i, node in enumerate(self.graph.nodes):
            optimized_node = Node(
                x=float(optimized_positions[2*i]),
                y=float(optimized_positions[2*i + 1]),
                node_type=node.node_type,
                label=node.label,
                semantic_role=node.semantic_role,
                confidence=node.confidence,
                properties=node.properties.copy(),
                id=node.id
            )
            optimized_graph.nodes.append(optimized_node)
        
        # Update edge geometries to reflect new node positions
        for edge in optimized_graph.edges:
            if isinstance(edge.geometry, LineSeg):
                start_node = optimized_graph.get_node_by_id(edge.start_node_id)
                end_node = optimized_graph.get_node_by_id(edge.end_node_id)
                
                if start_node and end_node:
                    # Create new LineSeg with updated positions
                    optimized_line = LineSeg(
                        p1=(start_node.x, start_node.y),
                        p2=(end_node.x, end_node.y),
                        dashed=edge.geometry.dashed,
                        thickness=edge.geometry.thickness,
                        role=edge.geometry.role,
                        confidence=edge.geometry.confidence,
                        id=edge.geometry.id
                    )
                    edge.geometry = optimized_line
        
        return optimized_graph


def evaluate_constraint_satisfaction(graph: TopologyGraph) -> Dict[str, float]:
    """
    Evaluate how well the graph satisfies its constraints.
    
    Args:
        graph: Topology graph to evaluate
        
    Returns:
        Dict[str, float]: Constraint satisfaction metrics
    """
    metrics = {
        'parallel_error': 0.0,
        'perpendicular_error': 0.0,
        'equal_length_error': 0.0,
        'point_on_error': 0.0,
        'total_relations': len(graph.relations)
    }
    
    parallel_errors = []
    perpendicular_errors = []
    equal_length_errors = []
    point_on_errors = []
    
    for relation in graph.relations:
        if relation.relation_type == "parallel" and len(relation.members) >= 2:
            error = _evaluate_parallel_constraint(graph, relation)
            if error is not None:
                parallel_errors.append(error)
        
        elif relation.relation_type == "perpendicular" and len(relation.members) >= 2:
            error = _evaluate_perpendicular_constraint(graph, relation)
            if error is not None:
                perpendicular_errors.append(error)
        
        elif relation.relation_type == "equal_length" and len(relation.members) >= 2:
            error = _evaluate_equal_length_constraint(graph, relation)
            if error is not None:
                equal_length_errors.append(error)
        
        elif relation.relation_type in ["point_on_line", "point_on_circle"]:
            error = _evaluate_point_on_constraint(graph, relation)
            if error is not None:
                point_on_errors.append(error)
    
    # Calculate average errors
    if parallel_errors:
        metrics['parallel_error'] = np.mean(parallel_errors)
    if perpendicular_errors:
        metrics['perpendicular_error'] = np.mean(perpendicular_errors)
    if equal_length_errors:
        metrics['equal_length_error'] = np.mean(equal_length_errors)
    if point_on_errors:
        metrics['point_on_error'] = np.mean(point_on_errors)
    
    return metrics


def _evaluate_parallel_constraint(graph: TopologyGraph, relation: Relation) -> Optional[float]:
    """Evaluate parallel constraint satisfaction."""
    edge1_id, edge2_id = relation.members[:2]
    edge1 = graph.get_edge_by_id(edge1_id)
    edge2 = graph.get_edge_by_id(edge2_id)
    
    if edge1 and edge2 and isinstance(edge1.geometry, LineSeg) and isinstance(edge2.geometry, LineSeg):
        angle1 = edge1.geometry.angle
        angle2 = edge2.geometry.angle
        
        angle_diff = abs(angle1 - angle2)
        angle_diff = min(angle_diff, math.pi - angle_diff)  # Handle wraparound
        
        return math.degrees(angle_diff)  # Return error in degrees
    
    return None


def _evaluate_perpendicular_constraint(graph: TopologyGraph, relation: Relation) -> Optional[float]:
    """Evaluate perpendicular constraint satisfaction."""
    edge1_id, edge2_id = relation.members[:2]
    edge1 = graph.get_edge_by_id(edge1_id)
    edge2 = graph.get_edge_by_id(edge2_id)
    
    if edge1 and edge2 and isinstance(edge1.geometry, LineSeg) and isinstance(edge2.geometry, LineSeg):
        angle1 = edge1.geometry.angle
        angle2 = edge2.geometry.angle
        
        angle_diff = abs(angle1 - angle2)
        perp_diff = min(abs(angle_diff - math.pi/2), abs(angle_diff - 3*math.pi/2))
        
        return math.degrees(perp_diff)  # Return error in degrees
    
    return None


def _evaluate_equal_length_constraint(graph: TopologyGraph, relation: Relation) -> Optional[float]:
    """Evaluate equal length constraint satisfaction."""
    lengths = []
    for edge_id in relation.members:
        edge = graph.get_edge_by_id(edge_id)
        if edge and isinstance(edge.geometry, LineSeg):
            lengths.append(edge.geometry.length)
    
    if len(lengths) >= 2:
        mean_length = np.mean(lengths)
        if mean_length > 0:
            relative_errors = [abs(l - mean_length) / mean_length for l in lengths]
            return np.mean(relative_errors)  # Return relative error
    
    return None


def _evaluate_point_on_constraint(graph: TopologyGraph, relation: Relation) -> Optional[float]:
    """Evaluate point-on constraint satisfaction."""
    if len(relation.members) >= 2:
        node_id = relation.members[0]
        edge_id = relation.members[1]
        
        node = graph.get_node_by_id(node_id)
        edge = graph.get_edge_by_id(edge_id)
        
        if node and edge:
            if relation.relation_type == "point_on_line" and isinstance(edge.geometry, LineSeg):
                line = edge.geometry
                distance = ConstraintSolver(graph, None)._point_to_line_distance(
                    (node.x, node.y), line.p1, line.p2
                )
                return distance  # Return pixel distance
            
            elif relation.relation_type == "point_on_circle" and isinstance(edge.geometry, CircleArc):
                circle = edge.geometry
                distance_to_center = math.sqrt(
                    (node.x - circle.cx)**2 + (node.y - circle.cy)**2
                )
                return abs(distance_to_center - circle.r)  # Return radial error
    
    return None