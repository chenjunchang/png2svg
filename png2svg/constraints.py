"""
Constraint solver module for PNG2SVG system.
Optimizes geometric positions using constraint-based optimization.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math
from scipy.optimize import least_squares

from .config import Config
from .topology import Graph, Node, Edge, Relation, LineSeg, CircleArc


@dataclass
class ConstraintProblem:
    """Constraint optimization problem definition."""
    variables: np.ndarray          # Flattened array of (x, y) coordinates
    node_indices: Dict[str, int]   # Map node ID to variable index
    constraints: List['Constraint'] # List of constraints
    observations: np.ndarray        # Original observed positions
    
    
class Constraint:
    """Base class for geometric constraints."""
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def residual(self, variables: np.ndarray, indices: Dict[str, int]) -> float:
        """Calculate residual (error) for this constraint."""
        raise NotImplementedError
        

class ParallelConstraint(Constraint):
    """Constraint for parallel lines."""
    
    def __init__(self, line1_nodes: Tuple[str, str], line2_nodes: Tuple[str, str], weight: float = 1.0):
        super().__init__(weight)
        self.line1_nodes = line1_nodes
        self.line2_nodes = line2_nodes
    
    def residual(self, variables: np.ndarray, indices: Dict[str, int]) -> float:
        """Calculate angle difference between lines."""
        # Get line 1 points
        idx1_start = indices[self.line1_nodes[0]]
        idx1_end = indices[self.line1_nodes[1]]
        p1_start = variables[idx1_start:idx1_start+2]
        p1_end = variables[idx1_end:idx1_end+2]
        
        # Get line 2 points
        idx2_start = indices[self.line2_nodes[0]]
        idx2_end = indices[self.line2_nodes[1]]
        p2_start = variables[idx2_start:idx2_start+2]
        p2_end = variables[idx2_end:idx2_end+2]
        
        # Calculate direction vectors
        v1 = p1_end - p1_start
        v2 = p2_end - p2_start
        
        # Normalize
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        v1 = v1 / norm1
        v2 = v2 / norm2
        
        # Cross product for parallelism (should be 0)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        return self.weight * cross


class PerpendicularConstraint(Constraint):
    """Constraint for perpendicular lines."""
    
    def __init__(self, line1_nodes: Tuple[str, str], line2_nodes: Tuple[str, str], weight: float = 1.0):
        super().__init__(weight)
        self.line1_nodes = line1_nodes
        self.line2_nodes = line2_nodes
    
    def residual(self, variables: np.ndarray, indices: Dict[str, int]) -> float:
        """Calculate dot product of line directions (should be 0 for perpendicular)."""
        # Get line 1 points
        idx1_start = indices[self.line1_nodes[0]]
        idx1_end = indices[self.line1_nodes[1]]
        p1_start = variables[idx1_start:idx1_start+2]
        p1_end = variables[idx1_end:idx1_end+2]
        
        # Get line 2 points
        idx2_start = indices[self.line2_nodes[0]]
        idx2_end = indices[self.line2_nodes[1]]
        p2_start = variables[idx2_start:idx2_start+2]
        p2_end = variables[idx2_end:idx2_end+2]
        
        # Calculate direction vectors
        v1 = p1_end - p1_start
        v2 = p2_end - p2_start
        
        # Normalize
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        v1 = v1 / norm1
        v2 = v2 / norm2
        
        # Dot product for perpendicularity (should be 0)
        dot = np.dot(v1, v2)
        
        return self.weight * dot


class CollinearConstraint(Constraint):
    """Constraint for collinear points."""
    
    def __init__(self, point_ids: List[str], weight: float = 1.0):
        super().__init__(weight)
        self.point_ids = point_ids
    
    def residual(self, variables: np.ndarray, indices: Dict[str, int]) -> float:
        """Calculate deviation from collinearity."""
        if len(self.point_ids) < 3:
            return 0.0
        
        # Get first two points to define line
        idx1 = indices[self.point_ids[0]]
        idx2 = indices[self.point_ids[1]]
        p1 = variables[idx1:idx1+2]
        p2 = variables[idx2:idx2+2]
        
        # Line direction
        v = p2 - p1
        norm = np.linalg.norm(v)
        
        if norm < 1e-6:
            return 0.0
        
        v = v / norm
        
        # Normal to line
        n = np.array([-v[1], v[0]])
        
        # Sum of distances from other points to line
        total_dist = 0.0
        for i in range(2, len(self.point_ids)):
            idx = indices[self.point_ids[i]]
            p = variables[idx:idx+2]
            
            # Distance to line
            dist = abs(np.dot(p - p1, n))
            total_dist += dist
        
        return self.weight * total_dist


class EqualLengthConstraint(Constraint):
    """Constraint for equal length segments."""
    
    def __init__(self, seg1_nodes: Tuple[str, str], seg2_nodes: Tuple[str, str], weight: float = 1.0):
        super().__init__(weight)
        self.seg1_nodes = seg1_nodes
        self.seg2_nodes = seg2_nodes
    
    def residual(self, variables: np.ndarray, indices: Dict[str, int]) -> float:
        """Calculate length difference."""
        # Get segment 1 points
        idx1_start = indices[self.seg1_nodes[0]]
        idx1_end = indices[self.seg1_nodes[1]]
        p1_start = variables[idx1_start:idx1_start+2]
        p1_end = variables[idx1_end:idx1_end+2]
        
        # Get segment 2 points
        idx2_start = indices[self.seg2_nodes[0]]
        idx2_end = indices[self.seg2_nodes[1]]
        p2_start = variables[idx2_start:idx2_start+2]
        p2_end = variables[idx2_end:idx2_end+2]
        
        # Calculate lengths
        len1 = np.linalg.norm(p1_end - p1_start)
        len2 = np.linalg.norm(p2_end - p2_start)
        
        # Return difference
        return self.weight * (len1 - len2)


class PointOnCircleConstraint(Constraint):
    """Constraint for point on circle."""
    
    def __init__(self, point_id: str, center_id: str, radius: float, weight: float = 1.0):
        super().__init__(weight)
        self.point_id = point_id
        self.center_id = center_id
        self.radius = radius
    
    def residual(self, variables: np.ndarray, indices: Dict[str, int]) -> float:
        """Calculate distance from point to circle."""
        idx_point = indices[self.point_id]
        idx_center = indices[self.center_id]
        
        point = variables[idx_point:idx_point+2]
        center = variables[idx_center:idx_center+2]
        
        # Distance to center
        dist = np.linalg.norm(point - center)
        
        # Deviation from radius
        return self.weight * (dist - self.radius)


class AngleConstraint(Constraint):
    """Constraint for fixed angle between lines."""
    
    def __init__(self, vertex_id: str, point1_id: str, point2_id: str, 
                 angle_degrees: float, weight: float = 1.0):
        super().__init__(weight)
        self.vertex_id = vertex_id
        self.point1_id = point1_id
        self.point2_id = point2_id
        self.angle_rad = math.radians(angle_degrees)
    
    def residual(self, variables: np.ndarray, indices: Dict[str, int]) -> float:
        """Calculate angle deviation."""
        idx_vertex = indices[self.vertex_id]
        idx_p1 = indices[self.point1_id]
        idx_p2 = indices[self.point2_id]
        
        vertex = variables[idx_vertex:idx_vertex+2]
        p1 = variables[idx_p1:idx_p1+2]
        p2 = variables[idx_p2:idx_p2+2]
        
        # Vectors from vertex
        v1 = p1 - vertex
        v2 = p2 - vertex
        
        # Normalize
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        v1 = v1 / norm1
        v2 = v2 / norm2
        
        # Calculate angle
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = math.acos(cos_angle)
        
        # Return difference from target angle
        return self.weight * (angle - self.angle_rad)


def solve(graph: Graph, cfg: Config) -> Graph:
    """
    Apply constraint solving to optimize geometry.
    
    - Build variables: key node coordinates
    - Objective: min Σ(data_term) + λΣ(soft_constraints)
      data_term: distance from observed geometry
      constraints: parallel, perpendicular, collinear, equal_length, angles
    - Use scipy.optimize.least_squares with Huber loss
    - Write optimized coordinates back to nodes and edges
    
    Args:
        graph: Input topology graph
        cfg: Configuration object
        
    Returns:
        Graph with optimized geometry
    """
    logger = logging.getLogger('png2svg.constraints')
    
    if not cfg.apply_constraint_solver:
        logger.info("Constraint solver disabled")
        return graph
    
    logger.info("Starting constraint optimization")
    
    # Build constraint problem
    problem = build_constraint_problem(graph, cfg)
    
    if not problem.constraints:
        logger.info("No constraints found, skipping optimization")
        return graph
    
    logger.info(f"Optimizing {len(problem.node_indices)} nodes with {len(problem.constraints)} constraints")
    
    # Run optimization
    result = optimize_constraints(problem, cfg)
    
    # Apply optimized positions back to graph
    apply_optimized_positions(graph, result, problem.node_indices)
    
    logger.info("Constraint optimization complete")
    
    return graph


def build_constraint_problem(graph: Graph, cfg: Config) -> ConstraintProblem:
    """
    Build constraint optimization problem from graph.
    
    Args:
        graph: Topology graph
        cfg: Configuration object
        
    Returns:
        ConstraintProblem object
    """
    # Build variable indices for nodes
    node_indices = {}
    variables = []
    observations = []
    
    for i, node in enumerate(graph.nodes):
        node_indices[node.id] = i * 2
        variables.extend([node.x, node.y])
        observations.extend([node.x, node.y])
    
    variables = np.array(variables)
    observations = np.array(observations)
    
    # Build constraints from relationships
    constraints = []
    
    for relation in graph.relations:
        if relation.type == "parallel":
            constraints.extend(build_parallel_constraints(graph, relation, cfg))
        elif relation.type == "perpendicular":
            constraints.extend(build_perpendicular_constraints(graph, relation, cfg))
        elif relation.type == "equal_length":
            constraints.extend(build_equal_length_constraints(graph, relation, cfg))
        elif relation.type == "point_on_circle":
            constraints.extend(build_point_on_circle_constraints(graph, relation, cfg))
    
    return ConstraintProblem(
        variables=variables,
        node_indices=node_indices,
        constraints=constraints,
        observations=observations
    )


def build_parallel_constraints(graph: Graph, relation: Relation, cfg: Config) -> List[Constraint]:
    """Build parallel constraints from relation."""
    constraints = []
    
    if len(relation.members) >= 2:
        # Find nodes for each line
        line1_nodes = find_line_nodes(graph, relation.members[0])
        line2_nodes = find_line_nodes(graph, relation.members[1])
        
        if line1_nodes and line2_nodes:
            weight = cfg.algorithms.constraints.lambda_parallel * relation.conf
            constraints.append(ParallelConstraint(line1_nodes, line2_nodes, weight))
    
    return constraints


def build_perpendicular_constraints(graph: Graph, relation: Relation, cfg: Config) -> List[Constraint]:
    """Build perpendicular constraints from relation."""
    constraints = []
    
    if len(relation.members) >= 2:
        # Find nodes for each line
        line1_nodes = find_line_nodes(graph, relation.members[0])
        line2_nodes = find_line_nodes(graph, relation.members[1])
        
        if line1_nodes and line2_nodes:
            weight = cfg.algorithms.constraints.lambda_perpendicular * relation.conf
            constraints.append(PerpendicularConstraint(line1_nodes, line2_nodes, weight))
    
    return constraints


def build_equal_length_constraints(graph: Graph, relation: Relation, cfg: Config) -> List[Constraint]:
    """Build equal length constraints from relation."""
    constraints = []
    
    if len(relation.members) >= 2:
        # Create pairwise constraints
        for i in range(len(relation.members) - 1):
            line1_nodes = find_line_nodes(graph, relation.members[i])
            line2_nodes = find_line_nodes(graph, relation.members[i+1])
            
            if line1_nodes and line2_nodes:
                weight = cfg.algorithms.constraints.lambda_equal_length * relation.conf
                constraints.append(EqualLengthConstraint(line1_nodes, line2_nodes, weight))
    
    return constraints


def build_point_on_circle_constraints(graph: Graph, relation: Relation, cfg: Config) -> List[Constraint]:
    """Build point-on-circle constraints from relation."""
    constraints = []
    
    if len(relation.members) >= 2:
        point_id = relation.members[0]
        circle_id = relation.members[1]
        
        # Find circle parameters
        for edge in graph.edges:
            if edge.id == circle_id and isinstance(edge.geom, CircleArc):
                # Find center node
                center_node = find_nearest_node_to_point(graph.nodes, (edge.geom.cx, edge.geom.cy))
                
                if center_node:
                    weight = cfg.algorithms.constraints.lambda_point_on_circle * relation.conf
                    constraints.append(PointOnCircleConstraint(
                        point_id, center_node.id, edge.geom.r, weight
                    ))
                break
    
    return constraints


def find_line_nodes(graph: Graph, line_id: str) -> Optional[Tuple[str, str]]:
    """Find endpoint nodes for a line edge."""
    for edge in graph.edges:
        if edge.id == line_id and isinstance(edge.geom, LineSeg):
            # Find nodes at endpoints
            start_node = find_nearest_node_to_point(graph.nodes, edge.geom.p1)
            end_node = find_nearest_node_to_point(graph.nodes, edge.geom.p2)
            
            if start_node and end_node:
                return (start_node.id, end_node.id)
    
    return None


def find_nearest_node_to_point(nodes: List[Node], point: Tuple[float, float]) -> Optional[Node]:
    """Find nearest node to a point."""
    min_dist = float('inf')
    nearest = None
    
    for node in nodes:
        dist = math.hypot(node.x - point[0], node.y - point[1])
        if dist < min_dist and dist < 5.0:  # Within 5 pixels
            min_dist = dist
            nearest = node
    
    return nearest


def optimize_constraints(problem: ConstraintProblem, cfg: Config) -> np.ndarray:
    """
    Run constraint optimization using least squares.
    
    Args:
        problem: Constraint problem definition
        cfg: Configuration object
        
    Returns:
        Optimized variable values
    """
    logger = logging.getLogger('png2svg.constraints')
    
    def objective(variables):
        """Objective function combining data term and constraints."""
        residuals = []
        
        # Data term: distance from observations
        data_residual = variables - problem.observations
        residuals.extend(data_residual * 0.1)  # Lower weight for data term
        
        # Constraint residuals
        for constraint in problem.constraints:
            residual = constraint.residual(variables, problem.node_indices)
            residuals.append(residual)
        
        return np.array(residuals)
    
    # Run optimization
    try:
        result = least_squares(
            objective,
            problem.variables,
            loss='huber',
            f_scale=cfg.algorithms.constraints.huber_delta,
            max_nfev=cfg.algorithms.constraints.max_iterations,
            ftol=cfg.algorithms.constraints.tolerance,
            verbose=0
        )
        
        if result.success:
            logger.info(f"Optimization converged in {result.nfev} iterations")
        else:
            logger.warning(f"Optimization failed to converge: {result.message}")
        
        return result.x
        
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        return problem.variables


def apply_optimized_positions(graph: Graph, optimized: np.ndarray, node_indices: Dict[str, int]) -> None:
    """
    Apply optimized positions back to graph nodes and edges.
    
    Args:
        graph: Topology graph
        optimized: Optimized variable values
        node_indices: Mapping from node IDs to variable indices
    """
    # Update node positions
    for node in graph.nodes:
        if node.id in node_indices:
            idx = node_indices[node.id]
            node.x = float(optimized[idx])
            node.y = float(optimized[idx + 1])
    
    # Update edge endpoints
    for edge in graph.edges:
        if isinstance(edge.geom, LineSeg):
            # Find nodes at endpoints
            start_node = find_nearest_node_to_point(graph.nodes, edge.geom.p1)
            end_node = find_nearest_node_to_point(graph.nodes, edge.geom.p2)
            
            if start_node and end_node:
                # Update line endpoints
                edge.geom.p1 = (start_node.x, start_node.y)
                edge.geom.p2 = (end_node.x, end_node.y)