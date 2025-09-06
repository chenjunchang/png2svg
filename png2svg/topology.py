from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
from shapely.geometry import LineString, Point

from .config import Cfg
from .detect_primitives import CircleArc, LineSeg, Primitives
from .detect_symbols import Symbols
from .ocr_text import OCRItem


log = logging.getLogger(__name__)


@dataclass
class Node:
    x: float
    y: float
    tag: str = ""
    kind: str = "point"
    id: str = ""


@dataclass
class Edge:
    geom: Union[LineSeg, CircleArc]
    role: str = "main"
    attrs: dict = field(default_factory=dict)


@dataclass
class Graph:
    nodes: List[Node]
    edges: List[Edge]
    relations: List[dict]
    ocr: List[OCRItem] = field(default_factory=list)


def _angle_of(l: LineSeg) -> float:
    dx = l.p2[0] - l.p1[0]
    dy = l.p2[1] - l.p1[1]
    return math.atan2(dy, dx)


def _add_node(nodes: List[Node], x: float, y: float) -> int:
    key = (round(x, 2), round(y, 2))
    for i, n in enumerate(nodes):
        if (round(n.x, 2), round(n.y, 2)) == key:
            return i
    nodes.append(Node(x=float(x), y=float(y)))
    return len(nodes) - 1


def build(prim: Primitives, sym: Symbols, ocr: List[OCRItem], cfg: Cfg) -> Graph:
    nodes: List[Node] = []
    edges: List[Edge] = []
    relations: List[dict] = []

    # Add edges and collect endpoints
    for i, l in enumerate(prim.lines):
        _add_node(nodes, l.p1[0], l.p1[1])
        _add_node(nodes, l.p2[0], l.p2[1])
        edges.append(Edge(geom=l, role=l.role))

    for c in prim.circles:
        _add_node(nodes, c.cx, c.cy)
        edges.append(Edge(geom=c, role="main"))

    # Compute intersections between line segments
    segs = [LineString([l.p1, l.p2]) for l in prim.lines]
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            si, sj = segs[i], segs[j]
            if not si.intersects(sj):
                continue
            inter = si.intersection(sj)
            if inter.is_empty:
                continue
            if inter.geom_type == "Point":
                _add_node(nodes, inter.x, inter.y)

    # Basic relations: parallel/perpendicular between lines
    angs = [_angle_of(l) for l in prim.lines]
    for i in range(len(prim.lines)):
        for j in range(i + 1, len(prim.lines)):
            ai, aj = angs[i], angs[j]
            da = abs(ai - aj)
            da = min(da, math.pi - da)
            if da < math.radians(3.0):
                relations.append({"type": "parallel", "members": [i, j], "conf": 0.6})
            if abs(da - math.pi / 2) < math.radians(3.0):
                relations.append({"type": "perp", "members": [i, j], "conf": 0.6})

    # Bind ocr raw (no semantic binding in MVP)
    g = Graph(nodes=nodes, edges=edges, relations=relations, ocr=ocr)
    log.info("graph: %d nodes, %d edges, %d relations, %d ocr", len(nodes), len(edges), len(relations), len(ocr))
    return g

