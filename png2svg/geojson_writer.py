from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import Cfg
from .detect_primitives import CircleArc, LineSeg
from .topology import Graph


log = logging.getLogger(__name__)


def write(img_path: str, g: Graph, cfg: Cfg) -> str:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / (Path(img_path).stem + ".geo.json")

    nodes = [
        {"id": i, "x": n.x, "y": n.y, "tag": n.tag, "kind": n.kind}
        for i, n in enumerate(g.nodes)
    ]
    edges = []
    for i, e in enumerate(g.edges):
        if isinstance(e.geom, LineSeg):
            edges.append({
                "id": i,
                "type": "line",
                "endpoints": [list(e.geom.p1), list(e.geom.p2)],
                "role": e.role,
            })
        elif isinstance(e.geom, CircleArc):
            edges.append({
                "id": i,
                "type": e.geom.kind,
                "center": [e.geom.cx, e.geom.cy],
                "radius": e.geom.r,
                "theta": [e.geom.theta1, e.geom.theta2],
                "role": e.role,
            })
    data = {
        "nodes": nodes,
        "edges": edges,
        "relations": g.relations,
        "ocr": [
            {"text": t.text, "bbox": list(t.bbox), "conf": float(t.conf)} for t in g.ocr
        ],
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info("geojson written: %s", p)
    return str(p)

