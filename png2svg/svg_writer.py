from __future__ import annotations

import logging
from pathlib import Path

import svgwrite

from .config import Cfg
from .detect_primitives import CircleArc, LineSeg
from .topology import Graph


log = logging.getLogger(__name__)


def _add_defs(dwg: svgwrite.Drawing):
    # MVP: no defs required
    return


def write(img_path: str, g: Graph, cfg: Cfg) -> str:
    p = Path(img_path)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / (p.stem + ".svg")

    # Canvas size unknown; let it be image-like preserveUnits
    dwg = svgwrite.Drawing(str(svg_path), profile="full", debug=False)
    _add_defs(dwg)

    style = f".main {{stroke:#000; stroke-width:{cfg.svg.stroke_main}px; fill:none}} " \
            f".aux {{stroke:#666; stroke-width:{cfg.svg.stroke_aux}px; fill:none; stroke-dasharray:{cfg.svg.dash_pattern}}} " \
            f"text {{font-family: Arial, sans-serif; font-size: 12px; fill:#111}}"
    dwg.defs.add(dwg.style(style))

    g_main = dwg.g(id="main")
    g_aux = dwg.g(id="aux")
    g_txt = dwg.g(id="text")

    scale = cfg.svg.scale
    for e in g.edges:
        if isinstance(e.geom, LineSeg):
            x1, y1 = e.geom.p1
            x2, y2 = e.geom.p2
            line = dwg.line(start=(x1 * scale, y1 * scale), end=(x2 * scale, y2 * scale))
            line.update({
                "class": e.role,
                "data-role": e.role,
            })
            (g_aux if e.role == "aux" else g_main).add(line)
        elif isinstance(e.geom, CircleArc):
            c = e.geom
            circ = dwg.circle(center=(c.cx * scale, c.cy * scale), r=c.r * scale)
            circ.update({"class": e.role, "data-role": e.role})
            g_main.add(circ)

    # OCR texts
    for t in g.ocr:
        x, y, w, h = t.bbox
        tx = dwg.text(t.text, insert=((x + 1) * scale, (y + h + 1) * scale))
        tx.update({"data-conf": f"{t.conf:.3f}"})
        g_txt.add(tx)

    # Relations as metadata (invisible)
    for i, r in enumerate(g.relations):
        meta = dwg.g(id=f"rel-{i}")
        for k, v in r.items():
            meta.attribs[f"data-{k}"] = str(v)
        meta.attribs["visibility"] = "hidden"
        dwg.add(meta)

    dwg.add(g_main)
    dwg.add(g_aux)
    dwg.add(g_txt)

    dwg.save()
    log.info("svg written: %s", svg_path)
    return str(svg_path)
