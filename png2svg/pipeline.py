from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .config import Cfg
from .constraints import solve as solve_constraints
from .detect_primitives import Primitives
from .detect_primitives import run as detect_prims
from .detect_symbols import Symbols
from .detect_symbols import run as detect_syms
from .geojson_writer import write as write_geo
from .io_utils import read_image
from .ocr_text import run as ocr_run
from .preprocess import PreprocessOut, run as preprocess_run
from .svg_writer import write as write_svg
from .topology import Graph, build as topo_build


log = logging.getLogger(__name__)


@dataclass
class Result:
    svg: str | None
    geo: str | None


def process_image(path: str, cfg: Cfg) -> Result:
    log.info("processing: %s", path)
    img = read_image(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    pre: PreprocessOut = preprocess_run(img, cfg)
    prim: Primitives = detect_prims(pre, cfg)
    sym: Symbols = detect_syms(pre, cfg)
    ocr = ocr_run(img, pre, cfg)
    topo: Graph = topo_build(prim, sym, ocr, cfg)
    if cfg.apply_constraint_solver:
        topo = solve_constraints(topo, cfg)

    svg_path = write_svg(path, topo, cfg) if cfg.export.write_svg else None
    geo_path = write_geo(path, topo, cfg) if cfg.export.write_geojson else None
    return Result(svg=svg_path, geo=geo_path)

