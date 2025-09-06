from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class SvgCfg:
    scale: float = 1.0
    stroke_main: int = 2
    stroke_aux: int = 1
    dash_pattern: str = "6,6"


@dataclass
class ExportCfg:
    write_geojson: bool = True
    write_svg: bool = True


@dataclass
class Cfg:
    input_dir: str = "./pngs"
    output_dir: str = "./out"
    jobs: int = 4
    deskew: bool = True
    min_line_len: int = 25
    line_merge_angle_deg: float = 3.0
    line_merge_gap_px: int = 6
    use_yolo_symbols: bool = True
    yolo_symbols_weights: str = "models/symbols_yolo.onnx"
    use_paddle_ocr: bool = True
    confidence_tta: bool = True
    apply_constraint_solver: bool = True
    svg: SvgCfg = field(default_factory=SvgCfg)
    export: ExportCfg = field(default_factory=ExportCfg)
    log_level: str = "INFO"

    # runtime-only
    _from: str = "defaults"


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _asdict(cfg: Cfg) -> dict[str, Any]:
    return {
        "input_dir": cfg.input_dir,
        "output_dir": cfg.output_dir,
        "jobs": cfg.jobs,
        "deskew": cfg.deskew,
        "min_line_len": cfg.min_line_len,
        "line_merge_angle_deg": cfg.line_merge_angle_deg,
        "line_merge_gap_px": cfg.line_merge_gap_px,
        "use_yolo_symbols": cfg.use_yolo_symbols,
        "yolo_symbols_weights": cfg.yolo_symbols_weights,
        "use_paddle_ocr": cfg.use_paddle_ocr,
        "confidence_tta": cfg.confidence_tta,
        "apply_constraint_solver": cfg.apply_constraint_solver,
        "svg": {
            "scale": cfg.svg.scale,
            "stroke_main": cfg.svg.stroke_main,
            "stroke_aux": cfg.svg.stroke_aux,
            "dash_pattern": cfg.svg.dash_pattern,
        },
        "export": {
            "write_geojson": cfg.export.write_geojson,
            "write_svg": cfg.export.write_svg,
        },
        "log_level": cfg.log_level,
    }


def _from_dict(d: dict[str, Any]) -> Cfg:
    svg = d.get("svg", {})
    export = d.get("export", {})
    return Cfg(
        input_dir=str(d.get("input_dir", "./pngs")),
        output_dir=str(d.get("output_dir", "./out")),
        jobs=int(d.get("jobs", 4)),
        deskew=bool(d.get("deskew", True)),
        min_line_len=int(d.get("min_line_len", 25)),
        line_merge_angle_deg=float(d.get("line_merge_angle_deg", 3.0)),
        line_merge_gap_px=int(d.get("line_merge_gap_px", 6)),
        use_yolo_symbols=bool(d.get("use_yolo_symbols", True)),
        yolo_symbols_weights=str(d.get("yolo_symbols_weights", "models/symbols_yolo.onnx")),
        use_paddle_ocr=bool(d.get("use_paddle_ocr", True)),
        confidence_tta=bool(d.get("confidence_tta", True)),
        apply_constraint_solver=bool(d.get("apply_constraint_solver", True)),
        svg=SvgCfg(
            scale=float(svg.get("scale", 1.0)),
            stroke_main=int(svg.get("stroke_main", 2)),
            stroke_aux=int(svg.get("stroke_aux", 1)),
            dash_pattern=str(svg.get("dash_pattern", "6,6")),
        ),
        export=ExportCfg(
            write_geojson=bool(export.get("write_geojson", True)),
            write_svg=bool(export.get("write_svg", True)),
        ),
        log_level=str(d.get("log_level", "INFO")),
    )


def load_config(args: Optional[Any]) -> Cfg:
    # Start with defaults
    base = _asdict(Cfg())

    # Merge YAML if present
    cfg_path: Optional[Path] = None
    if args and getattr(args, "config", None):
        cfg_path = Path(args.config)
    else:
        # Use default if exists
        p = Path("config.yaml")
        cfg_path = p if p.exists() else None

    if cfg_path and cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        base = _deep_update(base, y)

    # CLI overrides
    ov: dict[str, Any] = {}
    if args:
        if getattr(args, "input", None):
            ov["input_dir"] = str(args.input)
        if getattr(args, "output", None):
            ov["output_dir"] = str(args.output)
        if getattr(args, "jobs", None):
            ov["jobs"] = int(args.jobs)
        if getattr(args, "no_yolo", False):
            ov["use_yolo_symbols"] = False
        if getattr(args, "no_paddleocr", False):
            ov["use_paddle_ocr"] = False
        if getattr(args, "no_constraints", False):
            ov["apply_constraint_solver"] = False

    cfg = _from_dict(_deep_update(base, ov))
    cfg._from = f"yaml:{cfg_path}" if cfg_path else "defaults+cli"

    # Setup logging
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

    # Auto-disable if optional weights/libs missing
    weights = Path(cfg.yolo_symbols_weights)
    if cfg.use_yolo_symbols and not weights.exists():
        logging.getLogger(__name__).warning("YOLO weights not found: %s; disabling symbols detector", weights)
        cfg.use_yolo_symbols = False

    return cfg

