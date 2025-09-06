from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import Cfg
from .preprocess import PreprocessOut


log = logging.getLogger(__name__)


@dataclass
class Symbol:
    cls: str
    bbox: Tuple[int, int, int, int]
    conf: float


@dataclass
class Symbols:
    items: List[Symbol]


def _infer_with_onnx(pre: PreprocessOut, weights: Path) -> List[Symbol]:  # pragma: no cover - optional
    try:
        import onnxruntime as ort
        # This is a placeholder. Real deployment would depend on training specifics.
        # For MVP, we return empty to avoid false positives.
        log.info("ONNX runtime available, but no model IO defined. Skipping.")
        return []
    except Exception:
        log.warning("onnxruntime not available; fall back to rules")
        return []


def _rule_based(pre: PreprocessOut) -> List[Symbol]:
    # Minimal stub: return empty list; later can add template matching
    return []


def run(pre: PreprocessOut, cfg: Cfg) -> Symbols:
    items: List[Symbol] = []
    if cfg.use_yolo_symbols and Path(cfg.yolo_symbols_weights).exists():
        items = _infer_with_onnx(pre, Path(cfg.yolo_symbols_weights))
    else:
        items = _rule_based(pre)
    log.info("symbols: %d", len(items))
    return Symbols(items=items)

