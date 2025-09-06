from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import Cfg
from .preprocess import PreprocessOut


log = logging.getLogger(__name__)


@dataclass
class OCRItem:
    text: str
    bbox: Tuple[int, int, int, int]
    conf: float


def _paddleocr(img_bgr: np.ndarray) -> List[OCRItem]:  # pragma: no cover - optional
    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=True, show_log=False)
        h, w = img_bgr.shape[:2]
        res = ocr.ocr(img_bgr, cls=True)
        items: List[OCRItem] = []
        for line in res:
            for box, (txt, conf) in line:
                xs = [int(p[0]) for p in box]
                ys = [int(p[1]) for p in box]
                x, y, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                items.append(OCRItem(text=str(txt).strip(), bbox=(x, y, x2 - x, y2 - y), conf=float(conf)))
        return items
    except Exception:
        log.warning("PaddleOCR not available; using pytesseract")
        return []


def _tesseract(img_bgr: np.ndarray) -> List[OCRItem]:  # pragma: no cover - optional depending on env
    try:
        import pytesseract
        data = pytesseract.image_to_data(img_bgr, output_type=pytesseract.Output.DICT)
        items: List[OCRItem] = []
        n = len(data.get("text", []))
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue
            x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
            conf_str = data.get("conf", ["0"][i])
            try:
                conf = float(conf_str)
            except Exception:
                conf = 0.0
            items.append(OCRItem(text=txt, bbox=(x, y, w, h), conf=conf))
        return items
    except Exception:
        log.warning("pytesseract not available; OCR disabled")
        return []


def run(img_bgr: np.ndarray, pre: PreprocessOut, cfg: Cfg) -> List[OCRItem]:
    if cfg.use_paddle_ocr:
        items = _paddleocr(img_bgr)
        if items:
            log.info("ocr (paddle): %d", len(items))
            return items
    items = _tesseract(img_bgr)
    log.info("ocr (tesseract): %d", len(items))
    return items

