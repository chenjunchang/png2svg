from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .config import Cfg


log = logging.getLogger(__name__)


@dataclass
class PreprocessOut:
    img_gray: np.ndarray
    img_bw: np.ndarray
    transform: np.ndarray  # 2x3 affine


def _estimate_skew_angle(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 60, 120, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    if lines is None:
        return 0.0
    angles = []
    for rho_theta in lines[:200]:
        rho, theta = rho_theta[0]
        angle = theta * 180.0 / np.pi
        # Convert to [-90, 90)
        angle = (angle + 90) % 180 - 90
        # We care about near-horizontal being skew; bring to [-45,45]
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180
        angles.append(angle)
    if not angles:
        return 0.0
    # Use median for robustness
    med = float(np.median(np.array(angles)))
    # Only apply small corrections (<= 10 deg)
    if abs(med) > 10:
        return 0.0
    return med


def run(img_bgr: np.ndarray, cfg: Cfg) -> PreprocessOut:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 10)

    M = np.eye(2, 3, dtype=np.float32)
    if cfg.deskew:
        try:
            angle = _estimate_skew_angle(gray)
            if abs(angle) > 0.1:
                h, w = gray.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                bw = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                log.debug("deskew applied angle=%.3f deg", angle)
        except Exception:
            log.warning("deskew failed; continue without rotation", exc_info=True)

    return PreprocessOut(img_gray=gray, img_bw=bw, transform=M)

