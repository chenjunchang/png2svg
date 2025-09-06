from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .config import Cfg
from .preprocess import PreprocessOut


log = logging.getLogger(__name__)


@dataclass
class LineSeg:
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    dashed: bool = False
    thickness: int = 2
    role: str = "main"
    id: str = ""


@dataclass
class CircleArc:
    cx: float
    cy: float
    r: float
    theta1: float = 0.0
    theta2: float = 360.0
    kind: str = "circle"  # circle/arc
    id: str = ""


@dataclass
class Primitives:
    lines: List[LineSeg]
    circles: List[CircleArc]


def _merge_collinear(lines: List[LineSeg], angle_thr_deg: float, gap_px: float) -> List[LineSeg]:
    if not lines:
        return []

    def angle(l: LineSeg) -> float:
        dx = l.p2[0] - l.p1[0]
        dy = l.p2[1] - l.p1[1]
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        return ang

    merged: List[LineSeg] = []
    used = [False] * len(lines)
    for i, li in enumerate(lines):
        if used[i]:
            continue
        ai = angle(li)
        xys = [li.p1, li.p2]
        used[i] = True
        for j, lj in enumerate(lines):
            if i == j or used[j]:
                continue
            aj = angle(lj)
            da = min(abs(ai - aj), 180 - abs(ai - aj))
            if da > angle_thr_deg:
                continue
            # Check if roughly on same line by distance between endpoints and line equation
            # Simple heuristic: any endpoint within gap to any endpoint
            pts = [li.p1, li.p2, lj.p1, lj.p2]
            close = False
            for a in (li.p1, li.p2):
                for b in (lj.p1, lj.p2):
                    if math.hypot(a[0] - b[0], a[1] - b[1]) <= gap_px:
                        close = True
                        break
                if close:
                    break
            if not close:
                continue
            xys.extend([lj.p1, lj.p2])
            used[j] = True
        # create bounding segment along principal direction
        xs = [p[0] for p in xys]
        ys = [p[1] for p in xys]
        # project onto direction vector
        dx = math.cos(math.radians(ai))
        dy = math.sin(math.radians(ai))
        t = [x * dx + y * dy for x, y in zip(xs, ys)]
        i_min = int(np.argmin(t))
        i_max = int(np.argmax(t))
        p_min = (xs[i_min], ys[i_min])
        p_max = (xs[i_max], ys[i_max])
        merged.append(LineSeg(p1=p_min, p2=p_max))
    return merged


def _is_dashed(seg: LineSeg, img_bw: np.ndarray) -> bool:
    (x1, y1), (x2, y2) = seg.p1, seg.p2
    L = int(max(1.0, math.hypot(x2 - x1, y2 - y1)))
    if L < 20:
        return False
    xs = np.linspace(x1, x2, L).astype(int)
    ys = np.linspace(y1, y2, L).astype(int)
    xs = np.clip(xs, 0, img_bw.shape[1] - 1)
    ys = np.clip(ys, 0, img_bw.shape[0] - 1)
    vals = img_bw[ys, xs]
    runs = []
    last = vals[0]
    cnt = 1
    for v in vals[1:]:
        if v == last:
            cnt += 1
        else:
            runs.append((int(last), cnt))
            last = v
            cnt = 1
    runs.append((int(last), cnt))
    on_runs = [l for val, l in runs if val > 0]
    off_runs = [l for val, l in runs if val == 0]
    if len(on_runs) < 2:
        return False
    return (np.mean(on_runs) > 3) and (len(off_runs) >= 1)


def _detect_lines(pre: PreprocessOut, cfg: Cfg) -> List[LineSeg]:
    lines: List[LineSeg] = []
    gray = pre.img_gray
    try:
        lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_STD)
        detected = lsd.detect(gray)[0]
        if detected is not None:
            for x1, y1, x2, y2 in detected.reshape(-1, 4):
                lines.append(LineSeg((float(x1), float(y1)), (float(x2), float(y2))))
    except Exception:
        log.debug("LSD not available; fallback to HoughLinesP", exc_info=True)
    if not lines:
        edges = cv2.Canny(gray, 60, 120, apertureSize=3)
        hlines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            50,
            minLineLength=int(cfg.min_line_len),
            maxLineGap=4,
        )
        if hlines is not None:
            for x1, y1, x2, y2 in hlines.reshape(-1, 4):
                lines.append(LineSeg((float(x1), float(y1)), (float(x2), float(y2))))
    # merge & dashed
    lines = _merge_collinear(lines, cfg.line_merge_angle_deg, cfg.line_merge_gap_px)
    for ln in lines:
        ln.dashed = _is_dashed(ln, pre.img_bw)
        ln.role = "aux" if ln.dashed else "main"
    return lines


def _detect_circles(pre: PreprocessOut) -> List[CircleArc]:
    gray = pre.img_gray
    circles: List[CircleArc] = []
    try:
        edges = cv2.medianBlur(gray, 3)
        c = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=120,
            param2=30,
            minRadius=8,
            maxRadius=0,
        )
        if c is not None:
            for x, y, r in np.uint16(np.around(c[0, :])):
                circles.append(CircleArc(float(x), float(y), float(r)))
    except Exception:
        log.debug("HoughCircles failed", exc_info=True)
    return circles


def run(pre: PreprocessOut, cfg: Cfg) -> Primitives:
    lines = _detect_lines(pre, cfg)
    circles = _detect_circles(pre)
    log.info("primitives: %d lines, %d circles", len(lines), len(circles))
    return Primitives(lines=lines, circles=circles)

