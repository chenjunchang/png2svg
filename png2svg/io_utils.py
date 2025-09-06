from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np
from tqdm import tqdm


log = logging.getLogger(__name__)


def read_image(path: str):
    try:
        data = Path(path).read_bytes()
        nparr = np.frombuffer(data, dtype=np.uint8)  # type: ignore[name-defined]
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        # Fallback to cv2.imread if imdecode fails with some FS
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def list_pngs(input_dir: str) -> list[str]:
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"Input directory not found: {p}")
    seen = set()
    files: list[str] = []
    for ext in ("*.png", "*.PNG"):
        for x in p.glob(ext):
            s = str(x)
            if s in seen:
                continue
            seen.add(s)
            files.append(s)
    files.sort()
    return files


@dataclass
class TaskResult:
    path: str
    ok: bool
    svg: str | None = None
    geo: str | None = None
    error: str | None = None


def _worker(args):
    path, func, cfg = args
    try:
        res = func(path, cfg)
        return TaskResult(path=path, ok=True, svg=res.svg, geo=res.geo)
    except Exception as e:  # pragma: no cover - robustness path
        log.exception("Failed processing %s", path)
        return TaskResult(path=path, ok=False, error=str(e))


def run_parallel(files: Sequence[str], func: Callable, cfg, jobs: int = 1) -> list[TaskResult]:
    if jobs is None or jobs <= 1:
        out = []
        for f in tqdm(files, desc="png2svg", unit="img"):
            out.append(_worker((f, func, cfg)))
        return out
    with Pool(processes=int(jobs)) as pool:
        args = [(f, func, cfg) for f in files]
        results = []
        for r in tqdm(pool.imap_unordered(_worker, args), total=len(files), desc="png2svg", unit="img"):
            results.append(r)
        # keep order as input
        order = {p: i for i, p in enumerate(files)}
        results.sort(key=lambda r: order.get(r.path, 0))
        return results
