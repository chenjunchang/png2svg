from __future__ import annotations

import logging

from .config import Cfg
from .topology import Graph


log = logging.getLogger(__name__)


def solve(g: Graph, cfg: Cfg) -> Graph:
    # MVP: no-op solver; placeholder for future least-squares optimization
    log.info("constraint solver skipped (MVP)")
    return g

