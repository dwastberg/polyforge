"""Polygon overlap splitting functions (thin wrapper over the overlap engine)."""

from __future__ import annotations

from typing import Tuple

from shapely.geometry import Polygon

from .core.types import OverlapStrategy
from .overlap.engine import resolve_overlap_pair


def split_overlap(
    poly1: Polygon,
    poly2: Polygon,
    overlap_strategy: OverlapStrategy = OverlapStrategy.SPLIT,
) -> Tuple[Polygon, Polygon]:
    """Split or assign the overlapping area between two polygons."""
    return resolve_overlap_pair(poly1, poly2, strategy=overlap_strategy)


__all__ = ["split_overlap"]
