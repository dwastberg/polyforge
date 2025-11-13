"""Shared measurement helpers for polyforge geometries.

The high-level pipeline only needs a handful of scalar metrics to decide
whether a given geometry is getting better or worse. Centralizing the logic
here keeps the rest of the codebase free from ad-hoc ``minimum_clearance`` or
area checks.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree


def _safe_clearance(geometry: BaseGeometry) -> Optional[float]:
    try:
        return geometry.minimum_clearance
    except Exception:
        return None


def measure_geometry(
    geometry: BaseGeometry,
    original: Optional[BaseGeometry] = None,
    skip_clearance: bool = False,
) -> Dict[str, Optional[float]]:
    """Return core metrics for ``geometry``.

    Args:
        geometry: Geometry to measure
        original: Original geometry for area ratio calculation
        skip_clearance: If True, skip expensive clearance calculation (default: False)

    Returns:
        Dict with keys: is_valid, is_empty, clearance, area, area_ratio
        Note: clearance will be None if skip_clearance=True

    Examples:
        >>> poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> metrics = measure_geometry(poly)
        >>> metrics['area']
        1.0

        >>> # Skip clearance when not needed for better performance
        >>> metrics = measure_geometry(poly, skip_clearance=True)
        >>> metrics['clearance']  # Will be None
    """
    area = getattr(geometry, "area", None)
    original_area = getattr(original, "area", None) if original is not None else None
    area_ratio: Optional[float] = None

    if area is not None and original_area and original_area > 0:
        area_ratio = area / original_area

    # Only calculate clearance if requested (expensive operation)
    clearance = None if skip_clearance else _safe_clearance(geometry)

    return {
        "is_valid": getattr(geometry, "is_valid", False),
        "is_empty": getattr(geometry, "is_empty", True),
        "clearance": clearance,
        "area": area,
        "area_ratio": area_ratio,
    }


def total_overlap_area(geometries: Iterable[BaseGeometry]) -> float:
    """Compute the total overlapping area within ``geometries``."""
    geometries = [geom for geom in geometries if geom and not geom.is_empty]
    if len(geometries) < 2:
        return 0.0
    union = unary_union(geometries)
    combined_area = sum(getattr(geom, "area", 0.0) for geom in geometries)
    return combined_area - getattr(union, "area", 0.0)


def overlap_area_by_geometry(
    geometries: List[BaseGeometry],
    min_area_threshold: float = 1e-10,
) -> List[float]:
    """Return the overlapping area attributed to each geometry in ``geometries``."""
    if not geometries:
        return []

    overlaps = [0.0 for _ in geometries]
    if len(geometries) < 2:
        return overlaps

    tree = STRtree(geometries)
    processed = set()

    for i, geom_i in enumerate(geometries):
        if geom_i is None or geom_i.is_empty:
            continue

        try:
            candidates = tree.query(geom_i, predicate="intersects")
        except Exception:
            continue

        for j in candidates:
            if j <= i:
                continue
            pair = (i, j)
            if pair in processed:
                continue
            processed.add(pair)

            geom_j = geometries[j]
            if geom_j is None or geom_j.is_empty:
                continue

            try:
                overlap = geom_i.intersection(geom_j)
            except Exception:
                continue

            area = getattr(overlap, "area", 0.0)
            if area > min_area_threshold:
                overlaps[i] += area
                overlaps[j] += area

    return overlaps


__all__ = [
    "measure_geometry",
    "total_overlap_area",
    "overlap_area_by_geometry",
]
