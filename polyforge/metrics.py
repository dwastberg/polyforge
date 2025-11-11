"""Shared measurement helpers for polyforge geometries.

The high-level pipeline only needs a handful of scalar metrics to decide
whether a given geometry is getting better or worse. Centralizing the logic
here keeps the rest of the codebase free from ad-hoc ``minimum_clearance`` or
area checks.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def _safe_clearance(geometry: BaseGeometry) -> Optional[float]:
    try:
        return geometry.minimum_clearance
    except Exception:
        return None


def measure_geometry(
    geometry: BaseGeometry,
    original: Optional[BaseGeometry] = None,
) -> Dict[str, Optional[float]]:
    """Return core metrics for ``geometry``."""
    area = getattr(geometry, "area", None)
    original_area = getattr(original, "area", None) if original is not None else None
    area_ratio: Optional[float] = None

    if area is not None and original_area and original_area > 0:
        area_ratio = area / original_area

    return {
        "is_valid": getattr(geometry, "is_valid", False),
        "is_empty": getattr(geometry, "is_empty", True),
        "clearance": _safe_clearance(geometry),
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


__all__ = [
    "measure_geometry",
    "total_overlap_area",
]
