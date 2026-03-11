"""Shared micro-helpers for clearance fixing."""
from __future__ import annotations

from collections.abc import Iterable

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from polyforge.core.geometry_utils import to_single_polygon
from polyforge.metrics import _safe_clearance


def clearance_or_zero(geometry: Polygon) -> float:
    """Return _safe_clearance defaulting None to 0.0."""
    return _safe_clearance(geometry) or 0.0


def is_usable(candidate: Polygon | None) -> bool:
    """True if candidate is a valid, non-empty polygon."""
    return candidate is not None and candidate.is_valid and not candidate.is_empty


def pick_best_by_clearance(
    geometry: Polygon,
    candidates: Iterable[Polygon | None],
) -> Polygon | None:
    """Return the candidate with highest clearance exceeding geometry's current.

    Returns None if no candidate improves on the current clearance.
    """
    current_clearance = clearance_or_zero(geometry)
    best_candidate = None
    best_clearance = current_clearance

    for candidate in candidates:
        if not is_usable(candidate):
            continue
        cand_clearance = clearance_or_zero(candidate)
        if cand_clearance > best_clearance:
            best_clearance = cand_clearance
            best_candidate = candidate

    return best_candidate


def normalize_polygon(candidate: BaseGeometry | None) -> Polygon | None:
    """Normalize a geometry result to a single valid Polygon, or None."""
    if candidate is None:
        return None
    if candidate.is_empty or not candidate.is_valid:
        return None
    polygon = to_single_polygon(candidate)
    return polygon if polygon.is_valid and not polygon.is_empty else None
