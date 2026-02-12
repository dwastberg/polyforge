"""Geometry simplification functions.

This module provides high-level functions for simplifying Shapely geometries
by removing or snapping vertices based on various criteria. All public functions
accept and return Shapely geometry objects.

Includes wrappers for the high-performance simplification library algorithms:
- Ramer-Douglas-Peucker (RDP)
- Visvalingam-Whyatt (VW)
- Topology-preserving Visvalingam-Whyatt (VWP)
"""

from typing import Union, Optional
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon, MultiPolygon

from polyforge.process import process_geometry
from .core.types import CollapseMode, coerce_enum
from polyforge.ops.cleanup_ops import (
    remove_small_holes as _remove_small_holes_impl,
    remove_narrow_holes as _remove_narrow_holes_impl,
)
from polyforge.ops.clearance.protrusions import fill_narrow_wedge as _fill_narrow_wedge
from polyforge.ops.clearance.passages import _erode_dilate_fix
from polyforge.ops.simplify_ops import (
    simplify_rdp_coords,
    simplify_vw_coords,
    simplify_vwp_coords,
    snap_short_edges,
    remove_duplicate_vertices,
)
# ============================================================================
# Public API functions (work with Shapely geometries)
# ============================================================================

def collapse_short_edges(
    geometry: BaseGeometry,
    min_length: float,
    snap_mode: Union[CollapseMode, str] = CollapseMode.MIDPOINT,
) -> BaseGeometry:
    """Collapse edges shorter than min_length by snapping vertices together.

    This function cleans up geometries by collapsing very short edges. This is useful
    for removing noise, fixing numerical issues, or simplifying geometries with
    unnecessary detail at small scales.

    Args:
        geometry: Shapely geometry to process (any type)
        min_length: Minimum edge length threshold. Edges shorter than this will be collapsed.
        snap_mode: How to snap vertices together:
            - CollapseMode.MIDPOINT: Snap both vertices to their midpoint (default)
            - CollapseMode.FIRST: Keep the first vertex, remove the second
            - CollapseMode.LAST: Remove the first vertex, keep the second

    Returns:
        New Shapely geometry of the same type with short edges removed

    Examples:
        >>> from polyforge.core.types import CollapseMode
        >>> poly = Polygon([(0, 0), (10, 0), (10, 0.01), (10, 10), (0, 10)])
        >>> clean = collapse_short_edges(poly, min_length=0.1)
        >>> # Edge from (10, 0) to (10, 0.01) is collapsed

        >>> # Using specific mode
        >>> clean = collapse_short_edges(poly, min_length=0.1, snap_mode=CollapseMode.FIRST)

    """
    snap_label = coerce_enum(snap_mode, CollapseMode).value
    return process_geometry(
        geometry,
        snap_short_edges,
        min_length=min_length,
        snap_mode=snap_label,
    )


def deduplicate_vertices(
    geometry: BaseGeometry,
    tolerance: float = 1e-10
) -> BaseGeometry:
    """Remove consecutive duplicate vertices within tolerance.

    This removes exact duplicates (within numerical tolerance) without snapping
    non-duplicate vertices together. For collapsing short edges, use
    collapse_short_edges() instead.

    Args:
        geometry: Shapely geometry to process (any type)
        tolerance: Distance tolerance for considering vertices as duplicates
            (default: 1e-10)

    Returns:
        New Shapely geometry with duplicates removed

    Examples:
        >>> coords = [(0, 0), (0, 0), (10, 0), (10, 10), (0, 10)]
        >>> poly = Polygon(coords)
        >>> clean = deduplicate_vertices(poly)
        >>> # Duplicate (0, 0) vertex removed

    """
    return process_geometry(geometry, remove_duplicate_vertices, tolerance=tolerance)


def simplify_rdp(
    geometry: BaseGeometry,
    epsilon: float
) -> BaseGeometry:
    """Simplify geometry using the Ramer-Douglas-Peucker algorithm.

    RDP is a classic line simplification algorithm that recursively removes vertices
    that are within epsilon distance from the line connecting their neighbors. It's
    fast and produces good results for most use cases.

    Args:
        geometry: Shapely geometry to simplify (any type)
        epsilon: Tolerance value - vertices within this distance from simplified
            line segments will be removed. Larger values = more simplification.
            Suggested starting value: 1.0, or try 0.01-0.001 for fine-tuning.

    Returns:
        New Shapely geometry of the same type with fewer vertices

    """
    return process_geometry(geometry, simplify_rdp_coords, epsilon=epsilon)


def simplify_vw(
    geometry: BaseGeometry,
    threshold: float
) -> BaseGeometry:
    """Simplify geometry using the Visvalingam-Whyatt algorithm.

    VW progressively removes vertices with the smallest effective area until
    the threshold is reached. This often produces more visually pleasing results
    than RDP for certain datasets.

    Args:
        geometry: Shapely geometry to simplify (any type)
        threshold: Area threshold - vertices contributing less than this area
            will be removed. Larger values = more simplification.

    Returns:
        New Shapely geometry of the same type with fewer vertices

    """
    return process_geometry(geometry, simplify_vw_coords, threshold=threshold)


def simplify_vwp(
    geometry: BaseGeometry,
    threshold: float
) -> BaseGeometry:
    """Simplify geometry using topology-preserving Visvalingam-Whyatt.

    This is a slower but more robust variant of VW that ensures the output
    geometry remains topologically valid. Recommended when validity is critical.

    Args:
        geometry: Shapely geometry to simplify (any type)
        threshold: Area threshold - vertices contributing less than this area
            will be removed. Larger values = more simplification.

    Returns:
        New Shapely geometry of the same type with fewer vertices, guaranteed
        to be topologically valid


    """
    return process_geometry(geometry, simplify_vwp_coords, threshold=threshold)

def remove_small_holes(
    geometry: Union[Polygon, MultiPolygon],
    min_area: float,
) -> BaseGeometry:
    """Remove holes smaller than ``min_area``"""
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise TypeError("Input geometry must be a Polygon or MultiPolygon.")
    return _remove_small_holes_impl(geometry, min_area)


def remove_narrow_holes(
    geometry: Union[Polygon, MultiPolygon],
    max_aspect_ratio: float = 50.0,
    min_width: Optional[float] = None,
) -> BaseGeometry:
    """Remove narrow holes based on aspect ratio and/or minimum width."""
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise TypeError("Input geometry must be a Polygon or MultiPolygon.")
    return _remove_narrow_holes_impl(
        geometry,
        max_aspect_ratio=max_aspect_ratio,
        min_width=min_width,
    )


def remove_slivers(
    geometry: Union[Polygon, MultiPolygon],
    min_width: float,
    max_iterations: int = 10,
    min_area_ratio: float = 0.5,
) -> BaseGeometry:
    """Remove thin sliver intrusions from a polygon.

    Args:
        geometry: Input Polygon or MultiPolygon.
        min_width: Slivers narrower than this are removed.
        max_iterations: Maximum trace-based passes (default 10).
        min_area_ratio: Reject result if area drops below this fraction
            of the original (default 0.5).

    Returns:
        Geometry with slivers removed.

    Examples:
        >>> # Rectangle with a narrow slot cut into it
        >>> coords = [(0, 0), (10, 0), (10, 4), (5, 4), (5, 4.2),
        ...           (10, 4.2), (10, 10), (0, 10)]
        >>> poly = Polygon(coords)
        >>> fixed = remove_slivers(poly, min_width=1.0)
        >>> fixed.minimum_clearance >= 1.0
        True
    """
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise TypeError("Input geometry must be a Polygon or MultiPolygon.")

    if isinstance(geometry, MultiPolygon):
        parts = [remove_slivers(p, min_width, max_iterations, min_area_ratio)
                 for p in geometry.geoms]
        return MultiPolygon([p for p in parts if not p.is_empty])

    try:
        if geometry.minimum_clearance >= min_width:
            return geometry
    except Exception:
        return geometry

    original_area = geometry.area
    current = geometry

    # Phase 1: iterative trace-based removal
    for _ in range(max_iterations):
        try:
            if current.minimum_clearance >= min_width:
                break
        except Exception:
            break

        candidate = _fill_narrow_wedge(current, min_width, min_area_ratio)
        if candidate is None:
            break
        current = candidate

    # Phase 2: erosion-dilation fallback
    try:
        if current.minimum_clearance < min_width:
            fallback = _erode_dilate_fix(current, min_width, min_area_ratio)
            if fallback is not None:
                current = fallback
    except Exception:
        pass

    # Final area guard against original
    if original_area > 0 and current.area < min_area_ratio * original_area:
        return geometry

    return current


__all__ = [
    'collapse_short_edges',
    'deduplicate_vertices',
    'simplify_rdp',
    'simplify_vw',
    'simplify_vwp',
    'remove_small_holes',
    'remove_narrow_holes',
    'remove_slivers',
]
