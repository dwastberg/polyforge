"""Geometry simplification functions.

This module provides high-level functions for simplifying Shapely geometries
by removing or snapping vertices based on various criteria. All public functions
accept and return Shapely geometry objects.

Includes wrappers for the high-performance simplification library algorithms:
- Ramer-Douglas-Peucker (RDP)
- Visvalingam-Whyatt (VW)
- Topology-preserving Visvalingam-Whyatt (VWP)
"""

import numpy as np
from typing import Literal, Union
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon, MultiPolygon
from simplification.cutil import (
    simplify_coords as _rdp_simplify,
    simplify_coords_vw as _vw_simplify,
    simplify_coords_vwp as _vwp_simplify
)

from polyforge.process import process_geometry
from .core.types import CollapseMode


# ============================================================================
# Private processing functions (work with numpy arrays)
# ============================================================================

def _simplify_rdp_wrapper(vertices: np.ndarray, epsilon: float) -> np.ndarray:
    """Internal function: Simplify using Ramer-Douglas-Peucker algorithm.

    Args:
        vertices: Numpy array of 2D vertices (Nx2)
        epsilon: Tolerance value for RDP algorithm

    Returns:
        Numpy array of simplified vertices
    """
    if len(vertices) < 2:
        return vertices.copy()

    # The simplification library expects a list or numpy array
    # and returns a numpy array if input is numpy
    result = _rdp_simplify(vertices, epsilon)
    return np.array(result) if not isinstance(result, np.ndarray) else result


def _simplify_vw_wrapper(vertices: np.ndarray, threshold: float) -> np.ndarray:
    """Internal function: Simplify using Visvalingam-Whyatt algorithm.

    Args:
        vertices: Numpy array of 2D vertices (Nx2)
        threshold: Area threshold for VW algorithm

    Returns:
        Numpy array of simplified vertices
    """
    if len(vertices) < 2:
        return vertices.copy()

    result = _vw_simplify(vertices, threshold)
    return np.array(result) if not isinstance(result, np.ndarray) else result


def _simplify_vwp_wrapper(vertices: np.ndarray, threshold: float) -> np.ndarray:
    """Internal function: Simplify using topology-preserving Visvalingam-Whyatt.

    Args:
        vertices: Numpy array of 2D vertices (Nx2)
        threshold: Area threshold for VWP algorithm

    Returns:
        Numpy array of simplified vertices
    """
    if len(vertices) < 2:
        return vertices.copy()

    result = _vwp_simplify(vertices, threshold)
    return np.array(result) if not isinstance(result, np.ndarray) else result


def _snap_short_edges(
    vertices: np.ndarray,
    min_length: float,
    snap_mode: Literal['midpoint', 'first', 'last'] = 'midpoint'
) -> np.ndarray:
    """Internal function: Snap vertices together if edges are shorter than min_length.

    Args:
        vertices: Numpy array of 2D vertices (Nx2)
        min_length: Minimum edge length threshold
        snap_mode: How to snap vertices together

    Returns:
        Numpy array of vertices with short edges removed (Mx2 where M <= N)
    """
    if len(vertices) < 2:
        return vertices.copy()

    # Check if this is a closed ring (first and last vertices are the same)
    is_closed = np.allclose(vertices[0], vertices[-1])

    # We'll build a new vertex list
    result = []
    i = 0
    n = len(vertices)

    # For closed rings, we need to check the wrap-around edge
    if is_closed:
        n = n - 1  # Don't process the duplicate closing vertex yet

    while i < n:
        current = vertices[i]

        # Look ahead to find the next vertex that's far enough away
        j = i + 1
        while j < n:
            next_vertex = vertices[j]
            distance = np.linalg.norm(next_vertex - current)

            if distance >= min_length:
                # This edge is long enough, keep the current vertex
                break

            # Edge is too short, snap according to mode
            if snap_mode == 'midpoint':
                current = (current + next_vertex) / 2.0
            elif snap_mode == 'last':
                current = next_vertex
            # For 'first' mode, we keep current as is

            j += 1

        result.append(current)
        i = j

    # Handle closed rings
    if is_closed and len(result) > 0:
        # Check the edge between last and first vertex
        if len(result) >= 2:
            first = result[0]
            last = result[-1]
            distance = np.linalg.norm(last - first)

            if distance < min_length:
                # Snap the first and last vertices together
                if snap_mode == 'midpoint':
                    snapped = (first + last) / 2.0
                    result[0] = snapped
                    result[-1] = snapped
                elif snap_mode == 'first':
                    result[-1] = first
                elif snap_mode == 'last':
                    result[0] = last
            else:
                # Close the ring normally
                result.append(result[0].copy())
        else:
            # Only one vertex left, just close it
            result.append(result[0].copy())

    # Ensure we have at least 2 vertices for a valid geometry
    if len(result) < 2:
        # If we collapsed everything, return the first two original vertices
        return vertices[:2].copy()

    return np.array(result)


def _remove_duplicate_vertices(
    vertices: np.ndarray,
    tolerance: float = 1e-10
) -> np.ndarray:
    """Internal function: Remove consecutive duplicate vertices within tolerance.

    Args:
        vertices: Numpy array of 2D vertices (Nx2)
        tolerance: Distance tolerance for considering vertices as duplicates

    Returns:
        Numpy array of vertices with duplicates removed
    """
    if len(vertices) < 2:
        return vertices.copy()

    is_closed = np.allclose(vertices[0], vertices[-1])

    # Find non-duplicate vertices
    result = [vertices[0]]

    for i in range(1, len(vertices)):
        distance = np.linalg.norm(vertices[i] - result[-1])
        if distance > tolerance:
            result.append(vertices[i])

    # For closed rings, ensure it's properly closed
    if is_closed and len(result) > 1:
        if not np.allclose(result[0], result[-1]):
            result.append(result[0].copy())

    # Ensure at least 2 vertices
    if len(result) < 2:
        return vertices[:2].copy()

    return np.array(result)


# ============================================================================
# Public API functions (work with Shapely geometries)
# ============================================================================

def collapse_short_edges(
    geometry: BaseGeometry,
    min_length: float,
    snap_mode: CollapseMode = CollapseMode.MIDPOINT
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
    return process_geometry(geometry, _snap_short_edges, min_length=min_length, snap_mode=snap_mode.value)


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
    return process_geometry(geometry, _remove_duplicate_vertices, tolerance=tolerance)


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
    return process_geometry(geometry, _simplify_rdp_wrapper, epsilon=epsilon)


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
    return process_geometry(geometry, _simplify_vw_wrapper, threshold=threshold)


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
    return process_geometry(geometry, _simplify_vwp_wrapper, threshold=threshold)

def remove_small_holes(
    geometry: Union[Polygon, MultiPolygon],
    min_area: float
) -> BaseGeometry:
    """Remove small holes from Polygon geometries.

    This function removes interior rings (holes) from Polygon geometries
    that have an area smaller than the specified minimum area threshold.

    Args:
        geometry: Shapely Polygon geometry to process
        min_area: Minimum area threshold. Holes with area less than this
            will be removed.
    Returns:
        New Shapely Polygon geometry with small holes removed
    """

    def _remove_small_holes_from_polygon(polygon: Polygon, min_area: float) -> Polygon:
        # Filter interior rings based on area
        new_interiors = [
            interior for interior in polygon.interiors
            if Polygon(interior).area >= min_area
        ]
        return Polygon(polygon.exterior.coords, holes=[interior.coords for interior in new_interiors])

    geom_type = geometry.geom_type

    if geom_type == 'Polygon':
        return _remove_small_holes_from_polygon(geometry, min_area)

    elif geom_type == 'MultiPolygon':
        new_polygons = [
            _remove_small_holes_from_polygon(polygon, min_area)
            for polygon in geometry.geoms
        ]
        return MultiPolygon(new_polygons)

    else:
        raise TypeError("Input geometry must be a Polygon or MultiPolygon.")


__all__ = [
    'collapse_short_edges',
    'deduplicate_vertices',
    'simplify_rdp',
    'simplify_vw',
    'simplify_vwp',
    'remove_small_holes',
]
