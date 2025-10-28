"""Common validation utilities.

This module provides reusable utilities for validation operations
to eliminate code duplication across the codebase.
"""

from typing import Optional
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry


def is_valid_polygon(
    geometry: BaseGeometry,
    min_area: float = 0.0,
    required_type: Optional[str] = None,
    allow_empty: bool = False
) -> bool:
    """Check if geometry meets all validation criteria.

    Comprehensive validation check that combines multiple common checks:
    - Shapely validity (is_valid)
    - Non-empty check
    - Geometry type check (if specified)
    - Minimum area check (if specified)

    Args:
        geometry: Geometry to validate
        min_area: Minimum acceptable area (0 = no minimum)
        required_type: Required geometry type (e.g., 'Polygon', 'MultiPolygon')
        allow_empty: If True, allows empty geometries

    Returns:
        True if geometry meets all criteria, False otherwise

    Examples:
        >>> poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> is_valid_polygon(poly)
        True

        >>> is_valid_polygon(poly, min_area=10.0)
        False  # Area is only 1.0

        >>> is_valid_polygon(poly, required_type='MultiPolygon')
        False  # Wrong type
    """
    # Check Shapely validity
    if not geometry.is_valid:
        return False

    # Check empty
    if not allow_empty and geometry.is_empty:
        return False

    # Check geometry type
    if required_type is not None and geometry.geom_type != required_type:
        return False

    # Check minimum area
    if min_area > 0 and hasattr(geometry, 'area'):
        if geometry.area < min_area:
            return False

    return True


def is_ring_closed(
    coords: np.ndarray,
    tolerance: float = 1e-10
) -> bool:
    """Check if coordinate ring is closed (first == last).

    Args:
        coords: Coordinate array (Nx2 or Nx3)
        tolerance: Tolerance for coordinate comparison

    Returns:
        True if ring is closed (first point equals last point within tolerance)

    Examples:
        >>> coords = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
        >>> is_ring_closed(coords)
        True

        >>> coords = np.array([[0, 0], [1, 0], [1, 1]])
        >>> is_ring_closed(coords)
        False
    """
    if len(coords) < 2:
        return False

    return np.allclose(coords[0], coords[-1], atol=tolerance)


def ensure_ring_closed(coords: np.ndarray) -> np.ndarray:
    """Ensure coordinate ring is closed by appending first point if needed.

    Args:
        coords: Coordinate array (Nx2 or Nx3)

    Returns:
        Coordinate array guaranteed to be closed (may be original if already closed)

    Examples:
        >>> coords = np.array([[0, 0], [1, 0], [1, 1]])
        >>> closed = ensure_ring_closed(coords)
        >>> closed[-1] == closed[0]
        True
        >>> len(closed)
        4
    """
    if len(coords) < 3:
        return coords

    if not np.allclose(coords[0], coords[-1]):
        # Append first point to close ring
        coords = np.vstack([coords, coords[0:1]])

    return coords


def has_minimum_vertices(
    geometry: BaseGeometry,
    min_vertices: int = 4
) -> bool:
    """Check if geometry has minimum number of vertices.

    For Polygons, checks exterior ring. For MultiPolygons, checks
    all component polygons.

    Args:
        geometry: Geometry to check
        min_vertices: Minimum required vertices (default: 4 for valid polygon)

    Returns:
        True if geometry meets minimum vertex requirement

    Examples:
        >>> poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> has_minimum_vertices(poly, min_vertices=4)
        True
    """
    if isinstance(geometry, Polygon):
        return len(geometry.exterior.coords) >= min_vertices
    elif isinstance(geometry, MultiPolygon):
        return all(len(p.exterior.coords) >= min_vertices for p in geometry.geoms)

    return False


def check_area_preserved(
    original: BaseGeometry,
    modified: BaseGeometry,
    min_ratio: float = 0.5
) -> bool:
    """Check if modified geometry preserves sufficient area from original.

    Args:
        original: Original geometry
        modified: Modified geometry
        min_ratio: Minimum ratio of modified/original area (default: 0.5 = 50%)

    Returns:
        True if area is sufficiently preserved

    Examples:
        >>> original = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])  # area = 4
        >>> modified = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])  # area = 1
        >>> check_area_preserved(original, modified, min_ratio=0.2)
        True  # 1/4 = 0.25 > 0.2
    """
    if not (hasattr(original, 'area') and hasattr(modified, 'area')):
        return True  # Can't check area, assume OK

    if original.area == 0:
        return True  # Can't compute ratio

    ratio = modified.area / original.area
    return ratio >= min_ratio


def has_duplicate_vertices(
    coords: np.ndarray,
    tolerance: float = 1e-10
) -> bool:
    """Check if coordinate array has consecutive duplicate vertices.

    Args:
        coords: Coordinate array (Nx2 or Nx3)
        tolerance: Tolerance for considering points duplicate

    Returns:
        True if consecutive duplicates found

    Examples:
        >>> coords = np.array([[0, 0], [0, 0], [1, 1]])
        >>> has_duplicate_vertices(coords)
        True
    """
    if len(coords) < 2:
        return False

    for i in range(len(coords) - 1):
        distance = np.linalg.norm(coords[i] - coords[i + 1])
        if distance < tolerance:
            return True

    return False


__all__ = [
    'is_valid_polygon',
    'is_ring_closed',
    'ensure_ring_closed',
    'has_minimum_vertices',
    'check_area_preserved',
    'has_duplicate_vertices',
]
