"""Common geometry manipulation utilities.

This module provides reusable utilities for common geometry operations
to eliminate code duplication across the codebase.
"""

from typing import Union, List
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString
from shapely.geometry.base import BaseGeometry
from shapely.errors import GEOSException


def to_single_polygon(geometry: BaseGeometry) -> Polygon:
    """Convert geometry to a single Polygon by taking the largest piece.

    If the geometry is already a Polygon, returns it unchanged.
    If it's a MultiPolygon, returns the largest polygon by area.
    If it's a GeometryCollection, extracts polygons and returns the largest.

    Args:
        geometry: Input geometry

    Returns:
        Single Polygon (largest if multiple pieces exist)

    Examples:
        >>> poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> result = to_single_polygon(poly)
        >>> result.equals(poly)
        True

        >>> multi = MultiPolygon([poly1, poly2, poly3])
        >>> result = to_single_polygon(multi)
        >>> # Returns the polygon with largest area
    """
    if isinstance(geometry, Polygon):
        return geometry
    elif isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda p: p.area)
    elif isinstance(geometry, GeometryCollection):
        # Extract polygons from collection
        polygons = [g for g in geometry.geoms if isinstance(g, Polygon)]
        if polygons:
            return max(polygons, key=lambda p: p.area)
        # Try multipolygons
        multipolygons = [g for g in geometry.geoms if isinstance(g, MultiPolygon)]
        if multipolygons:
            all_polys = []
            for mp in multipolygons:
                all_polys.extend(mp.geoms)
            return max(all_polys, key=lambda p: p.area)
    # Fallback to empty polygon
    return Polygon()


def remove_holes(
    geometry: Union[Polygon, MultiPolygon],
    preserve_holes: bool
) -> Union[Polygon, MultiPolygon]:
    """Remove interior holes from geometry if preserve_holes is False.

    Args:
        geometry: Input Polygon or MultiPolygon
        preserve_holes: If False, removes all interior holes

    Returns:
        Geometry with holes removed (if preserve_holes=False)

    Examples:
        >>> poly_with_hole = Polygon(shell, [hole])
        >>> result = remove_holes(poly_with_hole, preserve_holes=False)
        >>> len(result.interiors)
        0
    """
    if preserve_holes:
        return geometry

    if isinstance(geometry, Polygon):
        if geometry.interiors:
            return Polygon(geometry.exterior)
        return geometry
    elif isinstance(geometry, MultiPolygon):
        return MultiPolygon([Polygon(p.exterior) for p in geometry.geoms])

    return geometry


def validate_and_fix(
    geometry: BaseGeometry,
    min_area: float = 0.0,
    return_largest_if_multi: bool = True
) -> BaseGeometry:
    """Validate geometry and attempt to fix if invalid.

    Uses the buffer(0) trick to fix invalid geometries. If the result
    is a MultiPolygon and return_largest_if_multi is True, returns only
    the largest piece.

    Args:
        geometry: Input geometry to validate/fix
        min_area: Minimum acceptable area (if > 0, checks area requirement)
        return_largest_if_multi: If True, converts MultiPolygon to largest Polygon

    Returns:
        Valid geometry (fixed if necessary)

    Examples:
        >>> invalid_poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])  # Bow-tie
        >>> result = validate_and_fix(invalid_poly)
        >>> result.is_valid
        True
    """
    # If already valid and meets area requirement, return as-is
    if geometry.is_valid:
        if min_area > 0 and hasattr(geometry, 'area') and geometry.area < min_area:
            return None
        return geometry

    # Try buffer(0) to fix
    try:
        fixed = geometry.buffer(0)

        # Handle MultiPolygon result
        if return_largest_if_multi and isinstance(fixed, MultiPolygon):
            fixed = max(fixed.geoms, key=lambda p: p.area)

        # Check validity and area
        if fixed.is_valid and not fixed.is_empty:
            if min_area > 0 and hasattr(fixed, 'area') and fixed.area < min_area:
                return None
            return fixed

    except GEOSException:
        pass

    return None


def update_coord_preserve_z(
    coords: np.ndarray,
    index: int,
    new_xy: np.ndarray
) -> None:
    """Update coordinate at index with new X,Y while preserving Z if present.

    Modifies coords array in-place.

    Args:
        coords: Coordinate array (Nx2 or Nx3)
        index: Index of coordinate to update
        new_xy: New 2D position [x, y]

    Examples:
        >>> coords = np.array([[0, 0, 10], [1, 1, 20], [2, 2, 30]])
        >>> update_coord_preserve_z(coords, 1, np.array([1.5, 1.5]))
        >>> coords[1]
        array([1.5, 1.5, 20])  # Z value preserved
    """
    if coords.shape[1] > 2:
        # 3D coordinates - preserve Z
        coords[index] = np.array([new_xy[0], new_xy[1], coords[index][2]])
    else:
        # 2D coordinates
        coords[index] = new_xy


def create_polygon_with_z_preserved(
    new_coords: np.ndarray,
    original_polygon: Polygon
) -> Polygon:
    """Create new polygon from coordinates, preserving holes from original.

    Args:
        new_coords: New exterior coordinates
        original_polygon: Original polygon (holes will be copied)

    Returns:
        New polygon with modified exterior and original holes

    Examples:
        >>> original = Polygon(shell, [hole1, hole2])
        >>> new_coords = modify_coordinates(original.exterior.coords)
        >>> result = create_polygon_with_z_preserved(new_coords, original)
        >>> len(result.interiors) == len(original.interiors)
        True
    """
    if original_polygon.interiors:
        holes = [list(interior.coords) for interior in original_polygon.interiors]
        return Polygon(new_coords, holes=holes)
    else:
        return Polygon(new_coords)


def calculate_internal_angles(
    geometry: Union[Polygon, LineString],
    degrees: bool = True
) -> List[float]:
    """Calculate internal angles at each vertex of a polygon or linestring.

    For a Polygon, calculates the internal angle at each vertex of the exterior ring.
    The internal angle is measured inside the polygon. For a counter-clockwise oriented
    polygon (standard for Shapely), the sum of internal angles should be (n-2) × 180°
    where n is the number of vertices.

    For a LineString, calculates the angle at each interior vertex (excluding endpoints).
    The angle is measured as the turning angle from one segment to the next.

    Args:
        geometry: Polygon or LineString to analyze
        degrees: If True, returns angles in degrees; if False, in radians (default: True)

    Returns:
        List of internal angles, one per vertex (excluding the closing vertex for polygons)

    Examples:
        >>> # Square with 90-degree corners
        >>> square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> angles = calculate_internal_angles(square)
        >>> angles
        [90.0, 90.0, 90.0, 90.0]

        >>> # Right triangle
        >>> triangle = Polygon([(0, 0), (1, 0), (0, 1)])
        >>> angles = calculate_internal_angles(triangle)
        >>> # Should have angles: 90°, 45°, 45°

        >>> # LineString with turn
        >>> line = LineString([(0, 0), (1, 0), (1, 1)])
        >>> angles = calculate_internal_angles(line)
        >>> angles
        [90.0]  # One angle at the interior vertex

    Notes:
        - For polygons, returns n angles where n is the number of vertices (excluding closing point)
        - For linestrings, returns n-2 angles (one per interior vertex)
        - Angles are always positive (0-360° or 0-2π)
        - For polygons, internal angles > 180° indicate reflex (concave) vertices
    """
    if isinstance(geometry, Polygon):
        coords = np.array(geometry.exterior.coords)
        # Exclude the closing vertex (last point == first point)
        coords = coords[:-1]
        calculate_for_closed = True
    elif isinstance(geometry, LineString):
        coords = np.array(geometry.coords)
        calculate_for_closed = False
    else:
        raise TypeError(f"Geometry must be Polygon or LineString, got {type(geometry)}")

    if len(coords) < 3:
        return []

    angles = []
    n = len(coords)

    if calculate_for_closed:
        # For closed polygon: calculate angle at each vertex
        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n

            # Vectors from current point to previous and next
            v1 = coords[prev_idx] - coords[i]
            v2 = coords[next_idx] - coords[i]

            # Calculate the internal angle
            angle = _calculate_angle_between_vectors(v1, v2, degrees=degrees)
            angles.append(angle)
    else:
        # For open linestring: calculate angle at interior vertices only
        for i in range(1, n - 1):
            # Vector from current to previous
            v1 = coords[i - 1] - coords[i]
            # Vector from current to next
            v2 = coords[i + 1] - coords[i]

            # Calculate the angle
            angle = _calculate_angle_between_vectors(v1, v2, degrees=degrees)
            angles.append(angle)

    return angles


def _calculate_angle_between_vectors(
    v1: np.ndarray,
    v2: np.ndarray,
    degrees: bool = True
) -> float:
    """Calculate the angle between two vectors.

    The angle is measured from v1 to v2 in the counter-clockwise direction.
    Returns a value in [0, 360°] or [0, 2π].

    Args:
        v1: First vector (2D or 3D, only first 2 components used)
        v2: Second vector (2D or 3D, only first 2 components used)
        degrees: If True, return angle in degrees; if False, in radians

    Returns:
        Angle between vectors in the range [0, 360°] or [0, 2π]
    """
    # Extract 2D components
    v1_2d = v1[:2]
    v2_2d = v2[:2]

    # Normalize vectors
    v1_len = np.linalg.norm(v1_2d)
    v2_len = np.linalg.norm(v2_2d)

    if v1_len < 1e-10 or v2_len < 1e-10:
        # Degenerate case: zero-length vector
        return 0.0

    v1_norm = v1_2d / v1_len
    v2_norm = v2_2d / v2_len

    # Calculate angle using atan2 for proper quadrant handling
    # atan2(y, x) gives the angle of vector (x, y) from the positive x-axis
    angle1 = np.arctan2(v1_norm[1], v1_norm[0])
    angle2 = np.arctan2(v2_norm[1], v2_norm[0])

    # Calculate the angle from v1 to v2 (counter-clockwise)
    angle = angle2 - angle1

    # Normalize to [0, 2π]
    if angle < 0:
        angle += 2 * np.pi

    # For polygon internal angles, we want the angle measured "inside" the polygon
    # For a counter-clockwise polygon, the internal angle is the complement
    # Actually, we need to measure the angle the "other way"
    # The internal angle is 2π - angle (or 360° - angle)
    angle = 2 * np.pi - angle

    if degrees:
        angle = np.degrees(angle)

    return float(angle)


__all__ = [
    'to_single_polygon',
    'remove_holes',
    'validate_and_fix',
    'update_coord_preserve_z',
    'create_polygon_with_z_preserved',
    'calculate_internal_angles',
]
