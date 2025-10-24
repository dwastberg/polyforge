"""Geometry fixing and validation functions.

This module provides functions to detect and fix various types of invalid
geometries using multiple strategies.
"""

import warnings
from typing import Optional, List, Tuple, Union
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, Point, GeometryCollection,
    LinearRing
)
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity
import numpy as np


class GeometryFixError(Exception):
    """Raised when geometry cannot be fixed."""
    pass


def fix_geometry(
    geometry: BaseGeometry,
    strategy: str = 'auto',
    buffer_distance: float = 0.0,
    tolerance: float = 1e-10,
    verbose: bool = False
) -> BaseGeometry:
    """Attempt to fix invalid geometries using various strategies.

    This function identifies different types of geometry invalidity and applies
    appropriate fixing strategies. It can handle:
    - Self-intersections
    - Ring self-intersections
    - Duplicate vertices
    - Unclosed rings
    - Invalid coordinate sequences
    - Topology errors
    - Bow-tie polygons
    - Overlapping holes

    Args:
        geometry: The geometry to fix
        strategy: Fixing strategy to use:
            - 'auto': Automatically detect and fix (default)
            - 'buffer': Use buffer(0) trick
            - 'simplify': Simplify and rebuild
            - 'reconstruct': Reconstruct from scratch
            - 'strict': Only fix if guaranteed to preserve intent
        buffer_distance: Small buffer distance for buffer-based fixes (default: 0.0)
        tolerance: Tolerance for coordinate comparisons (default: 1e-10)
        verbose: Print diagnostic information (default: False)

    Returns:
        Fixed geometry (same type as input if possible)

    Raises:
        GeometryFixError: If geometry cannot be fixed

    Examples:
        >>> # Fix self-intersecting polygon
        >>> poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])  # Bow-tie
        >>> fixed = fix_geometry(poly)
        >>> fixed.is_valid
        True

        >>> # Fix with specific strategy
        >>> fixed = fix_geometry(poly, strategy='buffer')

    Notes:
        - Multiple strategies are tried if 'auto' is selected
        - Original geometry is returned if already valid
        - Some fixes may slightly modify geometry shape
    """
    # Quick check: if valid, return as-is
    if geometry.is_valid:
        if verbose:
            print("Geometry is already valid")
        return geometry

    if verbose:
        reason = explain_validity(geometry)
        print(f"Invalid geometry: {reason}")

    # Determine strategy
    if strategy == 'auto':
        return _auto_fix_geometry(geometry, buffer_distance, tolerance, verbose)
    elif strategy == 'buffer':
        return _fix_with_buffer(geometry, buffer_distance, verbose)
    elif strategy == 'simplify':
        return _fix_with_simplify(geometry, tolerance, verbose)
    elif strategy == 'reconstruct':
        return _fix_with_reconstruct(geometry, tolerance, verbose)
    elif strategy == 'strict':
        return _fix_strict(geometry, tolerance, verbose)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _auto_fix_geometry(
    geometry: BaseGeometry,
    buffer_distance: float,
    tolerance: float,
    verbose: bool
) -> BaseGeometry:
    """Automatically detect and fix geometry issues.

    Tries multiple strategies in order of preference:
    1. Clean coordinates (remove duplicates, close rings)
    2. Buffer(0) trick
    3. Simplification
    4. Reconstruction
    """
    geom_type = geometry.geom_type

    # Strategy 1: Clean coordinates
    if verbose:
        print("Trying strategy: Clean coordinates")
    try:
        cleaned = _clean_coordinates(geometry, tolerance)
        if cleaned.is_valid:
            if verbose:
                print("Fixed with coordinate cleaning")
            return cleaned
    except Exception as e:
        if verbose:
            print(f"   Failed: {e}")

    # Strategy 2: Buffer(0)
    if verbose:
        print("Trying strategy: Buffer(0)")
    try:
        buffered = _fix_with_buffer(geometry, buffer_distance, verbose)
        if buffered.is_valid:
            if verbose:
                print("   Fixed with buffer")
            return buffered
    except Exception as e:
        if verbose:
            print(f"   Failed: {e}")

    # Strategy 3: Simplify
    if verbose:
        print("Trying strategy: Simplify")
    try:
        simplified = _fix_with_simplify(geometry, tolerance, verbose)
        if simplified.is_valid:
            if verbose:
                print("   Fixed with simplification")
            return simplified
    except Exception as e:
        if verbose:
            print(f"   Failed: {e}")

    # Strategy 4: Reconstruct
    if verbose:
        print("Trying strategy: Reconstruct")
    try:
        reconstructed = _fix_with_reconstruct(geometry, tolerance, verbose)
        if reconstructed.is_valid:
            if verbose:
                print("   Fixed with reconstruction")
            return reconstructed
    except Exception as e:
        if verbose:
            print(f"   Failed: {e}")

    # All strategies failed
    raise GeometryFixError(
        f"Could not fix {geom_type}: {explain_validity(geometry)}"
    )


def _clean_coordinates(
    geometry: BaseGeometry,
    tolerance: float
) -> BaseGeometry:
    """Clean coordinate sequences: remove duplicates, close rings."""
    geom_type = geometry.geom_type

    if geom_type == 'Polygon':
        # Clean exterior
        exterior_coords = np.array(geometry.exterior.coords)
        clean_exterior = _remove_duplicate_coords(exterior_coords, tolerance)
        clean_exterior = _ensure_closed_ring(clean_exterior)

        # Clean holes
        clean_holes = []
        for hole in geometry.interiors:
            hole_coords = np.array(hole.coords)
            clean_hole = _remove_duplicate_coords(hole_coords, tolerance)
            clean_hole = _ensure_closed_ring(clean_hole)
            if len(clean_hole) >= 4:  # Valid ring needs at least 4 points
                clean_holes.append(clean_hole)

        return Polygon(clean_exterior, holes=clean_holes)

    elif geom_type == 'LineString':
        coords = np.array(geometry.coords)
        clean_coords = _remove_duplicate_coords(coords, tolerance)
        return LineString(clean_coords)

    elif geom_type == 'MultiPolygon':
        clean_polys = []
        for poly in geometry.geoms:
            try:
                clean_poly = _clean_coordinates(poly, tolerance)
                if clean_poly.is_valid and not clean_poly.is_empty:
                    clean_polys.append(clean_poly)
            except Exception:
                pass
        return MultiPolygon(clean_polys) if clean_polys else geometry

    return geometry


def _remove_duplicate_coords(coords: np.ndarray, tolerance: float) -> np.ndarray:
    """Remove consecutive duplicate coordinates."""
    if len(coords) < 2:
        return coords

    unique_coords = [coords[0]]
    for i in range(1, len(coords)):
        distance = np.linalg.norm(coords[i] - unique_coords[-1])
        if distance > tolerance:
            unique_coords.append(coords[i])

    return np.array(unique_coords)


def _ensure_closed_ring(coords: np.ndarray) -> np.ndarray:
    """Ensure coordinate ring is closed (first == last)."""
    if len(coords) < 3:
        return coords

    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0:1]])

    return coords


def _fix_with_buffer(
    geometry: BaseGeometry,
    buffer_distance: float,
    verbose: bool
) -> BaseGeometry:
    """Fix geometry using the buffer(0) trick.

    The buffer(0) operation often fixes many topology errors.
    """
    try:
        # Try buffer(0) first
        fixed = geometry.buffer(buffer_distance)

        # Handle MultiPolygon results
        if isinstance(fixed, MultiPolygon) and isinstance(geometry, Polygon):
            # Return largest piece
            fixed = max(fixed.geoms, key=lambda p: p.area)

        # Handle GeometryCollection
        if isinstance(fixed, GeometryCollection):
            # Extract polygons
            polygons = [g for g in fixed.geoms if g.geom_type == 'Polygon']
            if polygons:
                fixed = max(polygons, key=lambda p: p.area)
            else:
                raise GeometryFixError("Buffer produced no valid polygons")

        return fixed

    except Exception as e:
        raise GeometryFixError(f"Buffer fix failed: {e}")


def _fix_with_simplify(
    geometry: BaseGeometry,
    tolerance: float,
    verbose: bool
) -> BaseGeometry:
    """Fix geometry using simplification.

    Simplification can remove problematic vertices causing invalidity.
    """
    try:
        # Clean coordinates first
        cleaned = _clean_coordinates(geometry, tolerance)

        # Apply simplification with increasing tolerance
        for epsilon in [tolerance * 10, tolerance * 100, tolerance * 1000]:
            simplified = cleaned.simplify(epsilon, preserve_topology=True)
            if simplified.is_valid and not simplified.is_empty:
                return simplified

        # Last resort: non-topology-preserving simplification
        simplified = cleaned.simplify(tolerance * 1000, preserve_topology=False)
        if simplified.is_valid:
            return simplified

        raise GeometryFixError("Simplification did not produce valid geometry")

    except Exception as e:
        raise GeometryFixError(f"Simplify fix failed: {e}")


def _fix_with_reconstruct(
    geometry: BaseGeometry,
    tolerance: float,
    verbose: bool
) -> BaseGeometry:
    """Fix geometry by reconstructing from points.

    Uses convex hull or point-based reconstruction.
    """
    try:
        geom_type = geometry.geom_type

        if geom_type in ('Polygon', 'MultiPolygon'):
            # Try convex hull
            hull = geometry.convex_hull
            if hull.is_valid:
                return hull

        # Extract all coordinates and rebuild
        coords = _extract_all_coords(geometry)
        if len(coords) < 3:
            raise GeometryFixError("Not enough coordinates to reconstruct")

        # Build polygon from points
        from shapely.ops import unary_union
        from shapely.geometry import MultiPoint

        points = MultiPoint(coords)
        hull = points.convex_hull

        if hull.is_valid:
            return hull

        raise GeometryFixError("Reconstruction failed")

    except Exception as e:
        raise GeometryFixError(f"Reconstruct fix failed: {e}")


def _fix_strict(
    geometry: BaseGeometry,
    tolerance: float,
    verbose: bool
) -> BaseGeometry:
    """Apply only conservative fixes that preserve intent.

    Only applies coordinate cleaning and closing rings.
    """
    cleaned = _clean_coordinates(geometry, tolerance)

    if cleaned.is_valid:
        return cleaned

    raise GeometryFixError(
        "Strict mode: geometry cannot be fixed without aggressive changes"
    )


def _extract_all_coords(geometry: BaseGeometry) -> List[Tuple[float, ...]]:
    """Extract all coordinates from any geometry."""
    coords = []

    if hasattr(geometry, 'coords'):
        coords.extend(list(geometry.coords))
    elif hasattr(geometry, 'exterior'):
        coords.extend(list(geometry.exterior.coords))
        for interior in geometry.interiors:
            coords.extend(list(interior.coords))
    elif hasattr(geometry, 'geoms'):
        for geom in geometry.geoms:
            coords.extend(_extract_all_coords(geom))

    return coords


def diagnose_geometry(geometry: BaseGeometry) -> dict:
    """Diagnose geometry validity issues.

    Returns a dictionary with diagnostic information about the geometry.

    Args:
        geometry: Geometry to diagnose

    Returns:
        Dictionary with keys:
            - 'is_valid': bool
            - 'validity_message': str (from Shapely)
            - 'issues': list of detected issues
            - 'suggestions': list of suggested fixes

    Examples:
        >>> poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        >>> diagnosis = diagnose_geometry(poly)
        >>> diagnosis['is_valid']
        False
        >>> 'Self-intersection' in diagnosis['issues']
        True
    """
    issues = []
    suggestions = []

    # Check validity
    is_valid = geometry.is_valid
    validity_msg = explain_validity(geometry)

    if not is_valid:
        # Analyze validity message
        msg_lower = validity_msg.lower()

        if 'self-intersection' in msg_lower or 'self intersection' in msg_lower:
            issues.append('Self-intersection')
            suggestions.append('Try buffer(0) or simplification')

        if 'duplicate' in msg_lower:
            issues.append('Duplicate vertices')
            suggestions.append('Clean coordinates')

        if 'not closed' in msg_lower or 'unclosed' in msg_lower:
            issues.append('Unclosed ring')
            suggestions.append('Close coordinate rings')

        if 'ring' in msg_lower and 'invalid' in msg_lower:
            issues.append('Invalid ring')
            suggestions.append('Reconstruct ring geometry')

        if 'hole' in msg_lower:
            issues.append('Invalid hole')
            suggestions.append('Remove or fix interior rings')

        if 'spike' in msg_lower or 'collapse' in msg_lower:
            issues.append('Collapsed/spike geometry')
            suggestions.append('Simplification or buffer')

        if not issues:
            issues.append('Unknown validity issue')
            suggestions.append('Try auto-fix strategy')

    # Check for other potential issues
    if hasattr(geometry, 'exterior'):
        exterior_coords = list(geometry.exterior.coords)
        if len(exterior_coords) < 4:
            issues.append('Too few vertices')
            suggestions.append('Geometry may be degenerate')

        # Check for duplicate consecutive vertices
        for i in range(len(exterior_coords) - 1):
            if exterior_coords[i] == exterior_coords[i + 1]:
                issues.append('Consecutive duplicate vertices')
                suggestions.append('Clean coordinates')
                break

    return {
        'is_valid': is_valid,
        'validity_message': validity_msg,
        'issues': issues,
        'suggestions': suggestions,
        'geometry_type': geometry.geom_type,
        'is_empty': geometry.is_empty,
        'area': geometry.area if hasattr(geometry, 'area') else None,
    }


def batch_fix_geometries(
    geometries: List[BaseGeometry],
    strategy: str = 'auto',
    on_error: str = 'skip',
    verbose: bool = False
) -> Tuple[List[BaseGeometry], List[int]]:
    """Fix multiple geometries in batch.

    Args:
        geometries: List of geometries to fix
        strategy: Fixing strategy (see fix_geometry)
        on_error: What to do on error:
            - 'skip': Skip invalid geometries
            - 'keep': Keep original invalid geometry
            - 'raise': Raise exception
        verbose: Print progress information

    Returns:
        Tuple of (fixed_geometries, failed_indices)

    Examples:
        >>> geometries = [poly1, poly2, poly3]
        >>> fixed, failed = batch_fix_geometries(geometries)
        >>> print(f"Fixed {len(fixed)}, failed {len(failed)}")
    """
    fixed = []
    failed_indices = []

    for i, geom in enumerate(geometries):
        try:
            if verbose and i % 100 == 0:
                print(f"Processing geometry {i}/{len(geometries)}...")

            fixed_geom = fix_geometry(geom, strategy=strategy, verbose=False)
            fixed.append(fixed_geom)

        except (GeometryFixError, Exception) as e:
            if on_error == 'raise':
                raise
            elif on_error == 'keep':
                fixed.append(geom)
            else:  # skip
                failed_indices.append(i)

            if verbose:
                print(f"  Failed to fix geometry {i}: {e}")

    return fixed, failed_indices
