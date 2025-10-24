"""Functions for fixing narrow protrusions and sharp intrusions.

This module provides functions to handle thin spikes, peninsulas, and
sharp indentations that create low minimum clearance.
"""

import numpy as np
from shapely.geometry import Polygon

from .utils import _point_to_segment_distance


def fix_narrow_protrusion(
    geometry: Polygon,
    min_clearance: float,
    max_iterations: int = 5
) -> Polygon:
    """Fix narrow protrusions by removing spike vertices.

    Narrow protrusions are thin spikes or peninsulas extending from the
    main polygon body. They create low clearance at their base.

    Args:
        geometry: Input polygon
        min_clearance: Target minimum clearance
        max_iterations: Maximum iterations to attempt (default 5)

    Returns:
        Polygon with narrow protrusions removed

    Examples:
        >>> # Polygon with narrow spike
        >>> coords = [(0, 0), (10, 0), (10, 5), (11, 10), (10, 5.1),
        ...           (10, 10), (0, 10)]
        >>> poly = Polygon(coords)
        >>> fixed = fix_narrow_protrusion(poly, min_clearance=1.0)
        >>> fixed.is_valid
        True
    """
    from polyforge.simplify import simplify_rdp

    result = geometry
    best_result = geometry
    best_clearance = geometry.minimum_clearance

    # Try progressive simplification with increasing tolerance
    for iteration in range(max_iterations):
        current_clearance = result.minimum_clearance

        if current_clearance >= min_clearance:
            return result

        # Calculate appropriate epsilon based on iteration
        # Start with min_clearance/2 and increase progressively
        base_epsilon = min_clearance / 2
        epsilon = base_epsilon * (2.0 ** iteration)

        # Safety limit: don't simplify too aggressively
        epsilon = min(epsilon, result.length / 10)

        try:
            simplified = simplify_rdp(result, epsilon=epsilon)

            if (simplified.is_valid and
                not simplified.is_empty and
                simplified.area > geometry.area * 0.5 and
                len(simplified.exterior.coords) >= 4):

                new_clearance = simplified.minimum_clearance

                # Only accept if clearance improved
                if new_clearance > best_clearance:
                    best_result = simplified
                    best_clearance = new_clearance
                    result = simplified
                elif iteration > 0:
                    # Simplification didn't help, stop trying
                    break

        except Exception:
            # Simplification failed
            break

    return best_result


def fix_sharp_intrusion(
    geometry: Polygon,
    min_clearance: float,
    strategy: str = 'fill',
    max_iterations: int = 5
) -> Polygon:
    """Fix sharp narrow intrusions by filling or smoothing.

    Sharp intrusions are deep, narrow indentations or coves in the
    polygon boundary that create low clearance.

    Args:
        geometry: Input polygon
        min_clearance: Target minimum clearance
        strategy: How to fix intrusions:
            - 'fill': Fill intrusion with straight edge (default)
            - 'smooth': Apply smoothing to widen intrusion
            - 'simplify': Use vertex simplification
        max_iterations: Maximum iterations to attempt

    Returns:
        Polygon with sharp intrusions fixed

    Examples:
        >>> # Polygon with narrow intrusion
        >>> coords = [(0, 0), (10, 0), (10, 10), (5, 5), (5, 4.9),
        ...           (0, 10)]
        >>> poly = Polygon(coords)
        >>> fixed = fix_sharp_intrusion(poly, min_clearance=1.0)
        >>> fixed.is_valid
        True
    """
    from polyforge.simplify import simplify_rdp

    # All strategies use progressive simplification
    # 'fill' and 'simplify' are effectively the same (remove vertices)
    # 'smooth' tries gentler simplification first

    result = geometry
    best_result = geometry
    best_clearance = geometry.minimum_clearance

    # Adjust base epsilon based on strategy
    if strategy == 'smooth':
        base_epsilon = min_clearance / 3  # Gentler
    else:  # 'fill' or 'simplify'
        base_epsilon = min_clearance / 2

    # Try progressive simplification with increasing tolerance
    for iteration in range(max_iterations):
        current_clearance = result.minimum_clearance

        if current_clearance >= min_clearance:
            return result

        # Calculate epsilon for this iteration
        epsilon = base_epsilon * (2.0 ** iteration)

        # Safety limit
        epsilon = min(epsilon, result.length / 10)

        try:
            simplified = simplify_rdp(result, epsilon=epsilon)

            if (simplified.is_valid and
                not simplified.is_empty and
                simplified.area >= geometry.area * 0.9 and  # Allow small area increase for intrusions
                len(simplified.exterior.coords) >= 4):

                new_clearance = simplified.minimum_clearance

                # Only accept if clearance improved
                if new_clearance > best_clearance:
                    best_result = simplified
                    best_clearance = new_clearance
                    result = simplified
                elif iteration > 0:
                    # Simplification didn't help, stop trying
                    break

        except Exception:
            # Simplification failed
            break

    return best_result


def _walk_to_width_threshold(
    coords: np.ndarray,
    start_idx: int,
    direction: int,
    min_width: float
) -> int:
    """Walk along polygon boundary until width exceeds threshold.

    Args:
        coords: Exterior coordinates
        start_idx: Starting vertex index
        direction: -1 for backward, 1 for forward
        min_width: Minimum width to find

    Returns:
        Index where width exceeds min_width
    """
    n = len(coords) - 1  # Exclude closing vertex
    current_idx = start_idx
    steps = 0
    max_steps = n // 2  # Safety limit

    while steps < max_steps:
        current_idx = (current_idx + direction) % n

        # Measure width at this point
        width = _measure_local_width(coords, current_idx)

        if width >= min_width:
            return current_idx

        steps += 1

    # Safety fallback: return point 1/4 way around polygon
    return (start_idx + direction * (n // 4)) % n


def _measure_local_width(coords: np.ndarray, idx: int) -> float:
    """Measure the width of polygon at a given vertex.

    Width = shortest distance from vertex to non-adjacent edge.

    Args:
        coords: Polygon coordinates
        idx: Vertex index

    Returns:
        Local width measurement
    """
    vertex = coords[idx][:2] if coords.shape[1] > 2 else coords[idx]
    n = len(coords) - 1
    min_dist = float('inf')

    # Check distance to all non-adjacent edges
    for i in range(n):
        # Skip adjacent edges (within 2 vertices)
        if abs(i - idx) <= 2 or abs(i - idx) >= n - 2:
            continue

        edge_start = coords[i][:2] if coords.shape[1] > 2 else coords[i]
        edge_end = coords[(i + 1) % n][:2] if coords.shape[1] > 2 else coords[(i + 1) % n]

        dist = _point_to_segment_distance(vertex, edge_start, edge_end)
        min_dist = min(min_dist, dist)

    return min_dist


def _smooth_intrusion(
    coords: np.ndarray,
    start_idx: int,
    end_idx: int,
    target_width: float
) -> np.ndarray:
    """Apply smoothing to intrusion vertices to widen opening.

    Args:
        coords: Polygon coordinates
        start_idx: Start of intrusion
        end_idx: End of intrusion
        target_width: Target width

    Returns:
        Smoothed coordinates
    """
    # Identify vertices in the intrusion
    if start_idx < end_idx:
        intrusion_indices = list(range(start_idx, end_idx + 1))
    else:
        n = len(coords) - 1
        intrusion_indices = list(range(start_idx, n)) + list(range(0, end_idx + 1))

    new_coords = coords.copy()

    # Apply Gaussian smoothing
    window_size = len(intrusion_indices)
    if window_size < 3:
        return coords  # Too few vertices to smooth

    # Create smoothed coordinates using moving average
    for i, idx in enumerate(intrusion_indices):
        # Weighted average with neighbors
        if i == 0 or i == len(intrusion_indices) - 1:
            continue  # Keep endpoints fixed

        # Average with neighbors
        prev_idx = intrusion_indices[i - 1]
        next_idx = intrusion_indices[i + 1]

        # Weighted average (more weight on current)
        new_coords[idx] = (coords[prev_idx] + 2 * coords[idx] + coords[next_idx]) / 4

    return new_coords
