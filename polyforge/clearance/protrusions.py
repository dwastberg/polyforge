"""Functions for fixing narrow protrusions and sharp intrusions.

This module provides functions to handle thin spikes, peninsulas, and
sharp indentations that create low minimum clearance.
"""

import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points
import shapely

from .utils import _point_to_segment_distance, _find_nearest_vertex_index
from polyforge.core.types import IntrusionStrategy


def fix_narrow_protrusion(
    geometry: Polygon,
    min_clearance: float,
    max_iterations: int = 10
) -> Polygon:
    """Fix narrow protrusions by moving vertices to increase clearance.

    Uses the minimum_clearance_line to identify the two closest points that
    define the clearance bottleneck, then moves the vertices at those points
    apart along the clearance line to achieve the target clearance.

    Args:
        geometry: Input polygon
        min_clearance: Target minimum clearance
        max_iterations: Maximum iterations to attempt (default 10)

    Returns:
        Polygon with narrow protrusions fixed by moving vertices

    Examples:
        >>> # Polygon with narrow spike
        >>> coords = [(0, 0), (10, 0), (10, 5), (11, 10), (10, 5.1),
        ...           (10, 10), (0, 10)]
        >>> poly = Polygon(coords)
        >>> fixed = fix_narrow_protrusion(poly, min_clearance=1.0)
        >>> fixed.is_valid
        True
    """
    result = geometry

    for iteration in range(max_iterations):
        current_clearance = result.minimum_clearance

        if current_clearance >= min_clearance:
            return result

        # Get the minimum clearance line - this is a LineString connecting
        # the two points that define the minimum clearance
        try:
            clearance_line = shapely.minimum_clearance_line(result)
        except Exception:
            # No clearance line available (polygon may be too simple)
            return result

        if clearance_line is None or clearance_line.is_empty:
            return result

        # Get the two endpoints of the clearance line
        coords_list = list(clearance_line.coords)
        if len(coords_list) != 2:
            return result

        pt1 = np.array(coords_list[0])
        pt2 = np.array(coords_list[1])

        # The clearance line direction (from pt1 to pt2)
        clearance_vector = pt2 - pt1
        clearance_distance = np.linalg.norm(clearance_vector)

        if clearance_distance < 1e-10:
            return result

        # Normalize the clearance vector
        clearance_direction = clearance_vector / clearance_distance

        # Find the vertices closest to these two points
        coords = np.array(result.exterior.coords)
        n = len(coords) - 1  # Exclude closing vertex

        # Find nearest vertices using utility function
        vertex_idx_1 = _find_nearest_vertex_index(coords, pt1)
        vertex_idx_2 = _find_nearest_vertex_index(coords, pt2)

        # Calculate how much we need to move the vertices apart
        # We want to increase clearance from current to min_clearance
        required_increase = min_clearance - current_clearance
        # Add 10% buffer
        movement_distance = required_increase * 0.55  # Each vertex moves half the required distance

        # Create new coordinates
        new_coords = coords.copy()

        # Move vertex 1 away from vertex 2 (in negative clearance direction)
        vertex_1 = coords[vertex_idx_1][:2]
        new_position_1 = vertex_1 - clearance_direction[:2] * movement_distance

        if coords.shape[1] > 2:
            # Preserve Z coordinate
            new_coords[vertex_idx_1] = np.array([new_position_1[0], new_position_1[1], coords[vertex_idx_1][2]])
        else:
            new_coords[vertex_idx_1] = new_position_1

        # Move vertex 2 away from vertex 1 (in positive clearance direction)
        vertex_2 = coords[vertex_idx_2][:2]
        new_position_2 = vertex_2 + clearance_direction[:2] * movement_distance

        if coords.shape[1] > 2:
            # Preserve Z coordinate
            new_coords[vertex_idx_2] = np.array([new_position_2[0], new_position_2[1], coords[vertex_idx_2][2]])
        else:
            new_coords[vertex_idx_2] = new_position_2

        # Update closing vertex if needed
        if vertex_idx_1 == 0 or vertex_idx_2 == 0:
            new_coords[-1] = new_coords[0]

        # Create new polygon (preserve holes if present)
        try:
            if len(result.interiors) > 0:
                # Preserve holes
                holes = [list(interior.coords) for interior in result.interiors]
                new_poly = Polygon(new_coords, holes=holes)
            else:
                new_poly = Polygon(new_coords)

            # Check if result is valid
            if not new_poly.is_valid:
                # Try to fix using buffer(0)
                new_poly = new_poly.buffer(0)
                # If result is MultiPolygon, take largest piece
                if new_poly.geom_type == 'MultiPolygon':
                    new_poly = max(new_poly.geoms, key=lambda p: p.area)

            # Validate the result
            if (new_poly.is_valid and
                not new_poly.is_empty and
                new_poly.geom_type == 'Polygon' and
                new_poly.area > geometry.area * 0.5):  # Allow some area loss

                new_clearance = new_poly.minimum_clearance

                # Only accept if clearance improved
                if new_clearance > current_clearance:
                    result = new_poly
                else:
                    # Movement didn't help, stop trying
                    return result
            else:
                # Invalid result, return current
                return result

        except Exception:
            # Failed to create valid polygon
            return result

    return result


def fix_sharp_intrusion(
    geometry: Polygon,
    min_clearance: float,
    strategy: IntrusionStrategy = IntrusionStrategy.FILL,
    max_iterations: int = 10
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
        max_iterations: Maximum iterations to attempt (default: 10)

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
    if strategy == IntrusionStrategy.SMOOTH:
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


__all__ = [
    'fix_narrow_protrusion',
    'fix_sharp_intrusion',
]
