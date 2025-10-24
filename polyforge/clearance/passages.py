"""Functions for fixing narrow passages and close edges.

This module provides functions to handle narrow passages (hourglass shapes),
near self-intersections, and parallel edges that run too close together.
"""

from typing import Union, List
from shapely.geometry import Polygon, MultiPolygon, LinearRing
import numpy as np
import shapely
import shapely.ops

from .utils import _find_nearest_vertex_index


def _validate_holes_after_buffer(
    buffered_exterior: LinearRing,
    original_holes: List[LinearRing],
    min_clearance: float
) -> List[List]:
    """Validate that holes are still valid after exterior has been buffered.

    When the exterior is buffered outward, original holes might:
    - Be too close to the new exterior (violating min_clearance)
    - Be partially or completely outside the new exterior
    - Create an invalid polygon

    Args:
        buffered_exterior: The new exterior ring after buffering
        original_holes: Original interior rings from input geometry
        min_clearance: Minimum required clearance

    Returns:
        List of hole coordinate lists that are still valid

    Note:
        Used by fix_near_self_intersection() when using buffer strategy.
    """
    if not original_holes:
        return []

    buffered_poly = Polygon(buffered_exterior)
    valid_holes = []

    for hole in original_holes:
        hole_poly = Polygon(hole)

        # Check 1: Hole must be completely inside the new exterior
        if not buffered_poly.contains(hole_poly):
            continue  # Skip hole - it's outside or intersects exterior

        # Check 2: Hole must be far enough from new exterior
        hole_ring = LinearRing(hole)
        exterior_ring = LinearRing(buffered_exterior)
        distance = float(hole_ring.distance(exterior_ring))

        if distance >= min_clearance:
            # Hole is valid - far enough from new exterior
            valid_holes.append(list(hole.coords))
        # else: Skip hole - too close to new exterior

    return valid_holes


def fix_narrow_passage(
    geometry: Polygon,
    min_clearance: float,
    strategy: str = 'widen'
) -> Union[Polygon, MultiPolygon]:
    """Fix narrow passages (hourglass/neck shapes) that cause low clearance.

    Narrow passages occur when a polygon has a thin section connecting two
    larger areas, creating an hourglass or dumbbell shape.

    Args:
        geometry: Input polygon
        min_clearance: Target minimum clearance
        strategy: How to fix the passage:
            - 'widen': Move vertices apart at narrow point to widen passage (default)
            - 'split': Split into separate polygons at narrow point

    Returns:
        Fixed geometry (Polygon if widened, MultiPolygon if split)

    Note:
        The 'widen' strategy uses minimum_clearance_line to identify the
        narrow point and moves the nearest vertices apart along that line.
        This preserves the overall polygon shape better than buffering.

    """
    if strategy == 'split':
        # Split at the narrow point
        clearance_line = shapely.minimum_clearance_line(geometry)

        if not clearance_line.is_empty:
            # Extend the line slightly to ensure it cuts through
            from shapely.affinity import scale
            extended_line = scale(clearance_line, xfact=1.5, yfact=1.5)

            try:
                result = shapely.ops.split(geometry, extended_line)
                if result.is_valid and not result.is_empty:
                    return result
            except Exception:
                pass

        # If split failed, return original
        return geometry

    else:  # 'widen' strategy
        # Move vertices apart at the narrow passage to widen it
        # This preserves the overall polygon shape better than buffering
        result = geometry

        for iteration in range(10):  # Max 10 iterations
            current_clearance = result.minimum_clearance

            if current_clearance >= min_clearance:
                return result

            # Get the minimum clearance line - connects the two closest points
            try:
                clearance_line = shapely.minimum_clearance_line(result)
            except Exception:
                return result

            if clearance_line is None or clearance_line.is_empty:
                return result

            # Get the two endpoints of the clearance line
            coords_list = list(clearance_line.coords)
            if len(coords_list) != 2:
                return result

            pt1 = np.array(coords_list[0])
            pt2 = np.array(coords_list[1])

            # Calculate clearance direction
            clearance_vector = pt2 - pt1
            clearance_distance = np.linalg.norm(clearance_vector)

            if clearance_distance < 1e-10:
                return result

            clearance_direction = clearance_vector / clearance_distance

            # Find the vertices closest to these two points
            coords = np.array(result.exterior.coords)

            vertex_idx_1 = _find_nearest_vertex_index(coords, pt1)
            vertex_idx_2 = _find_nearest_vertex_index(coords, pt2)

            # Calculate movement distance (each vertex moves half the required distance)
            required_increase = min_clearance - current_clearance
            movement_distance = required_increase * 0.55  # 55% = half + 10% buffer

            # Create new coordinates
            new_coords = coords.copy()

            # Move vertex 1 away from vertex 2
            vertex_1 = coords[vertex_idx_1][:2]
            new_position_1 = vertex_1 - clearance_direction[:2] * movement_distance

            if coords.shape[1] > 2:
                new_coords[vertex_idx_1] = np.array([new_position_1[0], new_position_1[1], coords[vertex_idx_1][2]])
            else:
                new_coords[vertex_idx_1] = new_position_1

            # Move vertex 2 away from vertex 1
            vertex_2 = coords[vertex_idx_2][:2]
            new_position_2 = vertex_2 + clearance_direction[:2] * movement_distance

            if coords.shape[1] > 2:
                new_coords[vertex_idx_2] = np.array([new_position_2[0], new_position_2[1], coords[vertex_idx_2][2]])
            else:
                new_coords[vertex_idx_2] = new_position_2

            # Update closing vertex if needed
            if vertex_idx_1 == 0 or vertex_idx_2 == 0:
                new_coords[-1] = new_coords[0]

            # Create new polygon (preserve holes if present)
            try:
                if len(result.interiors) > 0:
                    holes = [list(interior.coords) for interior in result.interiors]
                    new_poly = Polygon(new_coords, holes=holes)
                else:
                    new_poly = Polygon(new_coords)

                # Validate result
                if not new_poly.is_valid:
                    new_poly = new_poly.buffer(0)
                    if new_poly.geom_type == 'MultiPolygon':
                        new_poly = max(new_poly.geoms, key=lambda p: p.area)

                if (new_poly.is_valid and
                    not new_poly.is_empty and
                    new_poly.geom_type == 'Polygon' and
                    new_poly.area > geometry.area * 0.5):

                    new_clearance = new_poly.minimum_clearance

                    # Only accept if clearance improved
                    if new_clearance > current_clearance:
                        result = new_poly
                    else:
                        return result
                else:
                    return result

            except Exception:
                return result

        return result


def fix_near_self_intersection(
    geometry: Polygon,
    min_clearance: float,
    strategy: str = 'simplify'
) -> Polygon:
    """Fix near self-intersections where edges come very close.

    Near self-intersections occur when edges or vertices come very close
    to each other without actually touching, creating low clearance values.

    Args:
        geometry: Input polygon
        min_clearance: Target minimum clearance
        strategy: How to fix the issue:
            - 'simplify': Remove vertices causing near-intersections (default)
            - 'buffer': Use small buffer to smooth edges apart
            - 'smooth': Apply smoothing to separate close edges

    Returns:
        Fixed polygon with improved clearance

    Examples:
        >>> # Polygon with edges that come very close
        >>> coords = [(0, 0), (5, 0), (5, 5), (2, 2.1), (2, 1.9), (0, 5)]
        >>> poly = Polygon(coords)
        >>> fixed = fix_near_self_intersection(poly, min_clearance=0.5)
        >>> fixed.is_valid
        True
    """
    if strategy == 'buffer':
        # Use small buffer to push edges apart
        current_clearance = geometry.minimum_clearance

        if current_clearance >= min_clearance:
            return geometry

        buffer_dist = (min_clearance - current_clearance) / 2 + 0.01

        # Try progressive buffer sizes if initial buffer insufficient
        for multiplier in [1.0, 1.5, 2.0, 3.0]:
            try:
                buffered = geometry.buffer(buffer_dist * multiplier)

                if isinstance(buffered, Polygon) and buffered.is_valid:
                    # Validate and filter holes (added in Phase 1.3)
                    if len(geometry.interiors) > 0:
                        valid_holes = _validate_holes_after_buffer(
                            buffered.exterior,
                            list(geometry.interiors),
                            min_clearance
                        )
                        buffered = Polygon(buffered.exterior, holes=valid_holes)

                    # Validate that buffering achieved target clearance
                    if buffered.is_valid and buffered.minimum_clearance >= min_clearance:
                        return buffered
                    # else: try next multiplier

            except Exception:
                continue  # Try next multiplier

        # Could not achieve target clearance
        return geometry

    else:  # 'simplify' or 'smooth' - both use progressive simplification
        from polyforge.simplify import simplify_rdp

        result = geometry
        best_result = geometry
        best_clearance = geometry.minimum_clearance

        # Use gentler epsilon for smoothing
        if strategy == 'smooth':
            base_epsilon = min_clearance / 3
        else:
            base_epsilon = min_clearance / 2

        # Try progressive simplification
        for iteration in range(5):
            current_clearance = result.minimum_clearance

            if current_clearance >= min_clearance:
                return result

            epsilon = base_epsilon * (1.5 ** iteration)
            epsilon = min(epsilon, result.length / 12)

            try:
                simplified = simplify_rdp(result, epsilon=epsilon)

                if (simplified.is_valid and
                    not simplified.is_empty and
                    simplified.area > geometry.area * 0.8 and
                    len(simplified.exterior.coords) >= 4):

                    new_clearance = simplified.minimum_clearance

                    if new_clearance > best_clearance:
                        best_result = simplified
                        best_clearance = new_clearance
                        result = simplified
                    elif iteration > 0:
                        break

            except Exception:
                break

        return best_result


def fix_parallel_close_edges(
    geometry: Polygon,
    min_clearance: float,
    strategy: str = 'simplify'
) -> Polygon:
    """Fix parallel edges that run too close to each other.
    """
    # Parallel close edges are essentially a type of near-self-intersection
    # We can reuse the same fixing logic
    return fix_near_self_intersection(geometry, min_clearance, strategy)
