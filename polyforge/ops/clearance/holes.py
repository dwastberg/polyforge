"""Functions for fixing holes too close to polygon exterior.

This module provides functions to handle holes (interior rings) that are
positioned too close to the exterior boundary, causing low minimum clearance.
"""

import numpy as np
from typing import Optional, Union
from shapely.geometry import Polygon, LinearRing, Point
import shapely.ops
from polyforge.core.types import HoleStrategy, coerce_enum


def fix_hole_too_close(
    geometry: Polygon,
    min_clearance: float,
    strategy: Union[HoleStrategy, str] = HoleStrategy.REMOVE,
) -> Polygon:
    """Fix holes that are too close to the polygon exterior.

    Args:
        geometry: Input polygon (possibly with holes)
        min_clearance: Target minimum clearance
        strategy: How to handle close holes:
            - 'remove': Remove holes that are too close (default)
            - 'shrink': Make holes smaller via negative buffer
            - 'move': Move holes away from exterior (experimental)

    Returns:
        Polygon with holes fixed

    Examples:
        >>> exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        >>> hole = [(1, 1), (2, 1), (2, 2), (1, 2)]  # Very close to edge
        >>> poly = Polygon(exterior, holes=[hole])
        >>> fixed = fix_hole_too_close(poly, min_clearance=2.0)
        >>> len(fixed.interiors)  # Hole removed
        0
    """
    if not geometry.interiors:
        return geometry  # No holes to fix

    strategy_enum = coerce_enum(strategy, HoleStrategy)

    exterior = geometry.exterior
    good_holes = []

    for hole in geometry.interiors:
        hole_poly = Polygon(hole)

        # Calculate minimum distance from hole to exterior
        distance = _calculate_hole_to_exterior_distance(hole_poly, exterior)

        if distance >= min_clearance:
            # Hole is fine, keep it
            good_holes.append(hole.coords)
        else:
            # Hole is too close
            if strategy_enum == HoleStrategy.REMOVE:
                # Don't add to good_holes (removed)
                continue

            elif strategy_enum == HoleStrategy.SHRINK:
                # Shrink hole by buffering inward
                shrink_amount = min_clearance - distance
                shrunk_hole = hole_poly.buffer(-shrink_amount)

                if shrunk_hole.is_valid and not shrunk_hole.is_empty:
                    if shrunk_hole.geom_type == 'Polygon':
                        good_holes.append(shrunk_hole.exterior.coords)
                # else: hole shrunk to nothing, effectively removed

            elif strategy_enum == HoleStrategy.MOVE:
                # Move hole away from exterior
                moved_hole = _move_hole_away_from_exterior(
                    hole_poly, exterior, min_clearance
                )
                if moved_hole is not None:
                    good_holes.append(moved_hole.exterior.coords)

    return Polygon(exterior.coords, holes=good_holes)


def _calculate_hole_to_exterior_distance(
    hole: Polygon,
    exterior: LinearRing
) -> float:
    """Calculate minimum distance from hole to exterior boundary.

    Args:
        hole: Hole as a Polygon
        exterior: Exterior ring

    Returns:
        Minimum distance
    """
    hole_line = LinearRing(hole.exterior.coords)
    exterior_line = LinearRing(exterior.coords)

    return float(hole_line.distance(exterior_line))


def _move_hole_away_from_exterior(
    hole: Polygon,
    exterior: LinearRing,
    target_distance: float
) -> Optional[Polygon]:
    """Move hole away from exterior to achieve target distance.

    Uses actual boundary distances (not centroids) to determine movement direction.
    Moves the hole perpendicular to the line connecting the closest boundary points.

    Args:
        hole: Hole polygon
        exterior: Exterior ring
        target_distance: Target distance from exterior

    Returns:
        Moved hole polygon, or None if move not possible

    Note:
        Fixed in Phase 1.1: Now uses actual closest boundary points instead of
        centroids, providing correct movement direction for irregularly shaped holes.
    """
    from shapely.affinity import translate

    # Find the actual closest points between hole boundary and exterior boundary
    hole_ring = LinearRing(hole.exterior.coords)
    exterior_ring = LinearRing(exterior.coords)

    # Get closest points on both boundaries
    pt_on_hole, pt_on_exterior = shapely.ops.nearest_points(hole_ring, exterior_ring)

    # Calculate current minimum distance
    current_distance = pt_on_hole.distance(pt_on_exterior)

    if current_distance >= target_distance:
        return hole  # Already far enough

    # Calculate move direction: from exterior point toward hole point, then beyond
    move_vec = np.array(pt_on_hole.coords[0]) - np.array(pt_on_exterior.coords[0])
    move_dist = np.linalg.norm(move_vec)

    if move_dist < 1e-10:
        return None  # Points coincide, can't determine direction

    # Normalize move direction
    move_direction = move_vec / move_dist

    # Calculate required movement: need to increase distance from current to target
    required_move = target_distance - current_distance

    # Add small buffer to ensure we exceed target
    required_move *= 1.1

    # Translate hole away from exterior
    moved_hole = translate(
        hole,
        xoff=move_direction[0] * required_move,
        yoff=move_direction[1] * required_move
    )

    # Verify hole is still inside exterior
    exterior_poly = Polygon(exterior)
    if not exterior_poly.contains(moved_hole):
        return None  # Move would put hole outside exterior

    # Verify the move actually achieved the target distance
    new_distance = _calculate_hole_to_exterior_distance(moved_hole, exterior)

    if new_distance >= target_distance:
        return moved_hole
    else:
        # Move didn't achieve target (might happen with complex shapes)
        # Try with larger movement factor
        for multiplier in [1.5, 2.0, 3.0]:
            larger_move = translate(
                hole,
                xoff=move_direction[0] * required_move * multiplier,
                yoff=move_direction[1] * required_move * multiplier
            )

            if exterior_poly.contains(larger_move):
                new_distance = _calculate_hole_to_exterior_distance(larger_move, exterior)
                if new_distance >= target_distance:
                    return larger_move

        # Could not achieve target distance
        return None


__all__ = [
    'fix_hole_too_close',
]
