"""Functions for fixing holes too close to polygon exterior.

This module provides functions to handle holes (interior rings) that are
positioned too close to the exterior boundary, causing low minimum clearance.
"""

import numpy as np
from typing import Optional
from shapely.geometry import Polygon, LinearRing, Point
import shapely.ops


def fix_hole_too_close(
    geometry: Polygon,
    min_clearance: float,
    strategy: str = 'remove'
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
            if strategy == 'remove':
                # Don't add to good_holes (removed)
                continue

            elif strategy == 'shrink':
                # Shrink hole by buffering inward
                shrink_amount = min_clearance - distance
                shrunk_hole = hole_poly.buffer(-shrink_amount)

                if shrunk_hole.is_valid and not shrunk_hole.is_empty:
                    if shrunk_hole.geom_type == 'Polygon':
                        good_holes.append(shrunk_hole.exterior.coords)
                # else: hole shrunk to nothing, effectively removed

            elif strategy == 'move':
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

    This is complex and may not always succeed.

    Args:
        hole: Hole polygon
        exterior: Exterior ring
        target_distance: Target distance from exterior

    Returns:
        Moved hole polygon, or None if move not possible
    """
    from shapely.affinity import translate

    # Find nearest point on exterior to hole
    hole_centroid = hole.centroid
    exterior_poly = Polygon(exterior)

    # Find nearest point on exterior to centroid
    nearest_geom = shapely.ops.nearest_points(hole_centroid, exterior_poly.boundary)
    nearest_point = nearest_geom[1]

    # Calculate move direction (away from nearest point)
    move_direction = np.array(hole_centroid.coords[0]) - np.array(nearest_point.coords[0])
    move_norm = np.linalg.norm(move_direction)

    if move_norm == 0:
        return None  # Can't determine direction

    move_direction = move_direction / move_norm

    # Current distance
    current_distance = hole_centroid.distance(Point(nearest_point))

    # How far to move
    move_distance = target_distance - current_distance

    if move_distance <= 0:
        return hole  # Already far enough

    # Translate hole
    moved_hole = translate(
        hole,
        xoff=move_direction[0] * move_distance,
        yoff=move_direction[1] * move_distance
    )

    # Verify hole is still inside exterior
    if Polygon(exterior).contains(moved_hole):
        return moved_hole
    else:
        return None  # Can't move safely
