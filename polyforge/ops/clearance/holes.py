from __future__ import annotations

import math
import numpy as np
from shapely.geometry import Polygon, LinearRing, Point
from shapely.errors import GEOSException
import shapely.ops
from polyforge.core.types import HoleStrategy, coerce_enum

# Self-clearance that is below this fraction of sqrt(hole_area) indicates
# a near-self-intersection rather than just a small hole.
_SELF_CLEARANCE_RATIO = 0.1


def _has_self_clearance_issue(hole_ring: LinearRing, min_clearance: float) -> bool:
    """Check whether a hole ring has a near-self-intersection."""
    try:
        self_clearance = float(hole_ring.minimum_clearance)
    except (GEOSException, ValueError):
        return False

    if self_clearance >= min_clearance:
        return False

    # Only flag when self-clearance is disproportionately small
    # compared to the hole's overall size
    hole_area = Polygon(hole_ring).area
    if hole_area <= 0:
        return False
    return self_clearance < math.sqrt(hole_area) * _SELF_CLEARANCE_RATIO


def fix_hole_too_close(
    geometry: Polygon,
    min_clearance: float,
    strategy: HoleStrategy | str = HoleStrategy.REMOVE,
) -> Polygon:
    """Fix holes that cause low clearance.
    Args:
        geometry: Input polygon (possibly with holes)
        min_clearance: Target minimum clearance
        strategy: How to handle problematic holes:
            - 'remove': Remove holes that cause issues (default)
            - 'shrink': Make holes smaller via negative buffer
            - 'move': Move holes away from exterior (experimental,
              only applies to hole-to-exterior issues)

    Returns:
        Polygon with holes fixed
    """
    if not geometry.interiors:
        return geometry  # No holes to fix

    strategy_enum = coerce_enum(strategy, HoleStrategy)

    exterior = geometry.exterior
    good_holes = []

    for hole in geometry.interiors:
        hole_poly = Polygon(hole)
        hole_ring = LinearRing(hole.coords)

        # Check 1: hole-to-exterior distance
        distance = _calculate_hole_to_exterior_distance(hole_poly, exterior)
        too_close_to_exterior = distance < min_clearance

        # Check 2: hole near-self-intersection (disproportionately low
        # self-clearance relative to hole size)
        self_intersection = _has_self_clearance_issue(hole_ring, min_clearance)

        if not too_close_to_exterior and not self_intersection:
            good_holes.append(hole.coords)
            continue

        # Hole needs fixing
        if strategy_enum == HoleStrategy.REMOVE:
            continue  # Don't add to good_holes (removed)

        elif strategy_enum == HoleStrategy.SHRINK:
            if too_close_to_exterior:
                shrink_amount = min_clearance - distance
            else:
                # Self-clearance issue: buffer inward by half the deficit
                # (both sides of the pinch move inward)
                try:
                    self_cl = float(hole_ring.minimum_clearance)
                except (GEOSException, ValueError):
                    self_cl = 0.0
                shrink_amount = (min_clearance - self_cl) / 2

            shrunk_hole = hole_poly.buffer(-shrink_amount)

            if shrunk_hole.is_valid and not shrunk_hole.is_empty:
                if shrunk_hole.geom_type == "Polygon":
                    good_holes.append(shrunk_hole.exterior.coords)
            # else: hole shrunk to nothing, effectively removed

        elif strategy_enum == HoleStrategy.MOVE:
            if too_close_to_exterior:
                moved_hole = _move_hole_away_from_exterior(
                    hole_poly, exterior, min_clearance
                )
                if moved_hole is not None:
                    good_holes.append(moved_hole.exterior.coords)
            # else: moving doesn't help with self-clearance, hole is removed

    return Polygon(exterior.coords, holes=good_holes)


def _calculate_hole_to_exterior_distance(hole: Polygon, exterior: LinearRing) -> float:
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
    hole: Polygon, exterior: LinearRing, target_distance: float
) -> Polygon | None:
    """Move hole away from exterior to achieve target distance.

    Uses actual boundary distances (not centroids) to determine movement direction.
    Moves the hole perpendicular to the line connecting the closest boundary points.

    Args:
        hole: Hole polygon
        exterior: Exterior ring
        target_distance: Target distance from exterior

    Returns:
        Moved hole polygon, or None if move not possible
    """
    from shapely.affinity import translate

    closest = _closest_boundary_points(hole, exterior)
    if closest is None:
        return None

    pt_on_hole, pt_on_exterior, current_distance = closest
    if current_distance >= target_distance:
        return hole

    move_direction = _normalized_direction(pt_on_exterior, pt_on_hole)
    if move_direction is None:
        return None

    exterior_poly = Polygon(exterior)
    required_move = (target_distance - current_distance) * 1.1

    for multiplier in (1.0, 1.5, 2.0, 3.0):
        candidate = translate(
            hole,
            xoff=move_direction[0] * required_move * multiplier,
            yoff=move_direction[1] * required_move * multiplier,
        )
        if not exterior_poly.contains(candidate):
            continue
        if _calculate_hole_to_exterior_distance(candidate, exterior) >= target_distance:
            return candidate

    return None


def _closest_boundary_points(hole: Polygon, exterior: LinearRing):
    hole_ring = LinearRing(hole.exterior.coords)
    exterior_ring = LinearRing(exterior.coords)
    try:
        pt_on_hole, pt_on_exterior = shapely.ops.nearest_points(
            hole_ring, exterior_ring
        )
    except Exception:
        return None
    distance = pt_on_hole.distance(pt_on_exterior)
    return pt_on_hole, pt_on_exterior, distance


def _normalized_direction(source: Point, target: Point) -> np.ndarray | None:
    move_vec = np.array(target.coords[0]) - np.array(source.coords[0])
    move_dist = np.linalg.norm(move_vec)
    if move_dist < 1e-10:
        return None
    return move_vec / move_dist


__all__ = [
    "fix_hole_too_close",
]
