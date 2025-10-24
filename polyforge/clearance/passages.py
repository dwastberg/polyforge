"""Functions for fixing narrow passages and close edges.

This module provides functions to handle narrow passages (hourglass shapes),
near self-intersections, and parallel edges that run too close together.
"""

from typing import Union
from shapely.geometry import Polygon, MultiPolygon
import shapely
import shapely.ops


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
            - 'widen': Use buffering to widen the narrow section (default)
            - 'split': Split into separate polygons at narrow point

    Returns:
        Fixed geometry (Polygon if widened, MultiPolygon if split)

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
        # Use small buffer to widen narrow passages
        current_clearance = geometry.minimum_clearance

        if current_clearance >= min_clearance:
            return geometry

        # Calculate buffer distance needed
        buffer_dist = (min_clearance - current_clearance) / 2 + 0.01

        try:
            buffered = geometry.buffer(buffer_dist)

            if isinstance(buffered, Polygon) and buffered.is_valid:
                # Preserve holes if they exist
                if len(geometry.interiors) > 0:
                    buffered = Polygon(
                        buffered.exterior,
                        holes=[h.coords for h in geometry.interiors]
                    )
                return buffered

        except Exception:
            pass

        return geometry


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

        try:
            buffered = geometry.buffer(buffer_dist)

            if isinstance(buffered, Polygon) and buffered.is_valid:
                if len(geometry.interiors) > 0:
                    buffered = Polygon(
                        buffered.exterior,
                        holes=[h.coords for h in geometry.interiors]
                    )
                return buffered

        except Exception:
            pass

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
