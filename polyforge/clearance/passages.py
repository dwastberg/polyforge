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
from polyforge.core.types import PassageStrategy, IntersectionStrategy, EdgeStrategy


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


def _is_point_near_vertex(
    point: np.ndarray,
    coords: np.ndarray,
    threshold: float
) -> tuple[bool, int]:
    """Check if a point is close to any existing vertex.

    Used to determine if a clearance line endpoint is at a vertex
    or on an edge between vertices.

    Args:
        point: Point to check (from minimum_clearance_line)
        coords: Array of polygon coordinates
        threshold: Distance threshold (typically 5% of min_clearance)

    Returns:
        Tuple of (is_near_vertex: bool, vertex_index: int)
        If is_near_vertex is True, vertex_index is the nearby vertex
        If is_near_vertex is False, vertex_index is the nearest vertex
    """
    min_dist = float('inf')
    nearest_idx = -1

    for i in range(len(coords) - 1):  # Exclude closing vertex
        dist = np.linalg.norm(coords[i][:2] - point[:2])
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i

    is_near = min_dist < threshold
    return is_near, nearest_idx


def _calculate_edge_perpendicular(
    vertex_pos: np.ndarray,
    prev_vertex: np.ndarray,
    next_vertex: np.ndarray,
    away_from_point: np.ndarray
) -> np.ndarray:
    """Calculate perpendicular direction from an edge, pointing away from a point.

    When a clearance line endpoint falls on an edge (not at a vertex),
    we need to move the nearest vertex on that edge perpendicular to the edge,
    pushing the entire edge away from the opposite clearance point.

    Args:
        vertex_pos: Position of vertex to move (on the edge)
        prev_vertex: Previous vertex defining the edge
        next_vertex: Next vertex defining the edge
        away_from_point: Point to move away from (opposite clearance endpoint)

    Returns:
        Normalized perpendicular direction vector (2D)
    """
    # Calculate edge direction
    edge_vec = next_vertex[:2] - prev_vertex[:2]
    edge_length = np.linalg.norm(edge_vec)

    if edge_length < 1e-10:
        # Degenerate edge, fall back to direct away direction
        away_vec = vertex_pos[:2] - away_from_point[:2]
        away_length = np.linalg.norm(away_vec)
        if away_length < 1e-10:
            return np.array([1.0, 0.0])
        return away_vec / away_length

    edge_dir = edge_vec / edge_length

    # Calculate perpendicular to edge (rotate 90 degrees)
    # Two perpendicular directions: (dy, -dx) and (-dy, dx)
    perp1 = np.array([edge_dir[1], -edge_dir[0]])
    perp2 = np.array([-edge_dir[1], edge_dir[0]])

    # Choose perpendicular that points away from the opposite point
    to_away = away_from_point[:2] - vertex_pos[:2]

    # Use dot product to determine which perpendicular points away
    if np.dot(perp1, to_away) < 0:
        return perp1
    else:
        return perp2


def fix_narrow_passage(
    geometry: Polygon,
    min_clearance: float,
    strategy: PassageStrategy = PassageStrategy.WIDEN
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
        narrow point and moves the nearest vertices apart. This preserves
        the overall polygon shape better than buffering.

        The algorithm handles multiple clearance configurations:
        - Vertex-to-vertex: moves both vertices along clearance line
        - Vertex-to-edge: moves vertex along clearance line, edge vertex perpendicular
        - Same vertex (vertex to adjacent edge): moves vertex perpendicular to edge

        Uses 5% of min_clearance as threshold to detect edge vs vertex cases.
        Special handling for degenerate cases where both clearance points map to
        the same vertex (clearance from vertex to its adjacent edge).

    """
    if strategy == PassageStrategy.SPLIT:
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

            # Check if clearance points are at vertices or on edges
            # Use 5% of min_clearance as threshold
            threshold = min_clearance * 0.05
            is_near_1, vertex_idx_1 = _is_point_near_vertex(pt1, coords, threshold)
            is_near_2, vertex_idx_2 = _is_point_near_vertex(pt2, coords, threshold)

            # Special case: both clearance points map to the same vertex
            # This happens when clearance is from a vertex to an adjacent edge
            if vertex_idx_1 == vertex_idx_2:
                # Only move the vertex based on pt2's position (on the edge)
                # Use perpendicular movement from the adjacent edge
                n = len(coords) - 1  # Exclude closing vertex
                prev_idx = (vertex_idx_2 - 1) % n
                next_idx = (vertex_idx_2 + 1) % n

                # Move vertex perpendicular to its adjacent edge, away from pt2
                direction = _calculate_edge_perpendicular(
                    coords[vertex_idx_2],
                    coords[prev_idx],
                    coords[next_idx],
                    pt2
                )

                required_increase = min_clearance - current_clearance
                movement_distance = required_increase * 1.1  # Move full distance + buffer

                new_coords = coords.copy()
                vertex_pos = coords[vertex_idx_2][:2]
                new_position = vertex_pos + direction * movement_distance

                if coords.shape[1] > 2:
                    new_coords[vertex_idx_2] = np.array([new_position[0], new_position[1], coords[vertex_idx_2][2]])
                else:
                    new_coords[vertex_idx_2] = new_position

                # Update closing vertex if needed
                if vertex_idx_2 == 0:
                    new_coords[-1] = new_coords[0]

                # Create new polygon and continue to validation
                try:
                    if len(result.interiors) > 0:
                        holes = [list(interior.coords) for interior in result.interiors]
                        new_poly = Polygon(new_coords, holes=holes)
                    else:
                        new_poly = Polygon(new_coords)

                    if not new_poly.is_valid:
                        new_poly = new_poly.buffer(0)
                        if new_poly.geom_type == 'MultiPolygon':
                            new_poly = max(new_poly.geoms, key=lambda p: p.area)

                    if (new_poly.is_valid and
                        not new_poly.is_empty and
                        new_poly.geom_type == 'Polygon' and
                        new_poly.area > geometry.area * 0.5):

                        new_clearance = new_poly.minimum_clearance
                        if new_clearance > current_clearance:
                            result = new_poly
                            continue
                        else:
                            return result
                    else:
                        return result
                except Exception:
                    return result

            # Calculate movement distance (each vertex moves half the required distance)
            required_increase = min_clearance - current_clearance
            movement_distance = required_increase * 0.55  # 55% = half + 10% buffer

            # Determine movement direction for vertex 1
            if is_near_1:
                # pt1 is at a vertex - move along clearance line
                direction_1 = -clearance_direction[:2]
            else:
                # pt1 is on an edge - move nearest vertex perpendicular to edge
                n = len(coords) - 1  # Exclude closing vertex
                prev_idx = (vertex_idx_1 - 1) % n
                next_idx = (vertex_idx_1 + 1) % n
                direction_1 = _calculate_edge_perpendicular(
                    coords[vertex_idx_1],
                    coords[prev_idx],
                    coords[next_idx],
                    pt2  # Move away from opposite clearance point
                )

            # Determine movement direction for vertex 2
            if is_near_2:
                # pt2 is at a vertex - move along clearance line
                direction_2 = clearance_direction[:2]
            else:
                # pt2 is on an edge - move nearest vertex perpendicular to edge
                n = len(coords) - 1  # Exclude closing vertex
                prev_idx = (vertex_idx_2 - 1) % n
                next_idx = (vertex_idx_2 + 1) % n
                direction_2 = _calculate_edge_perpendicular(
                    coords[vertex_idx_2],
                    coords[prev_idx],
                    coords[next_idx],
                    pt1  # Move away from opposite clearance point
                )

            # Create new coordinates
            new_coords = coords.copy()

            # Move vertex 1 in calculated direction
            vertex_1 = coords[vertex_idx_1][:2]
            new_position_1 = vertex_1 + direction_1 * movement_distance

            if coords.shape[1] > 2:
                new_coords[vertex_idx_1] = np.array([new_position_1[0], new_position_1[1], coords[vertex_idx_1][2]])
            else:
                new_coords[vertex_idx_1] = new_position_1

            # Move vertex 2 in calculated direction
            vertex_2 = coords[vertex_idx_2][:2]
            new_position_2 = vertex_2 + direction_2 * movement_distance

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
    strategy: IntersectionStrategy = IntersectionStrategy.SIMPLIFY
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
    if strategy == IntersectionStrategy.BUFFER:
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
        if strategy == IntersectionStrategy.SMOOTH:
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
    strategy: IntersectionStrategy = IntersectionStrategy.SIMPLIFY
) -> Polygon:
    """Fix parallel edges that run too close to each other.
    """
    # Parallel close edges are essentially a type of near-self-intersection
    # We can reuse the same fixing logic
    return fix_near_self_intersection(geometry, min_clearance, strategy)


__all__ = [
    'fix_narrow_passage',
    'fix_near_self_intersection',
    'fix_parallel_close_edges',
]
