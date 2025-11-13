"""Functions for fixing narrow passages and close edges.

This module provides functions to handle narrow passages (hourglass shapes),
near self-intersections, and parallel edges that run too close together.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from shapely.geometry import Polygon, MultiPolygon, LinearRing
import numpy as np
import shapely
import shapely.ops

from .utils import _find_nearest_vertex_index
from polyforge.core.geometry_utils import safe_buffer_fix
from polyforge.core.types import (
    PassageStrategy,
    IntersectionStrategy,
    coerce_enum,
)


@dataclass
class SelfIntersectionContext:
    clearance: float
    point_a: np.ndarray
    point_b: np.ndarray
    vertex_idx_a: int
    vertex_idx_b: int


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


def _split_narrow_passage(
    geometry: Polygon
) -> Union[Polygon, MultiPolygon]:
    """Split a polygon at its narrowest passage.

    Args:
        geometry: Input polygon

    Returns:
        Split result (MultiPolygon if successful) or original polygon
    """
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


def _create_polygon_with_new_coords(
    new_coords: np.ndarray,
    original_geometry: Polygon,
    min_area: float
) -> Union[Polygon, None]:
    """Create and validate a polygon from modified coordinates.

    Args:
        new_coords: Modified coordinate array
        original_geometry: Original polygon (for preserving holes)
        min_area: Minimum acceptable area

    Returns:
        Valid polygon or None if validation fails
    """
    try:
        holes = [list(interior.coords) for interior in original_geometry.interiors] if original_geometry.interiors else None
        new_poly = Polygon(new_coords, holes=holes) if holes else Polygon(new_coords)
    except Exception:
        return None

    if not new_poly.is_valid or new_poly.is_empty:
        healed = safe_buffer_fix(new_poly, distance=0.0, return_largest=True)
        if healed is None:
            return None
        new_poly = healed

    if (
        new_poly.geom_type != 'Polygon'
        or not new_poly.is_valid
        or new_poly.is_empty
        or new_poly.area <= min_area
    ):
        return None

    return new_poly


def _move_single_vertex_perpendicular(
    coords: np.ndarray,
    vertex_idx: int,
    pt2: np.ndarray,
    movement_distance: float,
    original_geometry: Polygon,
    current_clearance: float
) -> Union[Polygon, None]:
    """Move a single vertex perpendicular to its adjacent edge.

    Used when both clearance points map to the same vertex
    (clearance from vertex to adjacent edge).

    Args:
        coords: Polygon coordinate array
        vertex_idx: Index of vertex to move
        pt2: Point on adjacent edge (to move away from)
        movement_distance: How far to move the vertex
        original_geometry: Original polygon for validation
        current_clearance: Current minimum clearance

    Returns:
        New polygon or None if unsuccessful
    """
    n = len(coords) - 1  # Exclude closing vertex
    prev_idx = (vertex_idx - 1) % n
    next_idx = (vertex_idx + 1) % n

    # Calculate perpendicular direction away from pt2
    direction = _calculate_edge_perpendicular(
        coords[vertex_idx],
        coords[prev_idx],
        coords[next_idx],
        pt2
    )

    # Move vertex in calculated direction
    new_coords = coords.copy()
    vertex_pos = coords[vertex_idx][:2]
    new_position = vertex_pos + direction * movement_distance

    # Update vertex (preserve Z if 3D)
    if coords.shape[1] > 2:
        new_coords[vertex_idx] = np.array([new_position[0], new_position[1], coords[vertex_idx][2]])
    else:
        new_coords[vertex_idx] = new_position

    # Update closing vertex if needed
    if vertex_idx == 0:
        new_coords[-1] = new_coords[0]

    # Create and validate polygon
    min_area = original_geometry.area * 0.5
    new_poly = _create_polygon_with_new_coords(new_coords, original_geometry, min_area)

    if new_poly is not None:
        # Only return if clearance improved
        new_clearance = new_poly.minimum_clearance
        if new_clearance > current_clearance:
            return new_poly

    return None


def _determine_movement_direction(
    is_at_vertex: bool,
    clearance_direction: np.ndarray,
    coords: np.ndarray,
    vertex_idx: int,
    opposite_clearance_pt: np.ndarray
) -> np.ndarray:
    """Determine direction to move a vertex for passage widening.

    Args:
        is_at_vertex: Whether clearance point is at the vertex
        clearance_direction: Normalized clearance line direction
        coords: Polygon coordinate array
        vertex_idx: Index of vertex to move
        opposite_clearance_pt: Opposite endpoint of clearance line

    Returns:
        Normalized movement direction vector
    """
    if is_at_vertex:
        # Point is at vertex - move along clearance line
        return clearance_direction[:2]
    else:
        # Point is on edge - move vertex perpendicular to edge
        n = len(coords) - 1  # Exclude closing vertex
        prev_idx = (vertex_idx - 1) % n
        next_idx = (vertex_idx + 1) % n
        return _calculate_edge_perpendicular(
            coords[vertex_idx],
            coords[prev_idx],
            coords[next_idx],
            opposite_clearance_pt
        )


def _move_two_vertices(
    coords: np.ndarray,
    vertex_idx_1: int,
    vertex_idx_2: int,
    is_near_1: bool,
    is_near_2: bool,
    pt1: np.ndarray,
    pt2: np.ndarray,
    clearance_direction: np.ndarray,
    movement_distance: float,
    original_geometry: Polygon,
    current_clearance: float
) -> Union[Polygon, None]:
    """Move two vertices apart to widen a narrow passage.

    Args:
        coords: Polygon coordinate array
        vertex_idx_1: Index of first vertex
        vertex_idx_2: Index of second vertex
        is_near_1: Whether pt1 is at vertex_idx_1 (vs on edge)
        is_near_2: Whether pt2 is at vertex_idx_2 (vs on edge)
        pt1: First clearance point
        pt2: Second clearance point
        clearance_direction: Normalized direction from pt1 to pt2
        movement_distance: How far to move each vertex
        original_geometry: Original polygon for validation
        current_clearance: Current minimum clearance

    Returns:
        New polygon or None if unsuccessful
    """
    # Determine movement direction for each vertex
    direction_1 = _determine_movement_direction(
        is_near_1, -clearance_direction, coords, vertex_idx_1, pt2
    )
    direction_2 = _determine_movement_direction(
        is_near_2, clearance_direction, coords, vertex_idx_2, pt1
    )

    # Create new coordinates
    new_coords = coords.copy()

    # Move vertex 1
    vertex_1 = coords[vertex_idx_1][:2]
    new_position_1 = vertex_1 + direction_1 * movement_distance

    if coords.shape[1] > 2:
        new_coords[vertex_idx_1] = np.array([new_position_1[0], new_position_1[1], coords[vertex_idx_1][2]])
    else:
        new_coords[vertex_idx_1] = new_position_1

    # Move vertex 2
    vertex_2 = coords[vertex_idx_2][:2]
    new_position_2 = vertex_2 + direction_2 * movement_distance

    if coords.shape[1] > 2:
        new_coords[vertex_idx_2] = np.array([new_position_2[0], new_position_2[1], coords[vertex_idx_2][2]])
    else:
        new_coords[vertex_idx_2] = new_position_2

    # Update closing vertex if needed
    if vertex_idx_1 == 0 or vertex_idx_2 == 0:
        new_coords[-1] = new_coords[0]

    # Create and validate polygon
    min_area = original_geometry.area * 0.5
    new_poly = _create_polygon_with_new_coords(new_coords, original_geometry, min_area)

    if new_poly is not None:
        # Only return if clearance improved
        new_clearance = new_poly.minimum_clearance
        if new_clearance > current_clearance:
            return new_poly

    return None


def _widen_narrow_passage(
    geometry: Polygon,
    min_clearance: float,
    max_iterations: int = 10
) -> Polygon:
    """Widen a narrow passage by iteratively moving vertices apart.

    Args:
        geometry: Input polygon
        min_clearance: Target minimum clearance
        max_iterations: Maximum number of widening iterations

    Returns:
        Widened polygon
    """
    result = geometry

    for iteration in range(max_iterations):
        current_clearance = result.minimum_clearance

        if current_clearance >= min_clearance:
            return result

        # Get minimum clearance line
        try:
            clearance_line = shapely.minimum_clearance_line(result)
        except Exception:
            return result

        if clearance_line is None or clearance_line.is_empty:
            return result

        # Get clearance line endpoints
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

        # Find nearest vertices
        coords = np.array(result.exterior.coords)
        threshold = min_clearance * 0.05
        is_near_1, vertex_idx_1 = _is_point_near_vertex(pt1, coords, threshold)
        is_near_2, vertex_idx_2 = _is_point_near_vertex(pt2, coords, threshold)

        # Special case: both points map to same vertex (vertex to adjacent edge)
        if vertex_idx_1 == vertex_idx_2:
            required_increase = min_clearance - current_clearance
            movement_distance = required_increase * 1.1

            new_poly = _move_single_vertex_perpendicular(
                coords, vertex_idx_2, pt2, movement_distance,
                geometry, current_clearance
            )

            if new_poly is not None:
                result = new_poly
                continue
            else:
                return result

        # General case: move two vertices apart
        required_increase = min_clearance - current_clearance
        movement_distance = required_increase * 0.55  # 55% = half + buffer

        new_poly = _move_two_vertices(
            coords, vertex_idx_1, vertex_idx_2,
            is_near_1, is_near_2, pt1, pt2,
            clearance_direction, movement_distance,
            geometry, current_clearance
        )

        if new_poly is not None:
            result = new_poly
        else:
            return result

    return result


def fix_narrow_passage(
    geometry: Polygon,
    min_clearance: float,
    strategy: Union[PassageStrategy, str] = PassageStrategy.WIDEN,
) -> Union[Polygon, MultiPolygon]:
    """Fix narrow passages (hourglass/neck shapes) that cause low clearance.

    Narrow passages occur when a polygon has a thin section connecting two
    larger areas, creating an hourglass or dumbbell shape.

    Args:
        geometry: Input polygon
        min_clearance: Target minimum clearance
        strategy: How to fix the passage:
            - PassageStrategy.WIDEN: Move vertices apart at narrow point (default)
            - PassageStrategy.SPLIT: Split into separate polygons at narrow point

    Returns:
        Fixed geometry (Polygon if widened, MultiPolygon if split)

    Note:
        The WIDEN strategy uses minimum_clearance_line to identify the
        narrow point and moves the nearest vertices apart. This preserves
        the overall polygon shape better than buffering.

        The algorithm handles multiple clearance configurations:
        - Vertex-to-vertex: moves both vertices along clearance line
        - Vertex-to-edge: moves vertex along clearance line, edge vertex perpendicular
        - Same vertex (vertex to adjacent edge): moves vertex perpendicular to edge

        Uses 5% of min_clearance as threshold to detect edge vs vertex cases.
    """
    strategy_enum = coerce_enum(strategy, PassageStrategy)

    if strategy_enum == PassageStrategy.SPLIT:
        return _split_narrow_passage(geometry)
    else:
        return _widen_narrow_passage(geometry, min_clearance)


def fix_near_self_intersection(
    geometry: Polygon,
    min_clearance: float,
    strategy: Union[IntersectionStrategy, str] = IntersectionStrategy.SIMPLIFY,
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
    strategy_enum = coerce_enum(strategy, IntersectionStrategy)
    context = _find_self_intersection_vertices(geometry)
    current_clearance = context.clearance if context else geometry.minimum_clearance

    if current_clearance >= min_clearance:
        return geometry

    if strategy_enum == IntersectionStrategy.BUFFER:
        return _fix_near_self_intersection_buffer(geometry, min_clearance, context, current_clearance)

    return _fix_near_self_intersection_simplify(geometry, min_clearance, strategy_enum)


def fix_parallel_close_edges(
    geometry: Polygon,
    min_clearance: float,
    strategy: Union[IntersectionStrategy, str] = IntersectionStrategy.SIMPLIFY,
) -> Polygon:
    """Fix parallel edges that run too close to each other.
    """
    # Parallel close edges are essentially a type of near-self-intersection
    # We can reuse the same fixing logic
    return fix_near_self_intersection(geometry, min_clearance, strategy)


def _find_self_intersection_vertices(geometry: Polygon) -> Optional[SelfIntersectionContext]:
    """Return information about the closest approach between two edges."""
    try:
        clearance_line = shapely.minimum_clearance_line(geometry)
    except Exception:
        return None

    if clearance_line is None or clearance_line.is_empty:
        return None

    coords = list(clearance_line.coords)
    if len(coords) != 2:
        return None

    point_a = np.array(coords[0], dtype=float)
    point_b = np.array(coords[1], dtype=float)
    exterior_coords = np.array(geometry.exterior.coords)
    idx_a = _find_nearest_vertex_index(exterior_coords, point_a)
    idx_b = _find_nearest_vertex_index(exterior_coords, point_b)

    try:
        clearance = float(geometry.minimum_clearance)
    except Exception:
        clearance = 0.0

    return SelfIntersectionContext(
        clearance=clearance,
        point_a=point_a,
        point_b=point_b,
        vertex_idx_a=idx_a,
        vertex_idx_b=idx_b,
    )


def _fix_near_self_intersection_buffer(
    geometry: Polygon,
    min_clearance: float,
    context: Optional[SelfIntersectionContext],
    current_clearance: float,
) -> Polygon:
    """Use buffering to push edges apart."""
    deficit = max(0.0, min_clearance - current_clearance)
    buffer_dist = deficit / 2 + 0.01
    candidates = [buffer_dist * m for m in (1.0, 1.5, 2.0, 3.0)]

    for dist in candidates:
        buffered = _buffer_geometry(geometry, dist)
        if buffered is None:
            continue

        if geometry.interiors:
            valid_holes = _validate_holes_after_buffer(
                buffered.exterior,
                list(geometry.interiors),
                min_clearance,
            )
            buffered = Polygon(buffered.exterior, holes=valid_holes)

        new_context = _find_self_intersection_vertices(buffered)
        if new_context and new_context.clearance >= min_clearance:
            return buffered
        if new_context is None:
            try:
                if buffered.minimum_clearance >= min_clearance:
                    return buffered
            except Exception:
                pass

    return geometry


def _fix_near_self_intersection_simplify(
    geometry: Polygon,
    min_clearance: float,
    strategy: IntersectionStrategy,
    max_iterations: int = 5,
) -> Polygon:
    """Use progressive simplification or smoothing to separate edges."""
    base_epsilon = (min_clearance / 3) if strategy == IntersectionStrategy.SMOOTH else (min_clearance / 2)
    return _run_simplification_loop(
        geometry,
        min_clearance=min_clearance,
        base_epsilon=base_epsilon,
        max_iterations=max_iterations,
        min_area_ratio=0.8,
    )


def _buffer_geometry(geometry: Polygon, distance: float) -> Optional[Polygon]:
    """Buffer geometry safely and return a polygon candidate."""
    if distance <= 0:
        return None
    try:
        buffered = geometry.buffer(distance)
    except Exception:
        return None

    if isinstance(buffered, Polygon):
        return buffered if buffered.is_valid else None

    if isinstance(buffered, MultiPolygon) and buffered.geoms:
        largest = max(buffered.geoms, key=lambda p: p.area)
        return largest if largest.is_valid else None

    return None


def _run_simplification_loop(
    geometry: Polygon,
    min_clearance: float,
    base_epsilon: float,
    max_iterations: int,
    min_area_ratio: float,
) -> Polygon:
    """Iteratively simplify geometry until clearance target is met."""
    from polyforge.simplify import simplify_rdp

    current = geometry
    best = geometry
    best_clearance = geometry.minimum_clearance

    for iteration in range(max_iterations):
        current_clearance = current.minimum_clearance
        if current_clearance >= min_clearance:
            break

        epsilon = base_epsilon * (1.5 ** iteration)
        if current.length > 0:
            epsilon = min(epsilon, current.length / 12)

        try:
            candidate = simplify_rdp(current, epsilon=epsilon)
        except Exception:
            break

        if not _is_valid_simplification_candidate(candidate, geometry, min_area_ratio):
            continue

        candidate_clearance = candidate.minimum_clearance
        if candidate_clearance > best_clearance:
            best = candidate
            best_clearance = candidate_clearance
            current = candidate
        elif iteration > 0:
            break

    return best


def _is_valid_simplification_candidate(
    candidate: Polygon,
    original: Polygon,
    min_area_ratio: float,
) -> bool:
    """Validate simplified polygon before acceptance."""
    if candidate is None or candidate.is_empty or not candidate.is_valid:
        return False
    if len(candidate.exterior.coords) < 4:
        return False
    if original.area > 0 and candidate.area < original.area * min_area_ratio:
        return False
    return True


__all__ = [
    'fix_narrow_passage',
    'fix_near_self_intersection',
    'fix_parallel_close_edges',
]
