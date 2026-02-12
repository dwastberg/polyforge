"""Functions for fixing narrow protrusions and sharp intrusions.

This module provides functions to handle thin spikes, peninsulas, and
sharp indentations that create low minimum clearance.
"""

import numpy as np
from typing import Optional, Tuple, Union
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points
import shapely

from .utils import (
    _point_to_segment_distance,
    _find_nearest_vertex_index,
    _find_nearest_edge_index,
    _remove_vertices_between,
)
from polyforge.core.geometry_utils import safe_buffer_fix
from polyforge.core.types import IntrusionStrategy, coerce_enum


def fix_narrow_protrusion(
    geometry: Polygon,
    min_clearance: float,
    max_iterations: int = 10,
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
    candidate_builder = _make_protrusion_candidate_builder(min_clearance)
    result, _ = _run_clearance_loop(
        geometry,
        min_clearance,
        max_iterations,
        candidate_builder,
        min_area_ratio=0.5,
    )
    return result


def fix_sharp_intrusion(
    geometry: Polygon,
    min_clearance: float,
    strategy: Union[IntrusionStrategy, str] = IntrusionStrategy.FILL,
    max_iterations: int = 10,
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
    strategy_enum = coerce_enum(strategy, IntrusionStrategy)
    candidate_builder = _make_intrusion_candidate_builder(strategy_enum, min_clearance, geometry)
    _, best = _run_clearance_loop(
        geometry,
        min_clearance,
        max_iterations,
        candidate_builder,
        min_area_ratio=0.9,
    )
    return best


def _detect_protrusion_bottleneck(geometry: Polygon) -> Optional[Tuple[np.ndarray, np.ndarray, int, int]]:
    """Return clearance line endpoints, direction, and nearest vertex indices."""
    try:
        clearance_line = shapely.minimum_clearance_line(geometry)
    except Exception:
        return None

    if clearance_line is None or clearance_line.is_empty:
        return None

    coords_list = list(clearance_line.coords)
    if len(coords_list) != 2:
        return None

    pt1 = np.array(coords_list[0])
    pt2 = np.array(coords_list[1])
    vector = pt2 - pt1
    distance = np.linalg.norm(vector)
    if distance < 1e-10:
        return None

    direction = vector / distance
    exterior_coords = np.array(geometry.exterior.coords)
    idx1 = _find_nearest_vertex_index(exterior_coords, pt1)
    idx2 = _find_nearest_vertex_index(exterior_coords, pt2)
    return pt1, direction, idx1, idx2


def _move_vertices_for_clearance(
    geometry: Polygon,
    bottleneck: Tuple[np.ndarray, np.ndarray, int, int],
    target_clearance: float,
    current_clearance: float,
) -> Optional[Polygon]:
    """Create a new polygon by moving the two bottleneck vertices apart."""
    pt1, direction, idx1, idx2 = bottleneck
    required_increase = max(0.0, target_clearance - current_clearance)
    if required_increase <= 0:
        return None
    movement_distance = required_increase * 0.55

    coords = np.array(geometry.exterior.coords)
    new_coords = coords.copy()

    new_coords[idx1] = _translate_vertex(coords[idx1], -direction[:2], movement_distance)
    new_coords[idx2] = _translate_vertex(coords[idx2], direction[:2], movement_distance)

    if idx1 == 0 or idx2 == 0:
        new_coords[-1] = new_coords[0]

    holes = [list(interior.coords) for interior in geometry.interiors]
    new_poly = Polygon(new_coords, holes=holes) if holes else Polygon(new_coords)

    if not new_poly.is_valid or new_poly.is_empty:
        healed = safe_buffer_fix(new_poly, distance=0.0, return_largest=True)
        if healed is None or healed.geom_type != "Polygon":
            return None
        new_poly = healed

    return new_poly


def _translate_vertex(
    vertex: np.ndarray,
    direction: np.ndarray,
    distance: float,
) -> np.ndarray:
    """Move a vertex along a direction, preserving optional Z component."""
    if len(vertex) > 2:
        xy = np.array(vertex[:2]) + direction * distance
        return np.array([xy[0], xy[1], vertex[2]])
    return np.array(vertex[:2]) + direction * distance


def _should_accept_candidate(
    candidate: Polygon,
    current: Polygon,
    original: Polygon,
    min_area_ratio: float,
) -> bool:
    """Decide whether to accept the candidate polygon."""
    if not candidate.is_valid or candidate.is_empty:
        return False
    if original.area > 0 and candidate.area < original.area * min_area_ratio:
        return False
    try:
        return candidate.minimum_clearance > current.minimum_clearance
    except Exception:
        return False


def _run_clearance_loop(
    geometry: Polygon,
    min_clearance: float,
    max_iterations: int,
    candidate_builder,
    min_area_ratio: float,
) -> Tuple[Polygon, Polygon]:
    """Generic improvement loop returning (current, best) geometries."""
    current = geometry
    best = geometry
    best_clearance = geometry.minimum_clearance

    for iteration in range(max_iterations):
        current_clearance = current.minimum_clearance
        if current_clearance >= min_clearance:
            break

        candidate = candidate_builder(current, iteration, current_clearance)
        if candidate is None:
            break

        if _should_accept_candidate(candidate, current, geometry, min_area_ratio):
            current = candidate
            try:
                candidate_clearance = candidate.minimum_clearance
            except Exception:
                break
            if candidate_clearance > best_clearance:
                best = candidate
                best_clearance = candidate_clearance
        else:
            break

    return current, best


def _make_protrusion_candidate_builder(min_clearance: float):
    """Return a closure that builds protrusion candidates."""
    def _builder(current: Polygon, iteration: int, current_clearance: float) -> Optional[Polygon]:
        bottleneck = _detect_protrusion_bottleneck(current)
        if bottleneck is None:
            return None
        return _move_vertices_for_clearance(
            current,
            bottleneck,
            min_clearance,
            current_clearance,
        )

    return _builder


def _make_intrusion_candidate_builder(
    strategy: IntrusionStrategy,
    min_clearance: float,
    original: Polygon,
):
    """Return a closure that builds sharp intrusion candidates via simplification."""
    from polyforge.simplify import simplify_rdp

    if strategy == IntrusionStrategy.SMOOTH:
        base_epsilon = min_clearance / 3
    else:
        base_epsilon = min_clearance / 2

    original_area = original.area

    def _builder(current: Polygon, iteration: int, _: float) -> Optional[Polygon]:
        epsilon = base_epsilon * (2.0 ** iteration)
        epsilon = min(epsilon, current.length / 10)
        try:
            simplified = simplify_rdp(current, epsilon=epsilon)
        except Exception:
            return None

        if not (
            simplified.is_valid
            and not simplified.is_empty
            and (original_area == 0 or simplified.area >= original_area * 0.9)
            and len(simplified.exterior.coords) >= 4
        ):
            return None

        return simplified

    return _builder


def fill_narrow_wedge(
    geometry: Polygon,
    min_clearance: float,
    min_area_ratio: float = 0.5,
) -> Optional[Polygon]:
    """Remove a narrow wedge intrusion by bridging its opening.

    Traces outward from the narrowest point along both sides of the wedge
    until the width exceeds min_clearance, then removes all intermediate
    vertices and connects the opening points with a direct edge.

    Handles both inward wedges (V-notches) and outward wedges (tapered
    peninsulas).

    Args:
        geometry: Input polygon with a narrow wedge feature.
        min_clearance: Target minimum clearance. Vertices are removed up
            to where the cross-width reaches this threshold.
        min_area_ratio: Minimum acceptable area ratio vs original polygon
            (default: 0.5).

    Returns:
        Polygon with the wedge removed, or None if the operation fails.
    """
    try:
        clearance_line = shapely.minimum_clearance_line(geometry)
    except Exception:
        return None

    if clearance_line is None or clearance_line.is_empty:
        return None

    coords_list = list(clearance_line.coords)
    if len(coords_list) != 2:
        return None

    pt1 = np.array(coords_list[0])
    pt2 = np.array(coords_list[1])

    exterior_coords = np.array(geometry.exterior.coords)
    n = len(exterior_coords) - 1  # closed ring: last == first
    if n < 4:
        return None

    idx1 = _find_nearest_vertex_index(exterior_coords, pt1)
    idx2 = _find_nearest_vertex_index(exterior_coords, pt2)

    if idx1 == idx2:
        # Both endpoints map to the same vertex — one is at the vertex,
        # the other is on an adjacent edge.  Use the edge index to find
        # the neighbouring vertex on the opposite side of the wedge.
        edge_idx = _find_nearest_edge_index(exterior_coords, pt2)
        if edge_idx == idx1:
            idx2 = (edge_idx + 1) % n
        else:
            idx2 = edge_idx
        if idx1 == idx2:
            return None

    # Determine short path (through wedge tip) direction
    fwd = (idx2 - idx1) % n
    bwd = (idx1 - idx2) % n
    separation = min(fwd, bwd)

    if separation < 1:
        return None  # Need at least one vertex between endpoints

    # Walking directions: away from the short path (toward the opening)
    if fwd == separation:
        # Short path: idx1 -> idx1+1 -> ... -> idx2
        # Walk away: idx1 backward, idx2 forward
        dir1 = -1
        dir2 = +1
    else:
        # Short path: idx2 -> idx2+1 -> ... -> idx1
        # Walk away: idx1 forward, idx2 backward
        dir1 = +1
        dir2 = -1

    # Walk outward until width >= min_clearance
    max_steps = n // 2
    open1 = idx1
    open2 = idx2

    for step in range(1, max_steps + 1):
        walk1 = (idx1 + dir1 * step) % n
        walk2 = (idx2 + dir2 * step) % n
        if walk1 == walk2:
            break
        dist = float(np.linalg.norm(
            exterior_coords[walk1][:2] - exterior_coords[walk2][:2]
        ))
        open1 = walk1
        open2 = walk2
        if dist >= min_clearance:
            break

    # Remove vertices on the short path between open1 and open2.
    # The short path goes: open1 -> (toward tip) -> open2
    # We want to keep the LONG path and remove the short path interior.
    # _remove_vertices_between(coords, start, end) keeps start and end,
    # removes everything between start and end in index order.
    # We need to figure out which index ordering removes the short path.

    # Vertices on the short path from open1 to open2 (through the tip):
    # If dir1 == -1, open1 moved backward from idx1, so the short path
    # goes forward from open1 to open2: open1, open1+1, ..., open2
    if dir1 == -1:
        # Short path is forward: open1 -> open1+1 -> ... -> open2
        start_idx = open1
        end_idx = open2
    else:
        # Short path is backward: open1 -> open1-1 -> ... -> open2
        # i.e. forward from open2 to open1
        start_idx = open2
        end_idx = open1

    new_coords = _remove_vertices_between(exterior_coords, start_idx, end_idx)

    if new_coords is None or len(new_coords) < 4:
        return None

    # Ensure closed ring
    if not np.allclose(new_coords[0], new_coords[-1]):
        new_coords = np.vstack([new_coords, new_coords[0:1]])

    # Build polygon preserving holes
    holes = [list(interior.coords) for interior in geometry.interiors]
    try:
        new_poly = Polygon(new_coords, holes=holes) if holes else Polygon(new_coords)
    except Exception:
        return None

    if not new_poly.is_valid or new_poly.is_empty:
        healed = safe_buffer_fix(new_poly, distance=0.0, return_largest=True)
        if healed is None or healed.geom_type != "Polygon":
            return None
        new_poly = healed

    # Validate area ratio
    original_area = geometry.area
    if original_area > 0 and new_poly.area < min_area_ratio * original_area:
        return None

    # Validate clearance improved
    try:
        if new_poly.minimum_clearance <= geometry.minimum_clearance:
            return None
    except Exception:
        return None

    return new_poly


__all__ = [
    'fix_narrow_protrusion',
    'fix_sharp_intrusion',
    'fill_narrow_wedge',
]
