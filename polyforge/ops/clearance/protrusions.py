from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points
from shapely.validation import make_valid

import shapely

from .utils import (
    _point_to_segment_distance,
    _find_nearest_vertex_index,
    _find_nearest_edge_index,
    _remove_vertices_between,
    _angle,
    _is_concave,
    _find_best_join,
    _compute_depth,
    _trace_wedge,
    _splice_polygon,
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
        Polygon with narrow protrusions fixed
    """
    candidate_builder = _make_protrusion_candidate_builder(min_clearance)
    result, _ = _run_clearance_loop(
        geometry,
        min_clearance,
        max_iterations,
        candidate_builder,
        min_area_ratio=0.9,
    )
    return result


def fix_sharp_intrusion(
    geometry: Polygon,
    min_clearance: float,
    strategy: IntrusionStrategy | str = IntrusionStrategy.FILL,
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
    """
    strategy_enum = coerce_enum(strategy, IntrusionStrategy)
    candidate_builder = _make_intrusion_candidate_builder(
        strategy_enum, min_clearance, geometry
    )
    _, best = _run_clearance_loop(
        geometry,
        min_clearance,
        max_iterations,
        candidate_builder,
        min_area_ratio=0.9,
    )
    return best


def _detect_protrusion_bottleneck(
    geometry: Polygon,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
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
    bottleneck: tuple[np.ndarray, np.ndarray, int, int],
    target_clearance: float,
    current_clearance: float,
) -> Polygon | None:
    """Create a new polygon by moving the two bottleneck vertices apart."""
    pt1, direction, idx1, idx2 = bottleneck
    required_increase = max(0.0, target_clearance - current_clearance)
    if required_increase <= 0:
        return None
    movement_distance = required_increase * 0.55

    coords = np.array(geometry.exterior.coords)
    new_coords = coords.copy()

    new_coords[idx1] = _translate_vertex(
        coords[idx1], -direction[:2], movement_distance
    )
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
) -> tuple[Polygon, Polygon]:
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

    def _builder(
        current: Polygon, iteration: int, current_clearance: float
    ) -> Polygon | None:
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

    def _builder(current: Polygon, iteration: int, _: float) -> Polygon | None:
        epsilon = base_epsilon * (2.0**iteration)
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


def _find_wedges(coords, angle_threshold, depth_width_ratio, min_depth, orientation=1):
    """Detect wedge candidates in a coordinate list.

    Returns list of dicts with keys: tip, left, right, depth, width, ratio.

    Parameters
    ----------
    orientation : int
        1 for CCW rings (exterior), -1 for CW rings (holes).
    """
    n = len(coords)
    wedges = []
    for i in range(n):
        prev = coords[(i - 1) % n]
        curr = coords[i]
        nxt = coords[(i + 1) % n]

        ang = _angle(prev, curr, nxt)
        if ang > angle_threshold:
            continue

        if not _is_concave(prev, curr, nxt, orientation=orientation):
            continue

        left_chain, right_chain = _trace_wedge(coords, i, orientation)

        join = _find_best_join(coords, left_chain, right_chain)
        if not join:
            continue

        li, ri, width = join

        depth = _compute_depth(coords, i, left_chain, right_chain)
        if depth < min_depth or width <= 0:
            continue

        ratio = depth / width
        if ratio < depth_width_ratio:
            continue

        wedges.append(
            {
                "tip": i,
                "left": li,
                "right": ri,
                "depth": depth,
                "width": width,
                "ratio": ratio,
            }
        )
    return wedges


def _collapse_near_coincident(coords, tolerance):
    """Merge consecutive vertices closer than tolerance into their midpoint.

    Works on an open coordinate list (no closing duplicate).
    Handles the wraparound case (last vertex near first vertex).
    Returns a new list; unchanged if no merges needed.
    """
    if len(coords) < 3:
        return coords
    n = len(coords)
    tol_sq = tolerance * tolerance

    # Check wraparound: last vertex near first vertex
    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    if (dx * dx + dy * dy) < tol_sq:
        mid = ((coords[-1][0] + coords[0][0]) / 2,
               (coords[-1][1] + coords[0][1]) / 2)
        # Replace first with midpoint, drop last
        coords = [mid] + list(coords[1:-1])
        n = len(coords)

    # Check interior consecutive pairs
    result = []
    skip = set()
    for i in range(n):
        if i in skip:
            continue
        if i + 1 < n:
            dx = coords[i][0] - coords[i + 1][0]
            dy = coords[i][1] - coords[i + 1][1]
            if (dx * dx + dy * dy) < tol_sq:
                mid = ((coords[i][0] + coords[i + 1][0]) / 2,
                       (coords[i][1] + coords[i + 1][1]) / 2)
                result.append(mid)
                skip.add(i + 1)
                continue
        result.append(coords[i])
    return result


def _remove_wedges_from_ring(coords, angle_threshold, depth_width_ratio,
                             min_depth, remove_multiple, orientation=1):
    """Remove narrow wedges from a single coordinate ring.

    Parameters
    ----------
    coords : list
        Open coordinate list (no closing duplicate).
    orientation : int
        1 for CCW (exterior), -1 for CW (holes).

    Returns
    -------
    list or None
        New open coordinate list, or None if no changes were made.
    """
    changed = False
    max_removals = len(coords)
    for _ in range(max_removals):
        wedges = _find_wedges(
            coords, angle_threshold, depth_width_ratio, min_depth,
            orientation=orientation,
        )
        if not wedges:
            break

        worst = max(wedges, key=lambda w: w["ratio"])
        neck_width = worst["width"]
        coords = _splice_polygon(coords, worst["left"], worst["right"], worst["tip"])

        # ensure open ring (drop closing dup if splice added one)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]

        # After splice, the two neck vertices remain and may be near-coincident.
        # Collapse them to avoid leaving a near-zero clearance slit.
        if neck_width > 0:
            coords = _collapse_near_coincident(coords, neck_width * 1.5)

        changed = True

        if not remove_multiple:
            break

    return coords if changed else None


def remove_narrow_wedges(
    polygon: Polygon,
    angle_threshold=20,
    depth_width_ratio=3.0,
    min_depth=0.0,
    remove_multiple=True,
) -> Polygon:
    """

    Parameters
    ----------
    angle_threshold : degrees
        Max angle at tip to consider wedge
    depth_width_ratio :
        Depth must be this multiple of neck width
    min_depth :
        Ignore tiny wedges below this
    remove_multiple :
        Remove all wedges (True) or only worst (False)
    """

    if polygon.is_empty:
        return polygon

    if not polygon.is_valid:
        polygon = make_valid(polygon)

    polygon = orient(polygon, sign=1.0)  # CCW exterior, CW holes

    # --- Process exterior ring (orientation=1, CCW) ---
    ext_coords = list(polygon.exterior.coords[:-1])
    holes = [list(h.coords[:-1]) for h in polygon.interiors]

    new_ext = _remove_wedges_from_ring(
        ext_coords, angle_threshold, depth_width_ratio,
        min_depth, remove_multiple, orientation=1,
    )
    if new_ext is not None:
        ext_coords = new_ext

    # --- Process each hole ring ---
    # After orient(sign=1.0), holes are CW. A wedge spike on a hole pokes
    # into the polygon body — its tip is concave in the CCW sense
    # (negative cross product), so we use orientation=1 same as exterior.
    new_holes = []
    for hole_coords in holes:
        new_hole = _remove_wedges_from_ring(
            hole_coords, angle_threshold, depth_width_ratio,
            min_depth, remove_multiple, orientation=1,
        )
        if new_hole is not None:
            # Only keep hole if it still has enough vertices for a polygon
            if len(new_hole) >= 3:
                new_holes.append(new_hole)
        else:
            new_holes.append(hole_coords)

    # --- Rebuild polygon ---
    # Close rings for Shapely
    ext_closed = ext_coords + [ext_coords[0]]
    holes_closed = [h + [h[0]] for h in new_holes]

    polygon = Polygon(ext_closed, holes_closed) if holes_closed else Polygon(ext_closed)

    if not polygon.is_valid:
        polygon = make_valid(polygon)
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)

    return polygon


__all__ = [
    "fix_narrow_protrusion",
    "fix_sharp_intrusion",
    "remove_narrow_wedges",
]
