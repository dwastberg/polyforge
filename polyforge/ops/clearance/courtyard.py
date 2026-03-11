"""Courtyard detection and passage-to-hole conversion.

Detects when two nearby points on the exterior ring enclose a significant
courtyard area behind a narrow passage, and converts it to a hole.
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.errors import GEOSException

from polyforge.ops.clearance.utils import (
    _find_nearest_vertex_index,
    _find_nearest_edge_index,
)
from polyforge.clearance._helpers import normalize_polygon
from polyforge.clearance._diagnosis import _classify_ring_types

# Minimum ratio of enclosed courtyard area to min_clearance² for courtyard
# passage detection.  A courtyard must be at least this many times larger than
# a min_clearance-sided square to be worth preserving as a hole.
_COURTYARD_AREA_THRESHOLD = 10.0


def try_close_passage_to_hole(
    geometry: Polygon, min_clearance: float
) -> Polygon | None:
    """Close a narrow exterior passage and convert the enclosed area to a hole.

    Detects when two nearby points on the exterior ring enclose a significant
    courtyard area behind a narrow passage.  Snaps the passage vertices together
    and uses make_valid() to split the polygon into exterior + hole.

    Args:
        geometry: Input polygon.
        min_clearance: Target minimum clearance.

    Returns:
        Polygon with courtyard as hole, or None if pattern not detected.
    """
    try:
        clearance_line = shapely.minimum_clearance_line(geometry)
    except (GEOSException, ValueError):
        return None
    if clearance_line.is_empty:
        return None

    cl_pts = np.array(clearance_line.coords)
    if len(cl_pts) < 2:
        return None

    ext_coords = np.array(geometry.exterior.coords[:-1])  # open ring
    n = len(ext_coords)
    if n < 6:
        # Need at least 6 vertices: 2 for passage + 3 for courtyard + 1 for exterior
        return None

    # Both endpoints must be on the exterior ring
    (rt1, _), (rt2, _) = _classify_ring_types(geometry, cl_pts[0], cl_pts[1])
    if rt1 != "exterior" or rt2 != "exterior":
        return None

    # Find nearest exterior vertices to each clearance endpoint
    raw_idx1 = _find_nearest_vertex_index(ext_coords, cl_pts[0])
    raw_idx2 = _find_nearest_vertex_index(ext_coords, cl_pts[1])
    if raw_idx1 == raw_idx2:
        # Both endpoints map to same vertex — use the nearest edge's closer endpoint
        edge_idx = _find_nearest_edge_index(ext_coords, cl_pts[1])
        cand_a, cand_b = edge_idx, (edge_idx + 1) % n
        # Pick the candidate that is different from raw_idx1; if both differ, pick closer
        if cand_a == raw_idx1:
            raw_idx2 = cand_b
        elif cand_b == raw_idx1:
            raw_idx2 = cand_a
        else:
            d_a = float(np.linalg.norm(ext_coords[cand_a] - cl_pts[1]))
            d_b = float(np.linalg.norm(ext_coords[cand_b] - cl_pts[1]))
            raw_idx2 = cand_a if d_a <= d_b else cand_b
    if raw_idx1 == raw_idx2:
        return None

    # Ensure idx1 < idx2
    idx1, idx2 = (raw_idx1, raw_idx2) if raw_idx1 < raw_idx2 else (raw_idx2, raw_idx1)

    # Two paths along the ring between idx1 and idx2
    path_a_indices = list(range(idx1, idx2 + 1))  # idx1 -> idx2 (forward)
    path_b_indices = list(range(idx2, n)) + list(range(0, idx1 + 1))  # idx2 -> idx1 (wrap)

    # Compute enclosed area of each path (closed with straight line between endpoints)
    area_a = _path_enclosed_area(ext_coords, path_a_indices)
    area_b = _path_enclosed_area(ext_coords, path_b_indices)

    # Courtyard is the smaller enclosed area
    if area_a <= area_b:
        courtyard_area = area_a
    else:
        courtyard_area = area_b

    # Check courtyard is significant
    if courtyard_area < _COURTYARD_AREA_THRESHOLD * min_clearance * min_clearance:
        return None

    # Need at least 4 vertices in courtyard path (to form a valid ring)
    courtyard_indices = path_a_indices if area_a <= area_b else path_b_indices
    if len(courtyard_indices) < 4:
        return None

    # --- Find passage mouth ---
    mouth = _find_passage_mouth(
        ext_coords, idx1, idx2, courtyard_indices, threshold=min_clearance * 2
    )

    if mouth is None:
        # Passage is only one vertex pair wide — use snap+make_valid fallback
        midpoint = (ext_coords[idx1] + ext_coords[idx2]) / 2
        new_coords = ext_coords.copy()
        new_coords[idx1] = midpoint
        new_coords[idx2] = midpoint
        ring = np.vstack([new_coords, [new_coords[0]]])
        holes = [list(h.coords) for h in geometry.interiors]
        try:
            self_touching = Polygon(ring, holes) if holes else Polygon(ring)
            candidate = shapely.make_valid(self_touching)
        except (GEOSException, ValueError):
            return None
        result = normalize_polygon(candidate)
        if result is None or len(result.interiors) <= len(geometry.interiors):
            return None
        return result

    mouth1, mouth2 = mouth
    midpoint = (ext_coords[mouth1] + ext_coords[mouth2]) / 2

    # Identify arc to remove: mouth1 through passage+courtyard to mouth2
    path_fwd = list(range(mouth1, mouth2 + 1))
    path_bck = list(range(mouth2, n)) + list(range(0, mouth1 + 1))
    courtyard_set = set(courtyard_indices)
    if courtyard_set.issubset(set(path_fwd)):
        arc_to_remove = set(path_fwd)
    else:
        arc_to_remove = set(path_bck)

    # Build exterior ring: main body vertices in order, midpoint replaces arc
    ext_ring_coords = []
    midpoint_inserted = False
    for i in range(n):
        if i in arc_to_remove:
            if not midpoint_inserted:
                ext_ring_coords.append(midpoint)
                midpoint_inserted = True
        else:
            ext_ring_coords.append(ext_coords[i])

    if len(ext_ring_coords) < 3:
        return None
    ext_ring_coords.append(ext_ring_coords[0])  # close ring

    # Build hole ring from courtyard vertices
    hole_coords = [ext_coords[i].tolist() for i in courtyard_indices]
    if len(hole_coords) < 3:
        return None
    hole_coords.append(hole_coords[0])  # close ring

    # Preserve existing holes
    existing_holes = [list(h.coords) for h in geometry.interiors]
    all_holes = existing_holes + [hole_coords]

    try:
        result = Polygon(ext_ring_coords, all_holes)
    except (GEOSException, ValueError):
        return None

    if not result.is_valid or result.is_empty:
        try:
            result = normalize_polygon(shapely.make_valid(result))
        except (GEOSException, ValueError):
            return None
        if result is None:
            return None

    # Must have gained at least one hole
    if len(result.interiors) <= len(geometry.interiors):
        return None

    return result


def _path_enclosed_area(coords: np.ndarray, indices: list[int]) -> float:
    """Compute the area enclosed by a path along the ring, closed by a straight line.

    Takes vertex indices along the ring and computes the area of the polygon
    formed by connecting them in order plus closing back to the first vertex.
    """
    if len(indices) < 3:
        return 0.0
    path_coords = coords[indices]
    # Close the ring
    closed = np.vstack([path_coords, [path_coords[0]]])
    try:
        return abs(Polygon(closed).area)
    except (GEOSException, ValueError):
        return 0.0


def _find_passage_mouth(
    ext_coords: np.ndarray,
    idx1: int,
    idx2: int,
    courtyard_indices: list[int],
    threshold: float,
) -> tuple[int, int] | None:
    """Walk outward from bottleneck to find passage mouth vertices.

    Starting from (idx1, idx2), walks along both passage walls away from
    the courtyard. Returns outermost pair still within threshold distance.

    Returns:
        (mouth_idx1, mouth_idx2) with mouth_idx1 < mouth_idx2, or None.
    """
    n = len(ext_coords)
    courtyard_set = set(courtyard_indices)

    # Determine walk direction: away from courtyard
    # From idx1, step to (idx1-1)%n; if that's in courtyard, reverse direction
    test_step = (idx1 - 1) % n
    if test_step in courtyard_set:
        dir1, dir2 = +1, -1
    else:
        dir1, dir2 = -1, +1

    mouth1, mouth2 = idx1, idx2
    max_steps = n // 2

    for _ in range(max_steps):
        next1 = (mouth1 + dir1) % n
        next2 = (mouth2 + dir2) % n

        if next1 in courtyard_set or next2 in courtyard_set:
            break
        if next1 == next2:
            break

        dist = float(np.linalg.norm(ext_coords[next1] - ext_coords[next2]))
        if dist > threshold:
            break

        mouth1, mouth2 = next1, next2

    if mouth1 == idx1 and mouth2 == idx2:
        return None  # No outward walk — passage is only at bottleneck

    m1, m2 = min(mouth1, mouth2), max(mouth1, mouth2)
    return (m1, m2)
