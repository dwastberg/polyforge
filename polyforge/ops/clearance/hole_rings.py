"""Hole ring fix operations for clearance issues.

Pure geometry operations that fix clearance problems on individual hole rings.
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, LinearRing
from shapely.errors import GEOSException

from polyforge.ops.clearance.utils import _find_nearest_vertex_index
from polyforge.metrics import _safe_clearance
from polyforge.clearance._helpers import clearance_or_zero, is_usable, pick_best_by_clearance
from polyforge.clearance._diagnosis import ClearanceContext, _build_clearance_context


def try_hole_ring_fix(
    geometry: Polygon,
    min_clearance: float,
    context: ClearanceContext | None = None,
) -> Polygon | None:
    """Try to fix clearance by reshaping the offending hole ring.

    Checks if the minimum_clearance bottleneck is on a single hole ring.
    Tries multiple strategies in order:
    1. Erosion-dilation on the hole ring (handles spikes and close passages)
    2. Spike removal (targeted fix for long narrow spikes)
    3. Simplification (general fallback)
    """
    if context is None:
        context = _build_clearance_context(geometry, min_clearance)
    if context is None:
        return None
    # Only applies to same-hole self-clearance
    if not (
        context.ring_types == ("hole", "hole")
        and context.hole_indices[0] is not None
        and context.hole_indices[0] == context.hole_indices[1]
    ):
        return None

    hole_index = context.hole_indices[0]

    candidates = [
        erode_dilate_hole_ring(geometry, hole_index, min_clearance),
        remove_hole_ring_spike(geometry, hole_index, min_clearance),
        simplify_hole_ring(geometry, hole_index, min_clearance),
    ]
    return pick_best_by_clearance(geometry, candidates)


def erode_dilate_hole_ring(
    geometry: Polygon,
    hole_index: int,
    min_clearance: float,
    min_area_ratio: float = 0.8,
) -> Polygon | None:
    """Apply erosion-dilation to a specific hole ring to fix self-clearance.

    Buffers the hole polygon inward (erode) then outward (dilate) to remove
    narrow features like spikes and near-self-intersections. The result is
    simplified to avoid vertex explosion from buffering.

    Args:
        geometry: Input polygon.
        hole_index: Index of the hole to fix.
        min_clearance: Target minimum clearance.
        min_area_ratio: Minimum ratio of fixed hole area to original (default 0.8).

    Returns:
        Fixed polygon or None if the fix fails or loses too much hole area.
    """
    holes = list(geometry.interiors)
    if hole_index < 0 or hole_index >= len(holes):
        return None

    hole_ring = holes[hole_index]
    hole_poly = Polygon(hole_ring)
    original_hole_area = hole_poly.area
    if original_hole_area <= 0:
        return None

    # Try increasing buffer distances until clearance is resolved
    for scale in (1.0, 2.0, 4.0):
        d = min_clearance * scale
        half_d = d / 2

        try:
            smoothed = hole_poly.buffer(-half_d).buffer(half_d)
        except (GEOSException, ValueError):
            continue

        if smoothed.is_empty or not smoothed.is_valid:
            continue

        # Take the largest polygon if buffer split the hole
        if isinstance(smoothed, MultiPolygon):
            smoothed = max(smoothed.geoms, key=lambda p: p.area)

        # Check hole area preservation
        if smoothed.area < original_hole_area * min_area_ratio:
            continue

        # Simplify to reduce vertex count from buffering
        smoothed = smoothed.simplify(d * 0.25)
        if smoothed.is_empty or not smoothed.is_valid:
            continue

        # Reconstruct polygon with the smoothed hole
        new_holes = list(holes)
        new_holes[hole_index] = smoothed.exterior

        try:
            candidate = Polygon(geometry.exterior, new_holes)
        except (GEOSException, ValueError):
            continue

        if candidate.is_valid and not candidate.is_empty:
            return candidate

    return None


def remove_hole_ring_spike(
    geometry: Polygon, hole_index: int, min_clearance: float
) -> Polygon | None:
    """Remove spike features from a hole ring that cause near-self-intersection.

    A spike is a pattern where the ring goes from vertex A to a distant vertex B
    and then back to vertex C near A, creating a long narrow protrusion.
    The fix removes the spike vertex B and merges A and C into their midpoint.

    This handles cases that simplification cannot fix: when the spike tip
    deviates far from the baseline (e.g., 10 units) but the base width is
    tiny (e.g., 0.15 units), RDP/VW won't remove the tip vertex.
    """
    holes = list(geometry.interiors)
    if hole_index < 0 or hole_index >= len(holes):
        return None

    hole_coords = np.array(holes[hole_index].coords)
    n = len(hole_coords) - 1  # exclude closing vertex
    if n < 5:  # need at least 4 vertices after spike removal
        return None

    # Find the clearance bottleneck location on this hole ring
    try:
        clearance_line = shapely.minimum_clearance_line(geometry)
    except (GEOSException, ValueError):
        return None
    if clearance_line.is_empty:
        return None

    cl_pts = np.array(clearance_line.coords)
    if len(cl_pts) < 2:
        return None

    # Find the nearest vertex on the hole ring to each clearance endpoint
    idx1 = _find_nearest_vertex_index(hole_coords, cl_pts[0])
    idx2 = _find_nearest_vertex_index(hole_coords, cl_pts[1])

    # Look for spike patterns around the clearance bottleneck.
    # A spike has two close base vertices with a far tip vertex between them.
    best_candidate = None
    best_clearance = clearance_or_zero(geometry)

    # Search vertices near the clearance endpoints for spike patterns
    search_indices = set()
    for idx in (idx1, idx2):
        for offset in range(-2, 3):
            search_indices.add((idx + offset) % n)

    for vi in search_indices:
        prev_i = (vi - 1) % n
        next_i = (vi + 1) % n

        base_dist = float(np.linalg.norm(
            hole_coords[prev_i][:2] - hole_coords[next_i][:2]
        ))
        tip_dist_prev = float(np.linalg.norm(
            hole_coords[vi][:2] - hole_coords[prev_i][:2]
        ))
        tip_dist_next = float(np.linalg.norm(
            hole_coords[vi][:2] - hole_coords[next_i][:2]
        ))

        # Spike pattern: base is narrow, both arms are much longer than base
        if base_dist >= min_clearance:
            continue
        min_arm = min(tip_dist_prev, tip_dist_next)
        if min_arm < base_dist * 3:
            continue

        # Remove the spike: delete tip vertex, merge base vertices to midpoint
        midpoint = (hole_coords[prev_i][:2] + hole_coords[next_i][:2]) / 2
        new_coords = []
        merged = False
        for j in range(n):
            if j == vi:
                continue  # skip spike tip
            if j == prev_i and not merged:
                new_coords.append(midpoint)
                merged = True
            elif j == next_i and merged:
                continue  # skip second base vertex (already merged)
            else:
                new_coords.append(hole_coords[j][:2])

        if len(new_coords) < 3:
            continue

        # Close the ring
        new_coords.append(new_coords[0].copy())
        new_ring_coords = np.array(new_coords)

        # Reconstruct polygon with modified hole
        new_holes = list(holes)
        try:
            new_hole_ring = LinearRing(new_ring_coords)
            new_holes[hole_index] = new_hole_ring
            candidate = Polygon(geometry.exterior, new_holes)
        except (GEOSException, ValueError):
            continue

        if not candidate.is_valid or candidate.is_empty:
            continue

        cand_clearance = clearance_or_zero(candidate)
        if cand_clearance > best_clearance:
            best_clearance = cand_clearance
            best_candidate = candidate

    return best_candidate


def simplify_hole_ring(
    geometry: Polygon, hole_index: int, min_clearance: float
) -> Polygon | None:
    """Simplify a specific hole ring to resolve self-clearance issues.

    Uses Shapely's simplify on the offending hole ring with a tolerance
    derived from min_clearance, then reconstructs the polygon.
    """
    holes = list(geometry.interiors)
    if hole_index < 0 or hole_index >= len(holes):
        return None

    hole_ring = holes[hole_index]
    simplified = Polygon(hole_ring).simplify(min_clearance * 0.5)
    if simplified.is_empty or not simplified.is_valid:
        # If simplification destroys the hole, remove it
        new_holes = [h for i, h in enumerate(holes) if i != hole_index]
    else:
        new_holes = list(holes)
        new_holes[hole_index] = simplified.exterior

    try:
        result = Polygon(geometry.exterior, new_holes)
        if result.is_valid and not result.is_empty:
            return result
    except (GEOSException, ValueError):
        pass
    return None
