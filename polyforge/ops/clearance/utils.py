"""Utility functions for clearance operations.

This module provides geometric utility functions used by clearance fix functions.
"""

import numpy as np
from typing import List, Union
from shapely.geometry import Point
import math


def _find_nearest_vertex_index(coords: np.ndarray, point: Union[Point, tuple, np.ndarray]) -> int:
    """Find index of vertex nearest to given point.

    Args:
        coords: Array of coordinates (Nx2 or Nx3)
        point: Point to find nearest vertex to (Point, tuple, or array)

    Returns:
        Index of nearest vertex
    """
    if isinstance(point, Point):
        point_array = np.array(point.coords[0])
    elif isinstance(point, (tuple, list)):
        point_array = np.array(point)
    else:
        point_array = point

    # Only use x,y coordinates for distance calculation
    coords_2d = coords[:, :2] if coords.shape[1] > 2 else coords
    point_2d = point_array[:2] if len(point_array) > 2 else point_array

    # Exclude duplicate closing vertex
    distances = np.linalg.norm(coords_2d[:-1] - point_2d, axis=1)
    return int(np.argmin(distances))


def _find_nearest_edge_index(coords: np.ndarray, point: np.ndarray) -> int:
    """Find index of edge nearest to given point.

    Args:
        coords: Array of coordinates (Nx2 or Nx3)
        point: Point as numpy array

    Returns:
        Index of edge start vertex (edge is from index to index+1)
    """
    if len(coords) < 2:
        return 0

    coords_2d = coords[:, :2] if coords.shape[1] > 2 else coords
    point_2d = point[:2] if len(point) > 2 else point
    point_2d = np.asarray(point_2d, dtype=float)

    segment_starts = coords_2d[:-1]
    segment_ends = coords_2d[1:]
    segment_vectors = segment_ends - segment_starts
    point_vectors = point_2d - segment_starts

    seg_len_sq = np.einsum('ij,ij->i', segment_vectors, segment_vectors)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(seg_len_sq > 0, np.einsum('ij,ij->i', point_vectors, segment_vectors) / seg_len_sq, 0.0)
    t = np.clip(t, 0.0, 1.0)

    projections = segment_starts + (segment_vectors.T * t).T
    distances = np.linalg.norm(projections - point_2d, axis=1)

    degenerate = seg_len_sq == 0
    if np.any(degenerate):
        distances[degenerate] = np.linalg.norm(point_2d - segment_starts[degenerate], axis=1)

    return int(np.argmin(distances))

def _angle(p_prev, p, p_next):
    """Interior angle at vertex p in degrees"""
    v1 = np.array(p_prev) - np.array(p)
    v2 = np.array(p_next) - np.array(p)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return 180.0

    v1 /= n1
    v2 /= n2

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    ang = math.degrees(math.acos(dot))
    return ang

def _cross(o, a, b):
    """2D cross product"""
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])


def _is_concave(prev, curr, nxt, orientation=1):
    """
    Determine if vertex is concave.
    orientation = 1 for CCW polygon
    """
    cross = _cross(prev, curr, nxt)
    return cross * orientation < 0


def _dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


# ------------------------------------------------------------
# wedge tracing
# ------------------------------------------------------------

def _trace_wedge(coords, tip_idx, orientation, max_steps=200):
    """
    Trace both sides of wedge starting at tip.
    Returns indices of left/right chains.
    """

    n = len(coords)

    left = []
    right = []

    i = tip_idx
    j = tip_idx

    for _ in range(max_steps):

        i = (i - 1) % n
        j = (j + 1) % n

        left.append(i)
        right.append(j)

        # stop if they meet
        if i == j:
            break

    return left, right


def _find_best_join(coords, left_chain, right_chain):
    """
    Find closest pair between chains → neck location.

    Both chains trace from the tip all the way around the polygon, so every
    vertex except the tip appears in both.  We must skip pairs where the
    remaining body (the arc from li to ri NOT going through the tip) is too
    short to form a valid polygon — otherwise we always pick li == ri with
    distance 0.

    For left_chain position *a* and right_chain position *b* the wedge
    path through the tip has ``a + b + 2`` edges, so the body path has
    ``n - (a + b + 2)`` edges.  We require the body to have at least 2
    edges (3 vertices → a triangle).
    """

    n = len(coords)
    best = None
    best_dist = float("inf")

    for a, li in enumerate(left_chain):
        for b, ri in enumerate(right_chain):
            remaining_edges = n - (a + b + 2)
            if remaining_edges < 2:
                continue
            d = _dist(coords[li], coords[ri])
            if d < best_dist:
                best_dist = d
                best = (li, ri, d)

    if best is None:
        return None

    return best  # left_idx, right_idx, width


def _compute_depth(coords, tip_idx, left_chain, right_chain):
    """Depth = max distance from tip to midpoint of neck"""
    tip = coords[tip_idx]

    max_d = 0
    for li in left_chain:
        max_d = max(max_d, _dist(tip, coords[li]))
    for ri in right_chain:
        max_d = max(max_d, _dist(tip, coords[ri]))

    return max_d


# ------------------------------------------------------------
# splice polygon
# ------------------------------------------------------------

def _splice_polygon(coords, left_idx, right_idx):
    """
    Remove wedge region between left_idx and right_idx.
    Keeps shortest boundary path.
    """

    n = len(coords)

    if left_idx < right_idx:
        new = coords[:left_idx+1] + coords[right_idx:]
    else:
        # wrap case
        new = coords[right_idx:left_idx+1]

    # ensure closed
    if new[0] != new[-1]:
        new.append(new[0])

    return new


def _point_to_segment_distance(
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray
) -> float:
    """Calculate distance from point to line segment.

    Args:
        point: Point coordinates (2D)
        segment_start: Segment start point (2D)
        segment_end: Segment end point (2D)

    Returns:
        Distance from point to closest point on segment
    """
    # Vector from start to end
    line_vec = segment_end - segment_start
    line_len_sq = np.dot(line_vec, line_vec)

    if line_len_sq == 0:
        # Degenerate segment (start == end)
        return float(np.linalg.norm(point - segment_start))

    # Project point onto line (clamped to segment)
    t = max(0, min(1, np.dot(point - segment_start, line_vec) / line_len_sq))

    # Closest point on segment
    projection = segment_start + t * line_vec

    return float(np.linalg.norm(point - projection))


def _point_to_line_perpendicular_distance(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray
) -> float:
    """Calculate perpendicular distance from a point to an infinite line.

    Unlike _point_to_segment_distance which clamps to the segment endpoints,
    this calculates the perpendicular distance to the infinite line passing
    through the two points. Useful for aspect ratio calculations.

    Args:
        point: Point coordinates (2D)
        line_start: Point on the line (2D)
        line_end: Another point on the line (2D)

    Returns:
        Perpendicular distance from point to line (always >= 0)

    Examples:
        >>> # Point above a horizontal line
        >>> point = np.array([5.0, 3.0])
        >>> line_start = np.array([0.0, 0.0])
        >>> line_end = np.array([10.0, 0.0])
        >>> dist = _point_to_line_perpendicular_distance(point, line_start, line_end)
        >>> abs(dist - 3.0) < 0.01  # Distance is 3.0
        True
    """
    # Vector from line_start to line_end
    line_vec = line_end - line_start
    line_length = np.linalg.norm(line_vec)

    if line_length < 1e-10:
        # Degenerate line (single point)
        return float(np.linalg.norm(point - line_start))

    # Normalize line vector
    line_vec_normalized = line_vec / line_length

    # Vector from line_start to point
    point_vec = point - line_start

    # Calculate perpendicular distance using cross product formula
    # For 2D: distance = |cross product| / |line vector|
    # But since line_vec is normalized, distance = |cross product|
    cross = abs(line_vec_normalized[0] * point_vec[1] - line_vec_normalized[1] * point_vec[0])

    return float(cross)


def _get_vertex_neighborhood(
    center_idx: int,
    coords: np.ndarray,
    radius: int
) -> List[int]:
    """Get indices of vertices within radius of center vertex.

    Args:
        center_idx: Index of center vertex
        coords: Array of coordinates
        radius: Number of vertices on each side to include

    Returns:
        List of vertex indices
    """
    n = len(coords) - 1  # Exclude duplicate closing vertex
    indices = []

    for offset in range(-radius, radius + 1):
        idx = (center_idx + offset) % n
        indices.append(idx)

    return indices


def _calculate_curvature_at_vertex(coords: np.ndarray, idx: int) -> float:
    """Calculate turning angle at vertex (in degrees).

    This measures the deviation from a straight line. A value of 0 means
    the vertex continues straight, 180 means a sharp reversal.

    Args:
        coords: Array of coordinates
        idx: Vertex index

    Returns:
        Turning angle in degrees (0-180), where:
        - ~0° = straight continuation
        - ~90° = right angle turn
        - ~180° = sharp reversal/spike
    """
    n = len(coords) - 1  # Exclude closing vertex
    prev_idx = (idx - 1) % n
    next_idx = (idx + 1) % n

    # Use only 2D coordinates
    coords_2d = coords[:, :2] if coords.shape[1] > 2 else coords

    # Vectors FROM vertex
    v1 = coords_2d[prev_idx] - coords_2d[idx]  # Vector to previous
    v2 = coords_2d[next_idx] - coords_2d[idx]  # Vector to next

    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm == 0 or v2_norm == 0:
        return 0.0

    v1 = v1 / v1_norm
    v2 = v2 / v2_norm

    # Angle between the two vectors
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot)

    return float(np.degrees(angle))


def _compute_wedge_tip_angle(
    coords: np.ndarray,
    idx1: int,
    idx2: int,
    separation: int,
) -> float:
    """Compute the angle at the tip of a wedge feature (in degrees).

    Finds the vertex on the short path between idx1 and idx2 that is
    farthest from the baseline (the line through idx1 and idx2), then
    computes the angle formed at that vertex by vectors to idx1 and idx2.

    Args:
        coords: Exterior ring coordinates (closed ring, last == first).
        idx1: First clearance endpoint vertex index.
        idx2: Second clearance endpoint vertex index.
        separation: Shortest ring distance between idx1 and idx2.

    Returns:
        Tip angle in degrees.  Small values (< 20°) indicate very acute
        wedges.  Returns 180.0 when no meaningful tip can be identified
        (e.g. separation < 2 or degenerate geometry).
    """
    n = len(coords) - 1  # closed ring

    if separation < 2:
        return 180.0

    # Determine short-path direction
    fwd = (idx2 - idx1) % n
    if fwd == separation:
        step_dir = +1
    else:
        step_dir = -1

    # Collect interior vertices on the short path (excluding endpoints)
    short_path_indices = [(idx1 + step_dir * s) % n for s in range(1, separation)]

    if not short_path_indices:
        return 180.0

    coords_2d = coords[:, :2] if coords.shape[1] > 2 else coords
    p1 = coords_2d[idx1]
    p2 = coords_2d[idx2]
    baseline = p2 - p1
    baseline_len = float(np.linalg.norm(baseline))

    if baseline_len < 1e-12:
        # Degenerate baseline — endpoints coincide
        tip_idx = short_path_indices[0]
    else:
        baseline_unit = baseline / baseline_len
        max_dist = -1.0
        tip_idx = short_path_indices[0]
        for vi in short_path_indices:
            v = coords_2d[vi] - p1
            perp = abs(float(v[0] * baseline_unit[1] - v[1] * baseline_unit[0]))
            if perp > max_dist:
                max_dist = perp
                tip_idx = vi

        if max_dist < 1e-12:
            # All short-path vertices are collinear with the baseline
            return 180.0

    tip = coords_2d[tip_idx]
    va = p1 - tip
    vb = p2 - tip
    la = float(np.linalg.norm(va))
    lb = float(np.linalg.norm(vb))

    if la < 1e-12 or lb < 1e-12:
        return 180.0

    cos_angle = float(np.clip(np.dot(va, vb) / (la * lb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _remove_vertices_between(
    coords: np.ndarray,
    start_idx: int,
    end_idx: int
) -> np.ndarray:
    """Remove all vertices between start and end, creating straight edge.

    Args:
        coords: Original coordinates
        start_idx: Start index (kept)
        end_idx: End index (kept)

    Returns:
        New coordinate array with interior vertices removed
    """
    n = len(coords)

    if start_idx < end_idx:
        # Simple case: continuous range
        new_coords = np.vstack([
            coords[:start_idx + 1],
            coords[end_idx:]
        ])
    else:
        # Wrapped case: range crosses array boundary
        new_coords = coords[end_idx:start_idx + 1]

    return new_coords
