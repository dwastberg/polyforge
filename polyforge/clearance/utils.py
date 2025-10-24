"""Utility functions for clearance operations.

This module provides geometric utility functions used by clearance fix functions.
"""

import numpy as np
from typing import List, Union
from shapely.geometry import Point


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
    min_dist = float('inf')
    nearest_idx = 0

    for i in range(len(coords) - 1):
        edge_start = coords[i, :2] if coords.shape[1] > 2 else coords[i]
        edge_end = coords[i + 1, :2] if coords.shape[1] > 2 else coords[i + 1]
        point_2d = point[:2] if len(point) > 2 else point

        dist = _point_to_segment_distance(point_2d, edge_start, edge_end)

        if dist < min_dist:
            min_dist = dist
            nearest_idx = i

    return nearest_idx


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
