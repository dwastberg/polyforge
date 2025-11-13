"""Common spatial operation utilities.

This module provides reusable utilities for spatial operations, indexing,
and geometric calculations to eliminate code duplication.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
from shapely.ops import nearest_points


def find_polygon_pairs(
    polygons: List[Polygon],
    margin: float = 0.0,
    predicate: str = 'intersects',
    validate_func: Optional[Callable[[Polygon, Polygon], bool]] = None,
    tree: Optional[STRtree] = None,
) -> List[Tuple[int, int]]:
    """Find pairs of polygons that satisfy spatial predicate within margin.

    Uses STRtree for efficient spatial indexing (O(n log n) instead of O(n²)).
    Returns unique pairs (i, j) where i < j.

    Args:
        polygons: List of polygons to search
        margin: Distance margin for proximity (0 = touching/overlapping only)
        predicate: Shapely spatial predicate ('intersects', 'within', etc.)
        validate_func: Optional function (poly_i, poly_j) -> bool for additional validation

    Returns:
        List of (index_i, index_j) tuples for polygon pairs that match criteria

    Examples:
        >>> polygons = [poly1, poly2, poly3, poly4]
        >>> # Find all overlapping pairs
        >>> pairs = find_polygon_pairs(polygons)

        >>> # Find pairs within 5 units
        >>> def within_distance(p1, p2):
        ...     return p1.distance(p2) <= 5.0
        >>> pairs = find_polygon_pairs(polygons, margin=5.0, validate_func=within_distance)
    """
    if not polygons:
        return []

    if tree is None:
        tree = STRtree(polygons)

    # Track checked pairs to avoid duplicates
    pairs = []
    checked: Set[Tuple[int, int]] = set()

    for i in range(len(polygons)):
        poly_i = polygons[i]

        # Query with distance predicate if possible
        if margin > 0 and (predicate is None or predicate == 'intersects'):
            candidate_indices = tree.query(poly_i, predicate='dwithin', distance=margin)
        else:
            search_geom = poly_i if margin <= 0 else poly_i.buffer(margin)
            candidate_indices = tree.query(search_geom, predicate=predicate)

        for j in candidate_indices:
            # Only process each pair once (i < j)
            if j > i:
                pair = (i, j)
                if pair in checked:
                    continue
                checked.add(pair)

                poly_j = polygons[j]

                # Apply validation function if provided
                if validate_func is None or validate_func(poly_i, poly_j):
                    pairs.append(pair)

    return pairs


def find_nearest_boundary_point(
    point: Point,
    geometry: BaseGeometry
) -> Point:
    """Find the nearest point on a geometry's boundary to a given point.

    Args:
        point: Reference point
        geometry: Geometry whose boundary to search

    Returns:
        Nearest point on geometry's boundary

    Examples:
        >>> poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        >>> point = Point(5, 5)
        >>> nearest = find_nearest_boundary_point(point, poly.boundary)
        >>> # Returns point on polygon edge nearest to (5, 5)
    """
    if hasattr(geometry, 'boundary'):
        _, nearest_pt = nearest_points(point, geometry.boundary)
    else:
        _, nearest_pt = nearest_points(point, geometry)

    return nearest_pt


def point_to_segment_projection(
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray
) -> Tuple[float, np.ndarray, float]:
    """Project a point onto a line segment and calculate distance.

    Uses parametric representation: P(t) = start + t * (end - start)
    where t is clamped to [0, 1] for segment projection.

    Args:
        point: Point coordinates (2D or 3D, only first 2 used)
        segment_start: Segment start point (2D or 3D, only first 2 used)
        segment_end: Segment end point (2D or 3D, only first 2 used)

    Returns:
        Tuple of (parameter t, projected_point, distance)
        - t: Parameter value in [0, 1] where projection occurs
        - projected_point: Closest point on segment to input point
        - distance: Distance from point to projected_point

    Examples:
        >>> point = np.array([0.5, 1.0])
        >>> seg_start = np.array([0.0, 0.0])
        >>> seg_end = np.array([1.0, 0.0])
        >>> t, proj, dist = point_to_segment_projection(point, seg_start, seg_end)
        >>> t
        0.5
        >>> proj
        array([0.5, 0.0])
        >>> dist
        1.0
    """
    # Extract 2D coordinates
    pt = point[:2]
    seg_start_2d = segment_start[:2]
    seg_end_2d = segment_end[:2]

    # Vector from start to end
    line_vec = seg_end_2d - seg_start_2d
    line_len_sq = np.dot(line_vec, line_vec)

    # Handle degenerate segment (start == end)
    if line_len_sq < 1e-10:
        distance = float(np.linalg.norm(pt - seg_start_2d))
        return 0.0, seg_start_2d.copy(), distance

    # Project point onto line (parametric representation)
    t = np.dot(pt - seg_start_2d, line_vec) / line_len_sq

    # Clamp to segment bounds [0, 1]
    t = max(0.0, min(1.0, t))

    # Calculate projected point
    projection = seg_start_2d + t * line_vec

    # Calculate distance
    distance = float(np.linalg.norm(pt - projection))

    return t, projection, distance


def build_adjacency_graph(
    polygons: List[Polygon],
    margin: float,
    tree: Optional[STRtree] = None
) -> dict[int, Set[int]]:
    """Build adjacency graph of polygons within margin distance.

    Creates a graph where nodes are polygon indices and edges connect
    polygons that are within margin distance of each other.

    Args:
        polygons: List of polygons
        margin: Maximum distance for adjacency
        tree: Optional pre-built STRtree (will build if not provided)

    Returns:
        Dictionary mapping polygon index to set of adjacent polygon indices

    Examples:
        >>> adjacency = build_adjacency_graph(polygons, margin=5.0)
        >>> # adjacency[0] = {1, 3}  means polygon 0 is close to 1 and 3
    """
    if not polygons:
        return {}

    # Build spatial index if not provided
    if tree is None:
        tree = STRtree(polygons)

    # Initialize adjacency dict
    adjacency: dict[int, Set[int]] = {i: set() for i in range(len(polygons))}

    for i in range(len(polygons)):
        poly_i = polygons[i]

        # Query spatial index
        if margin > 0:
            candidate_indices = tree.query(poly_i, predicate='dwithin', distance=margin)
        else:
            candidate_indices = tree.query(poly_i, predicate='intersects')

        # Check actual distance for each candidate
        for j in candidate_indices:
            if i != j and j not in adjacency[i]:
                distance = polygons[i].distance(polygons[j])
                if distance <= margin:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

    return adjacency


def find_connected_components(
    adjacency: dict[int, Set[int]]
) -> List[List[int]]:
    """Find connected components in an adjacency graph using DFS.

    Args:
        adjacency: Adjacency graph (dict of node -> set of neighbors)

    Returns:
        List of components, where each component is a list of node indices

    Examples:
        >>> adjacency = {0: {1}, 1: {0}, 2: {3}, 3: {2}, 4: set()}
        >>> components = find_connected_components(adjacency)
        >>> components
        [[0, 1], [2, 3], [4]]
    """
    visited = set()
    components = []

    def dfs(node: int, current_component: List[int]):
        """Depth-first search to find connected component."""
        visited.add(node)
        current_component.append(node)
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor, current_component)

    for node in adjacency:
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(sorted(component))

    return components


def iterate_unique_pairs(items: Sequence) -> Iterable[Tuple[int, int]]:
    """Yield unique index pairs (i, j) with i < j for the given sequence."""
    length = len(items)
    for i in range(length):
        for j in range(i + 1, length):
            yield i, j


__all__ = [
    'find_polygon_pairs',
    'find_nearest_boundary_point',
    'point_to_segment_projection',
    'build_adjacency_graph',
    'find_connected_components',
    'iterate_unique_pairs',
    'SegmentIndex',
    'build_segment_index',
    'query_close_segments',
]
@dataclass
class SegmentIndex:
    """Spatial index of polygon boundary segments."""

    segments: List[LineString]
    owners: List[Tuple[int, int]]  # (polygon_index, edge_index)
    tree: STRtree


def build_segment_index(
    polygons: List[Polygon],
    segment_length: float,
) -> SegmentIndex:
    """Discretize polygon boundaries into segments and build an STRtree."""
    segments: List[LineString] = []
    owners: List[Tuple[int, int]] = []

    for poly_idx, poly in enumerate(polygons):
        coords = list(poly.exterior.coords)
        for edge_idx in range(len(coords) - 1):
            seg = LineString([coords[edge_idx], coords[edge_idx + 1]])
            seg_len = seg.length
            if seg_len <= segment_length + 1e-9:
                segments.append(seg)
                owners.append((poly_idx, edge_idx))
                continue

            splits = max(1, int(np.ceil(seg_len / segment_length)))
            for j in range(splits):
                start_t = j / splits
                end_t = (j + 1) / splits
                subseg = LineString(
                    [
                        seg.interpolate(start_t, normalized=True).coords[0],
                        seg.interpolate(end_t, normalized=True).coords[0],
                    ]
                )
                segments.append(subseg)
                owners.append((poly_idx, edge_idx))

    tree = STRtree(segments) if segments else STRtree([])
    return SegmentIndex(segments=segments, owners=owners, tree=tree)


def query_close_segments(
    index: SegmentIndex,
    seg_idx: int,
    margin: float,
) -> List[int]:
    """Return indices of segments within ``margin`` distance."""
    segment = index.segments[seg_idx]
    matches = index.tree.query(segment, predicate="dwithin", distance=margin)
    return [j for j in matches if j != seg_idx]
