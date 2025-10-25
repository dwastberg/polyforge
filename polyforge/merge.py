"""Functions for merging close or overlapping polygons.

This module provides efficient algorithms for merging polygons that are
either overlapping or within a specified distance (margin) of each other.
Uses spatial indexing for O(n log n) performance.
"""

from typing import List, Tuple, Optional, Union, Set
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, MultiPoint
from shapely.strtree import STRtree
from shapely.ops import unary_union, nearest_points
from shapely.geometry.base import BaseGeometry


def merge_close_polygons(
    polygons: List[Polygon],
    margin: float = 0.0,
    strategy: str = 'selective_buffer',
    preserve_holes: bool = True,
    return_mapping: bool = False
) -> Union[List[Polygon], Tuple[List[Polygon], List[List[int]]]]:
    """Merge polygons that overlap or are within margin distance.

    This function efficiently identifies and merges polygons that are close to
    each other. In typical real-world cases where most polygons are isolated,
    the function uses spatial indexing to quickly identify and return unchanged
    the polygons that don't need merging (90-99% in typical cases).

    Only polygons that are actually close (within margin distance) are processed
    and merged according to the selected strategy.

    Args:
        polygons: List of input polygons
        margin: Maximum distance for merging (0.0 = only overlapping polygons)
        strategy: Merging strategy:
            - 'simple_buffer': Classic expand-contract (fast, changes shape)
            - 'selective_buffer': Only buffer near gaps (good balance, default)
            - 'vertex_movement': Move vertices toward each other (precise)
            - 'boundary_extension': Extend parallel edges (best for buildings)
            - 'convex_bridges': Use convex hull bridges (smooth connections)
        preserve_holes: Whether to preserve interior holes when merging
        return_mapping: If True, return (merged_polygons, groups) where groups[i]
                       contains indices of original polygons that were merged

    Returns:
        List of merged polygons, or (polygons, groups) if return_mapping=True

    Examples:
        >>> # Merge overlapping polygons only
        >>> merged = merge_close_polygons(polygons, margin=0.0)

        >>> # Merge polygons within 5 units
        >>> merged = merge_close_polygons(polygons, margin=5.0, strategy='selective_buffer')

        >>> # Get mapping of which polygons were merged
        >>> merged, groups = merge_close_polygons(polygons, margin=2.0, return_mapping=True)

    Notes:
        - Uses spatial indexing (STRtree) for O(n log n) performance
        - Most isolated polygons are returned unchanged (fast path)
        - Groups of close polygons are merged together
        - Different strategies offer different trade-offs between speed and shape preservation
    """
    if not polygons:
        return ([], []) if return_mapping else []

    # Phase 1: Find close polygon groups using spatial indexing
    isolated_indices, merge_groups = _find_close_polygon_groups(polygons, margin)

    # Build result list and mapping
    result = []
    mapping = []

    # Fast path: Add isolated polygons unchanged
    for idx in isolated_indices:
        result.append(polygons[idx])
        if return_mapping:
            mapping.append([idx])

    # Phase 2: Merge each group using selected strategy
    for group_indices in merge_groups:
        group_polygons = [polygons[i] for i in group_indices]

        # Select and apply merge strategy
        if strategy == 'simple_buffer':
            merged = _merge_simple_buffer(group_polygons, margin, preserve_holes)
        elif strategy == 'selective_buffer':
            merged = _merge_selective_buffer(group_polygons, margin, preserve_holes)
        elif strategy == 'vertex_movement':
            merged = _merge_vertex_movement(group_polygons, margin, preserve_holes)
        elif strategy == 'boundary_extension':
            merged = _merge_boundary_extension(group_polygons, margin, preserve_holes)
        elif strategy == 'convex_bridges':
            merged = _merge_convex_bridges(group_polygons, margin, preserve_holes)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Handle MultiPolygon results
        if isinstance(merged, MultiPolygon):
            # Split into separate polygons
            for poly in merged.geoms:
                result.append(poly)
                if return_mapping:
                    mapping.append(group_indices)
        elif isinstance(merged, Polygon):
            result.append(merged)
            if return_mapping:
                mapping.append(group_indices)

    if return_mapping:
        return result, mapping
    return result


def _find_close_polygon_groups(
    polygons: List[Polygon],
    margin: float
) -> Tuple[List[int], List[List[int]]]:
    """Find groups of polygons that are within margin distance of each other.

    Uses spatial indexing (STRtree) for efficient proximity detection.
    Returns isolated polygons separately for fast-path processing.

    Args:
        polygons: List of polygons
        margin: Maximum distance for grouping

    Returns:
        Tuple of (isolated_indices, merge_groups) where:
        - isolated_indices: List of indices of polygons with no close neighbors
        - merge_groups: List of groups, each group is a list of polygon indices

    Performance:
        O(n log n) using spatial index, compared to O(n²) for naive approach
    """
    if not polygons:
        return [], []

    n = len(polygons)

    # Build spatial index
    tree = STRtree(polygons)

    # Build adjacency graph of close polygons
    adjacency: dict[int, Set[int]] = {i: set() for i in range(n)}

    for i in range(n):
        poly_i = polygons[i]

        # Query spatial index with buffered polygon for proximity
        if margin > 0:
            # Buffer to find candidates within margin
            # Use a slightly larger buffer for the query to ensure we catch everything
            search_geom = poly_i.buffer(margin * 1.01)
            candidate_indices = tree.query(search_geom, predicate='intersects')
        else:
            # Only find overlapping polygons
            candidate_indices = tree.query(poly_i, predicate='intersects')

        # Check actual distance for each candidate
        for j in candidate_indices:
            if i != j and j not in adjacency[i]:
                # Check actual distance
                distance = polygons[i].distance(polygons[j])
                if distance <= margin:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

    # Find connected components using DFS
    visited = set()
    groups = []

    def dfs(node: int, current_group: List[int]):
        """Depth-first search to find connected component."""
        visited.add(node)
        current_group.append(node)
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                dfs(neighbor, current_group)

    for i in range(n):
        if i not in visited:
            group = []
            dfs(i, group)
            groups.append(sorted(group))

    # Separate isolated polygons from merge groups
    isolated = [g[0] for g in groups if len(g) == 1]
    to_merge = [g for g in groups if len(g) > 1]

    return isolated, to_merge


# ============================================================================
# Strategy 1: Simple Buffer Union
# ============================================================================

def _merge_simple_buffer(
    group_polygons: List[Polygon],
    margin: float,
    preserve_holes: bool
) -> Union[Polygon, MultiPolygon]:
    """Merge using classic expand-contract buffer method.

    Fast and simple, but changes polygon shape (rounds corners).

    Args:
        group_polygons: Polygons to merge
        margin: Distance for buffering
        preserve_holes: Whether to preserve holes

    Returns:
        Merged polygon(s)
    """
    if len(group_polygons) == 1:
        return group_polygons[0]

    # For overlapping polygons (margin=0), just use unary_union
    if margin <= 0:
        return unary_union(group_polygons)

    # Expand all polygons by margin/2
    buffer_dist = margin / 2.0
    expanded = [p.buffer(buffer_dist, quad_segs=16) for p in group_polygons]

    # Union all expanded polygons
    merged = unary_union(expanded)

    # Contract back by margin/2
    result = merged.buffer(-buffer_dist, quad_segs=16)

    # Handle holes
    if not preserve_holes and isinstance(result, Polygon) and result.interiors:
        # Remove holes
        result = Polygon(result.exterior)
    elif not preserve_holes and isinstance(result, MultiPolygon):
        # Remove holes from all polygons
        result = MultiPolygon([Polygon(p.exterior) for p in result.geoms])

    return result


# ============================================================================
# Strategy 2: Selective Buffer Union
# ============================================================================

def _merge_selective_buffer(
    group_polygons: List[Polygon],
    margin: float,
    preserve_holes: bool
) -> Union[Polygon, MultiPolygon]:
    """Merge by buffering only boundaries that are close to each other.

    Better shape preservation than simple buffer.

    Args:
        group_polygons: Polygons to merge
        margin: Distance threshold
        preserve_holes: Whether to preserve holes

    Returns:
        Merged polygon(s)
    """
    if len(group_polygons) == 1:
        return group_polygons[0]

    # For overlapping polygons, just use unary_union
    if margin <= 0:
        return unary_union(group_polygons)

    # Find close boundary segment pairs
    close_segments = _find_close_boundary_pairs(group_polygons, margin)

    if not close_segments:
        # No close segments, just union
        return unary_union(group_polygons)

    # Create buffer zones around close segments
    buffer_zones = []
    for seg1, seg2, distance in close_segments:
        # Create a thin buffer connecting the segments
        # Use the actual distance to create appropriate buffer
        buffer_dist = (margin - distance) / 2.0 + distance / 2.0

        # Create LineString connecting segment midpoints
        mid1 = seg1.centroid
        mid2 = seg2.centroid
        connector = LineString([mid1.coords[0], mid2.coords[0]])

        # Buffer the connector
        bridge = connector.buffer(buffer_dist, quad_segs=8)
        buffer_zones.append(bridge)

    # Union original polygons with buffer zones
    all_geoms = list(group_polygons) + buffer_zones
    result = unary_union(all_geoms)

    # Handle holes
    if not preserve_holes and isinstance(result, Polygon) and result.interiors:
        result = Polygon(result.exterior)
    elif not preserve_holes and isinstance(result, MultiPolygon):
        result = MultiPolygon([Polygon(p.exterior) for p in result.geoms])

    return result


def _find_close_boundary_pairs(
    polygons: List[Polygon],
    margin: float,
    segment_length: Optional[float] = None
) -> List[Tuple[LineString, LineString, float]]:
    """Find pairs of boundary segments that are within margin distance.

    Args:
        polygons: List of polygons
        margin: Maximum distance threshold
        segment_length: Optional length for discretizing boundaries (auto if None)

    Returns:
        List of (segment1, segment2, distance) tuples
    """
    # Auto-determine segment length based on margin
    if segment_length is None:
        segment_length = margin * 2.0

    # Extract boundary segments from all polygons
    all_segments = []
    for poly_idx, poly in enumerate(polygons):
        coords = list(poly.exterior.coords)

        # Discretize boundary into segments
        for i in range(len(coords) - 1):
            seg = LineString([coords[i], coords[i + 1]])

            # Further subdivide long segments
            seg_len = seg.length
            if seg_len > segment_length:
                # Split into smaller segments
                num_splits = int(np.ceil(seg_len / segment_length))
                for j in range(num_splits):
                    start = j / num_splits
                    end = (j + 1) / num_splits
                    subseg = LineString([
                        seg.interpolate(start, normalized=True).coords[0],
                        seg.interpolate(end, normalized=True).coords[0]
                    ])
                    all_segments.append((poly_idx, subseg))
            else:
                all_segments.append((poly_idx, seg))

    # Find close segment pairs from different polygons
    close_pairs = []
    for i, (poly_idx_i, seg_i) in enumerate(all_segments):
        for j, (poly_idx_j, seg_j) in enumerate(all_segments[i + 1:], i + 1):
            # Only consider segments from different polygons
            if poly_idx_i != poly_idx_j:
                distance = seg_i.distance(seg_j)
                if distance <= margin:
                    close_pairs.append((seg_i, seg_j, distance))

    return close_pairs


# ============================================================================
# Strategy 3: Vertex Movement
# ============================================================================

def _merge_vertex_movement(
    group_polygons: List[Polygon],
    margin: float,
    preserve_holes: bool
) -> Union[Polygon, MultiPolygon]:
    """Merge by moving vertices of close polygons toward each other.

    Precise control, preserves overall structure.

    Args:
        group_polygons: Polygons to merge
        margin: Distance threshold
        preserve_holes: Whether to preserve holes

    Returns:
        Merged polygon(s)
    """
    if len(group_polygons) == 1:
        return group_polygons[0]

    # For overlapping polygons, just use unary_union
    if margin <= 0:
        return unary_union(group_polygons)

    modified_polygons = []

    for poly_idx, poly in enumerate(group_polygons):
        coords = np.array(poly.exterior.coords)
        new_coords = coords.copy()

        # For each vertex, check if it's close to another polygon
        for i in range(len(coords) - 1):  # Exclude closing vertex
            vertex = Point(coords[i])

            # Find closest point on other polygons
            min_dist = float('inf')
            closest_point = None

            for other_idx, other_poly in enumerate(group_polygons):
                if other_idx == poly_idx:
                    continue

                # Use Shapely's distance for efficiency
                dist = vertex.distance(other_poly)

                if dist < min_dist and dist <= margin:
                    min_dist = dist
                    # Get actual closest point on boundary
                    _, closest_pt = nearest_points(vertex, other_poly.boundary)
                    closest_point = closest_pt

            # Move vertex toward closest point
            if closest_point is not None and min_dist <= margin:
                move_vector = np.array(closest_point.coords[0]) - coords[i][:2]
                # Move vertex all the way to create overlap
                new_coords[i][:2] = coords[i][:2] + move_vector

        # Create modified polygon, preserving holes if needed
        if preserve_holes and poly.interiors:
            holes = [np.array(hole.coords) for hole in poly.interiors]
            modified_polygons.append(Polygon(new_coords, holes=holes))
        else:
            modified_polygons.append(Polygon(new_coords))

    # Union the modified polygons
    result = unary_union(modified_polygons)

    # Validate and fix if needed
    if not result.is_valid:
        result = result.buffer(0)

    return result


# ============================================================================
# Strategy 4: Boundary Extension
# ============================================================================

def _merge_boundary_extension(
    group_polygons: List[Polygon],
    margin: float,
    preserve_holes: bool
) -> Union[Polygon, MultiPolygon]:
    """Merge by extending parallel edges toward each other.

    Best for rectangular/architectural features.

    Args:
        group_polygons: Polygons to merge
        margin: Distance threshold
        preserve_holes: Whether to preserve holes

    Returns:
        Merged polygon(s)
    """
    if len(group_polygons) == 1:
        return group_polygons[0]

    # For overlapping polygons, just use unary_union
    if margin <= 0:
        return unary_union(group_polygons)

    # Find parallel close edge pairs
    parallel_edges = _find_parallel_close_edges(group_polygons, margin)

    if not parallel_edges:
        # No parallel edges, fall back to selective buffer
        return _merge_selective_buffer(group_polygons, margin, preserve_holes)

    # Create rectangular bridges between parallel edges
    bridges = []
    for edge1, edge2, distance in parallel_edges:
        # Create rectangle connecting the two edges
        coords1 = list(edge1.coords)
        coords2 = list(edge2.coords)

        # Check which end of edge2 is closer to which end of edge1
        # to ensure correct winding order
        p1_start, p1_end = coords1[0], coords1[1]
        p2_start, p2_end = coords2[0], coords2[1]

        # Calculate distances to determine orientation
        dist_start_start = Point(p1_start).distance(Point(p2_start))
        dist_start_end = Point(p1_start).distance(Point(p2_end))

        # Choose orientation that keeps edges close
        if dist_start_start < dist_start_end:
            # Same orientation
            bridge_coords = [p1_start, p1_end, p2_end, p2_start]
        else:
            # Opposite orientation
            bridge_coords = [p1_start, p1_end, p2_start, p2_end]

        try:
            bridge = Polygon(bridge_coords)
            if bridge.is_valid and bridge.area > 1e-10:
                bridges.append(bridge)
        except Exception:
            # Skip invalid bridges
            continue

    # Union polygons with bridges
    all_geoms = list(group_polygons) + bridges
    result = unary_union(all_geoms)

    # Handle holes
    if not preserve_holes and isinstance(result, Polygon) and result.interiors:
        result = Polygon(result.exterior)
    elif not preserve_holes and isinstance(result, MultiPolygon):
        result = MultiPolygon([Polygon(p.exterior) for p in result.geoms])

    return result


def _find_parallel_close_edges(
    polygons: List[Polygon],
    margin: float,
    angle_threshold: float = 15.0
) -> List[Tuple[LineString, LineString, float]]:
    """Find pairs of parallel edges that are close to each other.

    Args:
        polygons: List of polygons
        margin: Maximum distance threshold
        angle_threshold: Maximum angle difference in degrees (default: 15)

    Returns:
        List of (edge1, edge2, distance) tuples
    """
    if len(polygons) < 2:
        return []

    # Extract edges from all polygons
    all_edges = []
    for poly_idx, poly in enumerate(polygons):
        coords = list(poly.exterior.coords)
        for i in range(len(coords) - 1):
            edge = LineString([coords[i], coords[i + 1]])
            # Filter out degenerate edges
            if edge.length > 1e-10:
                all_edges.append((poly_idx, edge))

    if not all_edges:
        return []

    # Find parallel close edge pairs
    parallel_pairs = []
    angle_threshold_rad = np.radians(angle_threshold)

    for i, (poly_idx_i, edge_i) in enumerate(all_edges):
        for j, (poly_idx_j, edge_j) in enumerate(all_edges[i + 1:], i + 1):
            # Only consider edges from different polygons
            if poly_idx_i != poly_idx_j:
                distance = edge_i.distance(edge_j)
                if distance <= margin:
                    # Check if edges are parallel
                    # Calculate edge directions
                    coords_i = np.array(edge_i.coords)
                    coords_j = np.array(edge_j.coords)

                    dir_i = coords_i[1] - coords_i[0]
                    dir_j = coords_j[1] - coords_j[0]

                    # Normalize
                    len_i = np.linalg.norm(dir_i)
                    len_j = np.linalg.norm(dir_j)

                    if len_i < 1e-10 or len_j < 1e-10:
                        continue

                    dir_i = dir_i / len_i
                    dir_j = dir_j / len_j

                    # Calculate angle between edges (consider both orientations)
                    dot_product = np.abs(np.dot(dir_i, dir_j))
                    dot_product = np.clip(dot_product, -1.0, 1.0)
                    angle = np.arccos(dot_product)

                    # Check if parallel (angle close to 0 or 180 degrees)
                    if angle <= angle_threshold_rad or angle >= (np.pi - angle_threshold_rad):
                        parallel_pairs.append((edge_i, edge_j, distance))

    return parallel_pairs


# ============================================================================
# Strategy 5: Convex Hull Bridges
# ============================================================================

def _merge_convex_bridges(
    group_polygons: List[Polygon],
    margin: float,
    preserve_holes: bool
) -> Union[Polygon, MultiPolygon]:
    """Merge using convex hull of close boundary regions.

    Creates smooth connections for irregular gaps.

    Args:
        group_polygons: Polygons to merge
        margin: Distance threshold
        preserve_holes: Whether to preserve holes

    Returns:
        Merged polygon(s)
    """
    if len(group_polygons) == 1:
        return group_polygons[0]

    # For overlapping polygons, just use unary_union
    if margin <= 0:
        return unary_union(group_polygons)

    bridges = []

    # Find close pairs and create convex bridges
    for i, poly1 in enumerate(group_polygons):
        for j, poly2 in enumerate(group_polygons[i + 1:], i + 1):
            distance = poly1.distance(poly2)
            if distance > margin:
                continue

            # Get closest points
            pt1, pt2 = nearest_points(poly1, poly2)

            # Extract boundary points near the gap
            boundary1_close = _get_boundary_points_near(poly1, pt1, margin * 2.0)
            boundary2_close = _get_boundary_points_near(poly2, pt2, margin * 2.0)

            if len(boundary1_close) < 2 or len(boundary2_close) < 2:
                continue

            # Convex hull of close points
            all_close_points = boundary1_close + boundary2_close
            if len(all_close_points) >= 3:
                bridge = MultiPoint(all_close_points).convex_hull
                if isinstance(bridge, Polygon) and bridge.area > 1e-10:
                    bridges.append(bridge)

    # Union all
    all_geoms = list(group_polygons) + bridges
    result = unary_union(all_geoms)

    # Handle holes
    if not preserve_holes and isinstance(result, Polygon) and result.interiors:
        result = Polygon(result.exterior)
    elif not preserve_holes and isinstance(result, MultiPolygon):
        result = MultiPolygon([Polygon(p.exterior) for p in result.geoms])

    return result


def _get_boundary_points_near(
    polygon: Polygon,
    point: Point,
    radius: float
) -> List[Tuple[float, float]]:
    """Extract boundary points within radius of a given point.

    Args:
        polygon: Input polygon
        point: Reference point
        radius: Search radius

    Returns:
        List of coordinate tuples
    """
    coords = list(polygon.exterior.coords)
    close_points = []

    for coord in coords[:-1]:  # Exclude duplicate closing point
        coord_point = Point(coord)
        if coord_point.distance(point) <= radius:
            close_points.append(coord)

    # If we didn't find enough points, also sample along the boundary
    if len(close_points) < 3:
        # Sample points along boundary near the reference point
        boundary = polygon.exterior
        num_samples = max(10, int(boundary.length / 2))

        for i in range(num_samples):
            t = i / num_samples
            sampled_point = boundary.interpolate(t, normalized=True)
            if sampled_point.distance(point) <= radius:
                close_points.append((sampled_point.x, sampled_point.y))

    return close_points
