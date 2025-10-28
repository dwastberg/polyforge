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

from polyforge.simplify import simplify_vwp
from .core.types import MergeStrategy


def merge_close_polygons(
    polygons: List[Polygon],
    margin: float = 0.0,
    strategy: MergeStrategy = MergeStrategy.SELECTIVE_BUFFER,
    preserve_holes: bool = True,
    return_mapping: bool = False,
    insert_vertices: bool = False
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
            - MergeStrategy.SIMPLE_BUFFER: Classic expand-contract (fast, changes shape)
            - MergeStrategy.SELECTIVE_BUFFER: Only buffer near gaps (good balance, default)
            - MergeStrategy.VERTEX_MOVEMENT: Move vertices toward each other (precise)
            - MergeStrategy.BOUNDARY_EXTENSION: Extend parallel edges (best for buildings)
            - MergeStrategy.CONVEX_BRIDGES: Use convex hull bridges (smooth connections)
        preserve_holes: Whether to preserve interior holes when merging
        return_mapping: If True, return (merged_polygons, groups) where groups[i]
                       contains indices of original polygons that were merged
        insert_vertices: If True, insert vertices at optimal connection points
                        before merging. This improves bridge precision for all
                        strategies by providing exact anchor points at gaps.

    Returns:
        List of merged polygons, or (polygons, groups) if return_mapping=True

    Examples:
        >>> from polyforge.core.types import MergeStrategy
        >>> # Merge overlapping polygons only
        >>> merged = merge_close_polygons(polygons, margin=0.0)

        >>> # Merge polygons within 5 units
        >>> merged = merge_close_polygons(polygons, margin=5.0, strategy=MergeStrategy.SELECTIVE_BUFFER)

        >>> # Get mapping of which polygons were merged
        >>> merged, groups = merge_close_polygons(polygons, margin=2.0, return_mapping=True)

        >>> # Use vertex insertion for optimal bridges
        >>> merged = merge_close_polygons(polygons, margin=2.0, insert_vertices=True)

    Notes:
        - Uses spatial indexing (STRtree) for O(n log n) performance
        - Most isolated polygons are returned unchanged (fast path)
        - Groups of close polygons are merged together
        - Different strategies offer different trade-offs between speed and shape preservation
        - insert_vertices adds computational overhead but improves merge quality
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

        # Optional: Insert vertices at optimal connection points
        if insert_vertices:
            group_polygons = _insert_connection_vertices(group_polygons, margin)

        # Select and apply merge strategy
        if strategy == MergeStrategy.SIMPLE_BUFFER:
            merged = _merge_simple_buffer(group_polygons, margin, preserve_holes)
        elif strategy == MergeStrategy.SELECTIVE_BUFFER:
            merged = _merge_selective_buffer(group_polygons, margin, preserve_holes)
        elif strategy == MergeStrategy.VERTEX_MOVEMENT:
            merged = _merge_vertex_movement(group_polygons, margin, preserve_holes)
        elif strategy == MergeStrategy.BOUNDARY_EXTENSION:
            merged = _merge_boundary_extension(group_polygons, margin, preserve_holes)
        elif strategy == MergeStrategy.CONVEX_BRIDGES:
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


def _insert_connection_vertices(
    polygons: List[Polygon],
    margin: float,
    tolerance: float = 0.01
) -> List[Polygon]:
    """Insert vertices at optimal connection points between close polygons.

    For each pair of polygons within margin distance, finds the closest points
    on their boundaries. If a closest point lies on an edge (not at an existing
    vertex), inserts a new vertex at that location. This gives subsequent merge
    strategies optimal anchor points for creating minimal bridges.

    Args:
        polygons: List of polygons to process
        margin: Maximum distance for considering polygons close
        tolerance: Minimum distance from existing vertex to insert new one (default: 0.01)

    Returns:
        List of polygons with new vertices inserted at connection points

    Notes:
        - Only inserts vertices when closest point is > tolerance from existing vertices
        - Inserts at closest point per edge pair (one per close edge)
        - Preserves holes and Z-coordinates if present
    """
    if len(polygons) < 2:
        return polygons

    # Build mapping of which polygons to modify
    modified_coords = {}  # poly_idx -> new exterior coords

    # Find close edge pairs between polygons
    # This approach ensures symmetry - both polygons in a pair get vertices
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            poly_i = polygons[i]
            poly_j = polygons[j]

            distance = poly_i.distance(poly_j)
            if distance > margin:
                continue

            # Initialize coordinate lists if not already done
            if i not in modified_coords:
                modified_coords[i] = list(poly_i.exterior.coords)
            if j not in modified_coords:
                modified_coords[j] = list(poly_j.exterior.coords)

            # Get closest points between the two polygons
            from shapely.ops import nearest_points
            pt_i, pt_j = nearest_points(poly_i.boundary, poly_j.boundary)

            # Process both polygons to ensure symmetry
            insertions = []  # List of (poly_idx, edge_idx, new_vertex)

            for poly_idx, poly, pt in [(i, poly_i, pt_i), (j, poly_j, pt_j)]:
                coords = modified_coords[poly_idx]
                pt_coords = pt.coords[0]

                # Check if point is already at a vertex (within tolerance)
                is_at_vertex = False
                for coord in coords[:-1]:  # Exclude closing vertex
                    dist_to_vertex = ((coord[0] - pt_coords[0])**2 +
                                     (coord[1] - pt_coords[1])**2)**0.5
                    if dist_to_vertex < tolerance:
                        is_at_vertex = True
                        break

                if is_at_vertex:
                    continue  # Skip insertion, already at vertex

                # Find which edge the point lies on
                for k in range(len(coords) - 1):
                    seg = LineString([coords[k], coords[k + 1]])
                    dist_to_seg = seg.distance(pt)

                    if dist_to_seg < 1e-6:  # Point is on this edge
                        # Prepare vertex insertion
                        # Determine 2D or 3D
                        if len(pt_coords) == 2 and len(coords[k]) == 3:
                            # 3D coords, interpolate Z
                            seg_2d = LineString([(coords[k][0], coords[k][1]),
                                                 (coords[k+1][0], coords[k+1][1])])
                            dist_along = seg_2d.project(pt)
                            total_length = seg_2d.length
                            if total_length > 1e-10:
                                t = dist_along / total_length
                                z = coords[k][2] + t * (coords[k+1][2] - coords[k][2])
                                new_vertex = (pt_coords[0], pt_coords[1], z)
                            else:
                                new_vertex = coords[k]
                        elif len(pt_coords) == 3:
                            new_vertex = pt_coords
                        else:
                            new_vertex = pt_coords

                        insertions.append((poly_idx, k, new_vertex))
                        break

            # Apply insertions (sorted by poly_idx, then edge_idx descending to avoid index shifts)
            for poly_idx, edge_idx, new_vertex in sorted(insertions, key=lambda x: (x[0], -x[1])):
                coords = modified_coords[poly_idx]
                coords.insert(edge_idx + 1, new_vertex)
                modified_coords[poly_idx] = coords

    # Reconstruct polygons with new vertices
    result = []
    for i, poly in enumerate(polygons):
        if i in modified_coords:
            # Create new polygon with modified exterior
            new_coords = modified_coords[i]
            # Ensure closed ring
            if new_coords[0] != new_coords[-1]:
                new_coords.append(new_coords[0])

            # Preserve holes
            if poly.interiors:
                holes = [list(hole.coords) for hole in poly.interiors]
                result.append(Polygon(new_coords, holes=holes))
            else:
                result.append(Polygon(new_coords))
        else:
            # No modification needed
            result.append(poly)

    return result


# ============================================================================
# Strategy 1: Simple Buffer Union
# ============================================================================

def _merge_simple_buffer(
    group_polygons: List[Polygon],
    margin: float,
    preserve_holes: bool,
    simplify = True
) -> Union[Polygon, MultiPolygon]:
    """Merge using classic expand-contract buffer method.

    Fast and simple, but changes polygon shape (rounds corners).

    Args:
        group_polygons: Polygons to merge
        margin: Distance for buffering
        preserve_holes: Whether to preserve holes
        simplify: Whether to simplify result to reduce complexity

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

    if simplify:
        result = simplify_vwp(result, threshold=margin / 2)

    return result


# ============================================================================
# Strategy 2: Selective Buffer Union
# ============================================================================

def _merge_selective_buffer(
    group_polygons: List[Polygon],
    margin: float,
    preserve_holes: bool,
    simplify: bool = True
) -> Union[Polygon, MultiPolygon]:
    """Merge by buffering only boundaries that are close to each other.

    Better shape preservation than simple buffer.

    Args:
        group_polygons: Polygons to merge
        margin: Distance threshold
        preserve_holes: Whether to preserve holes
        simplify: Whether to simplify result to reduce complexity

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

    # Create minimal bridge zones between close segments
    buffer_zones = []

    # Filter to only the closest segment pairs to avoid over-bridging
    # Group by distance and only use segments within a tight threshold
    min_distance = min(dist for _, _, dist in close_segments)
    tolerance = min(margin * 0.2, 0.5)  # Only use segments very close to minimum
    close_segments_filtered = [
        (seg1, seg2, dist) for seg1, seg2, dist in close_segments
        if dist <= min_distance + tolerance
    ]

    for seg1, seg2, distance in close_segments_filtered:
        # Create a minimal rectangular bridge connecting the segments
        # Buffer distance should just span the gap, not the margin
        buffer_dist = distance / 2.0 + 0.1  # Just enough to overlap both sides

        # Create LineString connecting segment midpoints
        mid1 = seg1.centroid
        mid2 = seg2.centroid
        connector = LineString([mid1.coords[0], mid2.coords[0]])

        # Use minimal quad_segs for more rectangular bridges
        bridge = connector.buffer(buffer_dist, quad_segs=2)
        buffer_zones.append(bridge)

    # Union original polygons with buffer zones
    all_geoms = list(group_polygons) + buffer_zones
    result = unary_union(all_geoms)

    # Handle holes
    if not preserve_holes and isinstance(result, Polygon) and result.interiors:
        result = Polygon(result.exterior)
    elif not preserve_holes and isinstance(result, MultiPolygon):
        result = MultiPolygon([Polygon(p.exterior) for p in result.geoms])

    if simplify:
        result = simplify_vwp(result, threshold=margin / 2)

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
        # Create rectangle connecting the OVERLAPPING portions of parallel edges
        coords1 = np.array(edge1.coords)
        coords2 = np.array(edge2.coords)

        # Determine edge direction (vertical, horizontal, or angled)
        if abs(coords1[1][0] - coords1[0][0]) < 1e-6:
            # Vertical edges - find overlapping Y range
            range1_y = [min(coords1[0][1], coords1[1][1]), max(coords1[0][1], coords1[1][1])]
            range2_y = [min(coords2[0][1], coords2[1][1]), max(coords2[0][1], coords2[1][1])]

            overlap_start = max(range1_y[0], range2_y[0])
            overlap_end = min(range1_y[1], range2_y[1])

            if overlap_end - overlap_start < 1e-6:
                continue  # No overlap, skip

            # Create bridge spanning only the overlap
            x1 = coords1[0][0]
            x2 = coords2[0][0]
            bridge_coords = [
                (x1, overlap_start),
                (x2, overlap_start),
                (x2, overlap_end),
                (x1, overlap_end)
            ]

        elif abs(coords1[1][1] - coords1[0][1]) < 1e-6:
            # Horizontal edges - find overlapping X range
            range1_x = [min(coords1[0][0], coords1[1][0]), max(coords1[0][0], coords1[1][0])]
            range2_x = [min(coords2[0][0], coords2[1][0]), max(coords2[0][0], coords2[1][0])]

            overlap_start = max(range1_x[0], range2_x[0])
            overlap_end = min(range1_x[1], range2_x[1])

            if overlap_end - overlap_start < 1e-6:
                continue  # No overlap, skip

            # Create bridge spanning only the overlap
            y1 = coords1[0][1]
            y2 = coords2[0][1]
            bridge_coords = [
                (overlap_start, y1),
                (overlap_start, y2),
                (overlap_end, y2),
                (overlap_end, y1)
            ]

        else:
            # Angled edges - use original approach for general case
            p1_start, p1_end = tuple(coords1[0]), tuple(coords1[1])
            p2_start, p2_end = tuple(coords2[0]), tuple(coords2[1])

            dist_start_start = Point(p1_start).distance(Point(p2_start))
            dist_start_end = Point(p1_start).distance(Point(p2_end))
            dist_end_start = Point(p1_end).distance(Point(p2_start))
            dist_end_end = Point(p1_end).distance(Point(p2_end))

            if dist_start_start + dist_end_end < dist_start_end + dist_end_start:
                bridge_coords = [p1_start, p2_start, p2_end, p1_end]
            else:
                bridge_coords = [p1_start, p2_end, p2_start, p1_end]

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

    # Filter overlapping matches: if the same edge is matched with multiple
    # collinear segments, keep only the match with the best overlap
    if len(parallel_pairs) > 1:
        parallel_pairs = _filter_redundant_parallel_pairs(parallel_pairs)

    return parallel_pairs


def _filter_redundant_parallel_pairs(
    pairs: List[Tuple[LineString, LineString, float]]
) -> List[Tuple[LineString, LineString, float]]:
    """Filter out redundant parallel edge pairs.

    When an edge is matched with multiple collinear segments from another polygon,
    keep only the pair with the best overlap along the edge direction.

    Args:
        pairs: List of (edge1, edge2, distance) tuples

    Returns:
        Filtered list of parallel pairs
    """
    if not pairs:
        return pairs

    # Group pairs by the first edge (edges that get matched multiple times)
    from collections import defaultdict
    edge_matches = defaultdict(list)

    for edge1, edge2, dist in pairs:
        # Use edge coords as key (tuple of tuples)
        edge1_key = tuple(tuple(coord) for coord in edge1.coords)
        edge_matches[edge1_key].append((edge1, edge2, dist))

    filtered = []

    for edge1_key, matches in edge_matches.items():
        if len(matches) == 1:
            # Only one match, keep it
            filtered.append(matches[0])
        else:
            # Multiple matches - check if they're collinear segments
            # Keep only matches where edges actually overlap in projection
            edge1, _, _ = matches[0]

            # For each match, check if the second edges are collinear
            # and filter to the one with best overlap
            best_match = None
            best_overlap = 0

            for edge1_m, edge2_m, dist_m in matches:
                # Calculate overlap along the direction of edge1
                coords1 = np.array(edge1_m.coords)
                coords2 = np.array(edge2_m.coords)

                # Project edge2 endpoints onto edge1's line
                # Simple approach: use the range of coordinates
                if abs(coords1[1][0] - coords1[0][0]) < 1e-6:
                    # Vertical edge - compare Y coordinates
                    range1 = [min(coords1[0][1], coords1[1][1]), max(coords1[0][1], coords1[1][1])]
                    range2 = [min(coords2[0][1], coords2[1][1]), max(coords2[0][1], coords2[1][1])]
                else:
                    # Horizontal or angled - compare X coordinates
                    range1 = [min(coords1[0][0], coords1[1][0]), max(coords1[0][0], coords1[1][0])]
                    range2 = [min(coords2[0][0], coords2[1][0]), max(coords2[0][0], coords2[1][0])]

                # Calculate overlap
                overlap_start = max(range1[0], range2[0])
                overlap_end = min(range1[1], range2[1])
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = (edge1_m, edge2_m, dist_m)

            if best_match and best_overlap > 1e-6:
                filtered.append(best_match)

    return filtered


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

            # Get closest points between the two polygons
            pt1, pt2 = nearest_points(poly1, poly2)

            # Extract boundary points near the gap on each polygon
            # Key: use a very tight search radius to avoid collecting distant points
            # that would create diagonal bridges
            search_radius = min(margin * 0.75, distance * 2.0 + 0.5)
            boundary1_close = _get_boundary_points_near(poly1, pt1, search_radius)
            boundary2_close = _get_boundary_points_near(poly2, pt2, search_radius)

            # Need at least 2 points from each polygon
            if len(boundary1_close) < 2 or len(boundary2_close) < 2:
                # Fall back to simple point-to-point bridge
                bridge_line = LineString([pt1.coords[0], pt2.coords[0]])
                bridge = bridge_line.buffer(max(margin * 0.5, 0.1), quad_segs=4)
                if bridge.is_valid and bridge.area > 1e-10:
                    bridges.append(bridge)
                continue

            # Create bridge by finding the convex hull, but only of nearby points
            # The tight search_radius ensures we don't get distant corners
            try:
                bridge_points = []

                # Add boundary points from both polygons
                bridge_points.extend(boundary1_close)
                bridge_points.extend(boundary2_close)

                # Always include the actual closest points
                bridge_points.append(pt1.coords[0])
                bridge_points.append(pt2.coords[0])

                if len(bridge_points) >= 3:
                    # Create convex hull of the nearby points only
                    bridge = MultiPoint(bridge_points).convex_hull

                    # Buffer the bridge slightly to ensure it overlaps with both polygons
                    # This is critical - without overlap, union won't merge them
                    if isinstance(bridge, LineString):
                        # LineString needs more buffering
                        buffer_dist = max(margin * 0.3, distance * 0.5 + 0.1)
                        bridge = bridge.buffer(buffer_dist, quad_segs=4)
                    elif isinstance(bridge, Polygon):
                        # Small polygon bridge still needs buffering to ensure overlap
                        buffer_dist = max(0.1, distance * 0.05 + 0.05)
                        bridge = bridge.buffer(buffer_dist, quad_segs=4)

                    if isinstance(bridge, Polygon) and bridge.is_valid and bridge.area > 1e-10:
                        bridges.append(bridge)
            except Exception:
                # Skip if bridge creation fails
                continue

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


__all__ = [
    'merge_close_polygons',
]
