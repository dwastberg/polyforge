"""Core merge orchestration logic."""

from typing import List, Tuple, Union, Set
from shapely.geometry import Polygon, MultiPolygon
from shapely.strtree import STRtree

from ..core.types import MergeStrategy
from .strategies.simple_buffer import merge_simple_buffer
from .strategies.selective_buffer import merge_selective_buffer
from .strategies.vertex_movement import merge_vertex_movement
from .strategies.boundary_extension import merge_boundary_extension
from .strategies.convex_bridges import merge_convex_bridges
from .utils.vertex_insertion import insert_connection_vertices


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
    isolated_indices, merge_groups = find_close_polygon_groups(polygons, margin)

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
            group_polygons = insert_connection_vertices(group_polygons, margin)

        # Select and apply merge strategy
        if strategy == MergeStrategy.SIMPLE_BUFFER:
            merged = merge_simple_buffer(group_polygons, margin, preserve_holes)
        elif strategy == MergeStrategy.SELECTIVE_BUFFER:
            merged = merge_selective_buffer(group_polygons, margin, preserve_holes)
        elif strategy == MergeStrategy.VERTEX_MOVEMENT:
            merged = merge_vertex_movement(group_polygons, margin, preserve_holes)
        elif strategy == MergeStrategy.BOUNDARY_EXTENSION:
            merged = merge_boundary_extension(group_polygons, margin, preserve_holes)
        elif strategy == MergeStrategy.CONVEX_BRIDGES:
            merged = merge_convex_bridges(group_polygons, margin, preserve_holes)
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


def find_close_polygon_groups(
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


__all__ = ['merge_close_polygons', 'find_close_polygon_groups']
