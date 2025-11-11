"""Core merge orchestration logic."""

from typing import List, Tuple, Union
from shapely.geometry import Polygon, MultiPolygon

from ..core.types import MergeStrategy, coerce_enum
from ..core.spatial_utils import build_adjacency_graph, find_connected_components
from polyforge.ops.merge import (
    merge_simple_buffer,
    merge_selective_buffer,
    merge_vertex_movement,
    merge_boundary_extension,
    merge_convex_bridges,
    insert_connection_vertices,
)


def merge_close_polygons(
    polygons: List[Polygon],
    margin: float = 0.0,
    merge_strategy: Union[MergeStrategy, str] = MergeStrategy.SELECTIVE_BUFFER,
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
        merge_strategy: Merging strategy (enum or string literal):
            - MergeStrategy.SIMPLE_BUFFER: Classic expand-contract (fast, changes shape)
            - MergeStrategy.SELECTIVE_BUFFER: Only buffer near gaps (good balance, default)
            - MergeStrategy.VERTEX_MOVEMENT: Move vertices toward each other (precise)
            - MergeStrategy.BOUNDARY_EXTENSION: Extend parallel edges (best for buildings)
            - MergeStrategy.CONVEX_BRIDGES: Use convex hull bridges (smooth connections)
          String values should match the enum value names (e.g., ``"selective_buffer"``).
        preserve_holes: Whether to preserve interior holes when merging
        return_mapping: If True, return (merged_polygons, groups) where groups[i]
                       contains indices of original polygons that were merged
        insert_vertices: If True, insert vertices at optimal connection points
                        before merging. This improves bridge precision for all
                        strategies by providing exact anchor points at gaps.

    Returns:
        List of merged polygons, or (polygons, groups) if return_mapping=True
    """
    if not polygons:
        return ([], []) if return_mapping else []

    strategy = coerce_enum(merge_strategy, MergeStrategy)

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
            raise ValueError(f"Unknown merge_strategy: {merge_strategy}")

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
    """
    if not polygons:
        return [], []

    # Build adjacency graph using shared utility
    adjacency = build_adjacency_graph(polygons, margin)

    # Find connected components using shared utility
    groups = find_connected_components(adjacency)

    # Separate isolated polygons from merge groups
    isolated = [g[0] for g in groups if len(g) == 1]
    to_merge = [g for g in groups if len(g) > 1]

    return isolated, to_merge


__all__ = ['merge_close_polygons', 'find_close_polygon_groups']
