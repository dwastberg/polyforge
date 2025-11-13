"""Core merge orchestration logic."""

from typing import List, Tuple, Union
from shapely.geometry import Polygon, MultiPolygon
from shapely.strtree import STRtree

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
    insert_vertices: bool = False,
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
    isolated_indices, merge_groups = find_close_polygon_groups(polygons, margin)

    result: List[Polygon] = []
    mapping: List[List[int]] = []

    _append_isolated_polygons(result, mapping, polygons, isolated_indices, return_mapping)

    for group_indices in merge_groups:
        merged_group = _merge_group_polygons(
            [polygons[i] for i in group_indices],
            group_indices,
            strategy,
            margin,
            preserve_holes,
            insert_vertices,
        )
        _append_merge_result(result, mapping, merged_group, group_indices, return_mapping)

    return (result, mapping) if return_mapping else result


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

    tree = STRtree(polygons)
    adjacency = build_adjacency_graph(polygons, margin, tree=tree)

    # Find connected components using shared utility
    groups = find_connected_components(adjacency)

    # Separate isolated polygons from merge groups
    isolated = [g[0] for g in groups if len(g) == 1]
    to_merge = [g for g in groups if len(g) > 1]

    return isolated, to_merge


def _merge_group_polygons(
    group_polygons: List[Polygon],
    group_indices: List[int],
    strategy: MergeStrategy,
    margin: float,
    preserve_holes: bool,
    insert_vertices: bool,
):
    """Merge a group of polygons according to the selected strategy."""
    if insert_vertices:
        group_polygons = insert_connection_vertices(group_polygons, margin)

    strategy_map = {
        MergeStrategy.SIMPLE_BUFFER: merge_simple_buffer,
        MergeStrategy.SELECTIVE_BUFFER: merge_selective_buffer,
        MergeStrategy.VERTEX_MOVEMENT: merge_vertex_movement,
        MergeStrategy.BOUNDARY_EXTENSION: merge_boundary_extension,
        MergeStrategy.CONVEX_BRIDGES: merge_convex_bridges,
    }

    if strategy not in strategy_map:
        raise ValueError(f"Unknown merge_strategy: {strategy}")

    merge_func = strategy_map[strategy]
    return merge_func(group_polygons, margin, preserve_holes)


def _append_isolated_polygons(
    result: List[Polygon],
    mapping: List[List[int]],
    polygons: List[Polygon],
    isolated_indices: List[int],
    return_mapping: bool,
) -> None:
    """Append isolated polygons directly to the result."""
    for idx in isolated_indices:
        result.append(polygons[idx])
        if return_mapping:
            mapping.append([idx])


def _append_merge_result(
    result: List[Polygon],
    mapping: List[List[int]],
    merged,
    group_indices: List[int],
    return_mapping: bool,
) -> None:
    """Append merged geometry to outputs, handling MultiPolygon cases."""
    if isinstance(merged, MultiPolygon):
        for poly in merged.geoms:
            result.append(poly)
            if return_mapping:
                mapping.append(group_indices)
    elif isinstance(merged, Polygon):
        result.append(merged)
        if return_mapping:
            mapping.append(group_indices)


__all__ = ['merge_close_polygons', 'find_close_polygon_groups']
