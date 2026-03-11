from __future__ import annotations

from shapely.geometry import Polygon, MultiPolygon
from shapely.strtree import STRtree
from shapely.ops import unary_union
from shapely.errors import GEOSException, TopologicalError

from ..core.errors import ConfigurationError
from ..core.types import MergeStrategy, coerce_enum
from ..core.spatial_utils import build_adjacency_graph, find_connected_components
from ..core.geometry_utils import remove_holes, to_single_polygon
from polyforge.ops.merge import (
    merge_simple_buffer,
    merge_selective_buffer,
    merge_vertex_movement,
    merge_boundary_extension,
    merge_convex_bridges,
    insert_connection_vertices,
)


def merge_close_polygons(
    polygons: list[Polygon],
    margin: float = 0.0,
    merge_strategy: MergeStrategy | str = MergeStrategy.SELECTIVE_BUFFER,
    preserve_holes: bool = True,
    return_mapping: bool = False,
    insert_vertices: bool = False,
    buffer_cleaning: bool = False,
) -> list[Polygon] | tuple[list[Polygon], list[list[int]]]:
    """Merge polygons that overlap or are within margin distance.

    Args:
        polygons: List of input polygons.
        margin: Maximum distance for merging (0.0 = only overlapping polygons).
        merge_strategy: Merging strategy enum or string (default: SELECTIVE_BUFFER).
        preserve_holes: Whether to preserve interior holes when merging.
        return_mapping: If True, return (merged_polygons, groups) where groups[i]
            contains indices of original polygons that were merged.
        insert_vertices: If True, insert vertices at connection points before merging.
        buffer_cleaning: If True, apply a small buffering to clean up geometry after merging.

    Returns:
        List of merged polygons, or (polygons, groups) if return_mapping=True.
    """
    if not polygons:
        return ([], []) if return_mapping else []

    if margin < 0:
        raise ConfigurationError(f"margin must be >= 0, got {margin}")

    # Fast path for margin=0: just merge overlapping/touching polygons
    # Note: merge_strategy is ignored when margin=0 (unary_union handles it directly).
    if margin == 0:
        merged = unary_union(polygons)
        if not preserve_holes:
            merged = remove_holes(merged, preserve_holes=False)

        # Convert result to list format
        if isinstance(merged, Polygon):
            result = [merged]
        elif isinstance(merged, MultiPolygon):
            result = list(merged.geoms)
        else:
            result = []

        if return_mapping:
            mapping = _map_components_to_inputs(result, polygons)
            return result, mapping
        return result

    strategy = coerce_enum(merge_strategy, MergeStrategy)
    isolated_indices, merge_groups = _find_close_polygon_groups(polygons, margin)

    result: list[Polygon] = []
    mapping: list[list[int]] = []

    for idx in isolated_indices:
        result.append(polygons[idx])
        if return_mapping:
            mapping.append([idx])

    for group_indices in merge_groups:
        merged_group = _merge_group_polygons(
            [polygons[i] for i in group_indices],
            group_indices,
            strategy,
            margin,
            preserve_holes,
            insert_vertices,
        )
        if isinstance(merged_group, MultiPolygon):
            for poly in merged_group.geoms:
                result.append(poly)
                if return_mapping:
                    mapping.append(group_indices)
        elif isinstance(merged_group, Polygon):
            result.append(merged_group)
            if return_mapping:
                mapping.append(group_indices)
    if buffer_cleaning:
        for idx, poly in enumerate(result):
            if poly.is_empty:
                continue
            # Apply a tiny buffer to clean up geometry (fixes minor artifacts and internaL edges)
            buffer_ammout = margin / 10
            cleaned = poly.buffer(buffer_ammout, cap_style= 'flat', join_style= 'mitre' ).buffer(-buffer_ammout, cap_style= 'flat', join_style= 'mitre')
            if cleaned.is_empty:
                continue
            if isinstance(cleaned, Polygon):
                result[idx] = cleaned
            elif isinstance(cleaned, MultiPolygon) and len(cleaned.geoms) == 1:
                result[idx] = to_single_polygon(cleaned)
    return (result, mapping) if return_mapping else result


def _find_close_polygon_groups(
    polygons: list[Polygon], margin: float
) -> tuple[list[int], list[list[int]]]:
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
    group_polygons: list[Polygon],
    group_indices: list[int],
    strategy: MergeStrategy,
    margin: float,
    preserve_holes: bool,
    insert_vertices: bool,
):
    """Merge a group of polygons according to the selected strategy."""
    # Early exit for single polygon
    if len(group_polygons) == 1:
        return group_polygons[0]

    # OPTIMIZATION: Try unary_union first to merge overlapping/touching polygons
    # This is a fast operation and handles the common case where polygons already touch
    base_union = unary_union(group_polygons)

    # If union produces a single polygon, we're done
    if isinstance(base_union, Polygon):
        return remove_holes(base_union, preserve_holes)
    if isinstance(base_union, MultiPolygon) and len(base_union.geoms) == 1:
        return remove_holes(base_union.geoms[0], preserve_holes)

    # Still multiple polygons after union -> apply strategy to bridge gaps
    # Update group_polygons to the result of unary_union (removes overlaps)
    if isinstance(base_union, MultiPolygon):
        group_polygons = list(base_union.geoms)
    else:
        # base_union is a single polygon but we already handled that case above
        # This shouldn't happen, but handle it safely
        group_polygons = [base_union]

    # Pre-processing: absorb polygons sitting inside another polygon's hole
    group_polygons = _absorb_hole_polygons(group_polygons, margin)
    if len(group_polygons) == 1:
        return remove_holes(group_polygons[0], preserve_holes)

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
        raise ConfigurationError(f"Unknown merge_strategy: {strategy}")

    merge_func = strategy_map[strategy]
    return merge_func(group_polygons, margin, preserve_holes)


def _absorb_hole_polygons(
    polygons: list[Polygon], margin: float
) -> list[Polygon]:
    """Absorb polygons that sit inside another polygon's hole.

    When polygon B is inside a hole of polygon A and within margin distance
    of the hole boundary, remove the hole from A and union A with B.
    """
    if len(polygons) < 2:
        return polygons

    changed = True
    while changed:
        changed = False
        tree = STRtree(polygons)
        absorbed: set[int] = set()
        processed: set[int] = set()
        new_polygons: list[Polygon] = []

        for i, outer in enumerate(polygons):
            if i in absorbed:
                continue
            processed.add(i)
            if not outer.interiors:
                new_polygons.append(outer)
                continue

            # Check which other polygons sit inside a hole of outer
            # A polygon is in a hole if it's inside the exterior ring but not
            # inside the polygon itself (meaning it falls in a hole).
            outer_no_holes = Polygon(outer.exterior)
            candidates = tree.query(outer_no_holes)
            merged_outer = outer
            for j in candidates:
                if j == i or j in absorbed:
                    continue
                inner = polygons[j]
                # inner must be inside exterior but not inside the polygon
                # (i.e. it sits in a hole)
                if not outer_no_holes.contains(inner.representative_point()):
                    continue
                if merged_outer.contains(inner.representative_point()):
                    continue

                # Find which hole contains the inner polygon
                hole_idx = None
                for hi, hole in enumerate(merged_outer.interiors):
                    hole_poly = Polygon(hole)
                    if hole_poly.contains(inner.representative_point()):
                        hole_idx = hi
                        break
                if hole_idx is None:
                    continue

                # Check distance to hole boundary
                hole_boundary = Polygon(merged_outer.interiors[hole_idx]).boundary
                if inner.distance(hole_boundary) > margin:
                    continue

                # Absorb: merge inner into outer through the hole.
                # 1. Remove the hole from outer (fills it completely)
                # 2. Subtract remaining hole area (hole - inner) to preserve
                #    the part of the hole not covered by inner.
                hole_poly = Polygon(merged_outer.interiors[hole_idx])

                # Build outer without this hole
                remaining_holes = [
                    h
                    for hi, h in enumerate(merged_outer.interiors)
                    if hi != hole_idx
                ]
                outer_filled = Polygon(merged_outer.exterior, remaining_holes)

                # Remaining hole area = hole minus inner (with small buffer
                # to bridge the gap and ensure a clean connection)
                gap = inner.distance(hole_boundary)
                if gap > 0:
                    bridged = inner.buffer(gap + 1e-6, join_style='mitre')
                else:
                    bridged = inner.buffer(1e-6, join_style='mitre')
                remaining_hole = hole_poly.difference(bridged)

                # Subtract remaining hole area from the filled outer
                if not remaining_hole.is_empty and remaining_hole.area > 0:
                    merged_outer = outer_filled.difference(remaining_hole)
                else:
                    merged_outer = outer_filled

                if isinstance(merged_outer, MultiPolygon):
                    merged_outer = max(merged_outer.geoms, key=lambda g: g.area)
                absorbed.add(j)
                changed = True

            new_polygons.append(merged_outer)

        # Add any polygons that weren't processed or absorbed
        for j in range(len(polygons)):
            if j not in processed and j not in absorbed:
                new_polygons.append(polygons[j])

        polygons = new_polygons

    return polygons


def _map_components_to_inputs(
    components: list[Polygon],
    inputs: list[Polygon],
    area_eps: float = 1e-12,
) -> list[list[int]]:
    """Build component-to-input index mapping for unary_union results."""
    mapping: list[list[int]] = []

    for comp in components:
        contributors: list[int] = []
        if comp.is_empty:
            mapping.append(contributors)
            continue

        for idx, poly in enumerate(inputs):
            try:
                if poly.is_empty:
                    continue
                if not comp.intersects(poly):
                    continue
                # Prefer positive-area intersection; fall back to touches/containment.
                intersection = comp.intersection(poly)
                if getattr(intersection, "area", 0.0) > area_eps or comp.touches(poly):
                    contributors.append(idx)
            except (GEOSException, TopologicalError, ValueError):
                continue

        mapping.append(contributors)

    return mapping


__all__ = ["merge_close_polygons"]
