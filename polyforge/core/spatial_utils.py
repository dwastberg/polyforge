from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
from shapely.ops import nearest_points


def find_polygon_pairs(
    polygons: list[Polygon],
    margin: float = 0.0,
    predicate: str = "intersects",
    validate_func: Callable[[Polygon, Polygon], bool] | None = None,
    tree: STRtree | None = None,
) -> list[tuple[int, int]]:
    """Find pairs of polygons that satisfy spatial predicate within margin."""
    if not polygons:
        return []

    if tree is None:
        tree = STRtree(polygons)

    # Track checked pairs to avoid duplicates
    pairs = []
    checked: set[tuple[int, int]] = set()

    for i in range(len(polygons)):
        poly_i = polygons[i]

        # Query with distance predicate if possible
        if margin > 0 and (predicate is None or predicate == "intersects"):
            candidate_indices = tree.query(poly_i, predicate="dwithin", distance=margin)
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


def find_nearest_boundary_point(point: Point, geometry: BaseGeometry) -> Point:
    """Find the nearest point on a geometry's boundary to a given point."""
    if hasattr(geometry, "boundary"):
        _, nearest_pt = nearest_points(point, geometry.boundary)
    else:
        _, nearest_pt = nearest_points(point, geometry)

    return nearest_pt


def build_adjacency_graph(
    polygons: list[Polygon], margin: float, tree: STRtree | None = None
) -> dict[int, set[int]]:
    if not polygons:
        return {}

    # Build spatial index if not provided
    if tree is None:
        tree = STRtree(polygons)

    # Initialize adjacency dict
    adjacency: dict[int, set[int]] = {i: set() for i in range(len(polygons))}

    for i in range(len(polygons)):
        poly_i = polygons[i]

        # Query spatial index
        if margin > 0:
            candidate_indices = tree.query(poly_i, predicate="dwithin", distance=margin)
        else:
            candidate_indices = tree.query(poly_i, predicate="intersects")

        # dwithin already filters to exact distance <= margin
        for j in candidate_indices:
            if i != j and j not in adjacency[i]:
                adjacency[i].add(j)
                adjacency[j].add(i)

    return adjacency


def find_connected_components(adjacency: dict[int, set[int]]) -> list[list[int]]:
    visited = set()
    components = []

    def dfs(node: int, current_component: list[int]):
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


def iterate_unique_pairs(items: Sequence) -> Iterable[tuple[int, int]]:
    """Yield unique index pairs (i, j) with i < j for the given sequence."""
    length = len(items)
    for i in range(length):
        for j in range(i + 1, length):
            yield i, j


__all__ = [
    "find_polygon_pairs",
    "find_nearest_boundary_point",
    "build_adjacency_graph",
    "find_connected_components",
    "iterate_unique_pairs",
    "SegmentIndex",
    "build_segment_index",
    "query_close_segments",
]


@dataclass
class SegmentIndex:
    """Spatial index of polygon boundary segments."""

    segments: list[LineString]
    owners: list[tuple[int, int]]  # (polygon_index, edge_index)
    tree: STRtree


def build_segment_index(
    polygons: list[Polygon],
    segment_length: float,
) -> SegmentIndex:
    """Discretize polygon boundaries into segments and build an STRtree."""
    segments: list[LineString] = []
    owners: list[tuple[int, int]] = []

    for poly_idx, poly in enumerate(polygons):
        coords = list(poly.exterior.coords)
        for edge_idx in range(len(coords) - 1):
            p0 = coords[edge_idx]
            p1 = coords[edge_idx + 1]
            seg_len = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if seg_len <= segment_length + 1e-9:
                segments.append(LineString([p0, p1]))
                owners.append((poly_idx, edge_idx))
                continue

            splits = max(1, int(np.ceil(seg_len / segment_length)))
            for j in range(splits):
                t0 = j / splits
                t1 = (j + 1) / splits
                sp0 = (p0[0] + t0 * (p1[0] - p0[0]), p0[1] + t0 * (p1[1] - p0[1]))
                sp1 = (p0[0] + t1 * (p1[0] - p0[0]), p0[1] + t1 * (p1[1] - p0[1]))
                segments.append(LineString([sp0, sp1]))
                owners.append((poly_idx, edge_idx))

    tree = STRtree(segments) if segments else STRtree([])
    return SegmentIndex(segments=segments, owners=owners, tree=tree)


def query_close_segments(
    index: SegmentIndex,
    seg_idx: int,
    margin: float,
) -> list[int]:
    """Return indices of segments within ``margin`` distance."""
    segment = index.segments[seg_idx]
    matches = index.tree.query(segment, predicate="dwithin", distance=margin)
    return [j for j in matches if j != seg_idx]
