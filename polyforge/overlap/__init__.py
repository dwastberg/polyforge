"""Batch overlap resolution utilities."""

from __future__ import annotations

from typing import List

from shapely.geometry import Polygon
from shapely.strtree import STRtree

from ..core.types import OverlapStrategy
from .engine import resolve_overlap_pair


def remove_overlaps(
    polygons: List[Polygon],
    overlap_strategy: OverlapStrategy = OverlapStrategy.SPLIT,
    max_iterations: int = 100,
) -> List[Polygon]:
    """Remove overlaps from a list of polygons using the shared overlap engine."""
    if not polygons:
        return []

    result = list(polygons)
    changed = True
    iteration = 0

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        tree = STRtree(result)
        overlapping_pairs = []
        seen = set()

        for i, poly_i in enumerate(result):
            for j in tree.query(poly_i, predicate="intersects"):
                if j <= i or (i, j) in seen:
                    continue
                seen.add((i, j))
                poly_j = result[j]

                if poly_i.touches(poly_j):
                    continue

                overlap = poly_i.intersection(poly_j)
                if getattr(overlap, "area", 0.0) > 1e-10:
                    overlapping_pairs.append((i, j))

        if not overlapping_pairs:
            break

        processed = set()
        independent_pairs = []

        for i, j in overlapping_pairs:
            if i in processed or j in processed:
                continue
            processed.add(i)
            processed.add(j)
            independent_pairs.append((i, j))

        for i, j in independent_pairs:
            result[i], result[j] = resolve_overlap_pair(
                result[i],
                result[j],
                strategy=overlap_strategy,
            )
            changed = True

    return result


def count_overlaps(polygons: List[Polygon], min_area_threshold: float = 1e-10) -> int:
    """Count the number of overlapping polygon pairs."""
    if not polygons:
        return 0

    tree = STRtree(polygons)
    count = 0
    seen = set()

    for i, poly_i in enumerate(polygons):
        for j in tree.query(poly_i, predicate="intersects"):
            if j <= i or (i, j) in seen:
                continue
            seen.add((i, j))
            overlap = poly_i.intersection(polygons[j])
            if getattr(overlap, "area", 0.0) > min_area_threshold:
                count += 1

    return count


def find_overlapping_groups(
    polygons: List[Polygon],
    min_area_threshold: float = 1e-10,
) -> List[List[int]]:
    """Return components of polygons where overlaps are present."""
    if not polygons:
        return []

    tree = STRtree(polygons)
    adjacency = {i: set() for i in range(len(polygons))}

    for i, poly_i in enumerate(polygons):
        for j in tree.query(poly_i, predicate="intersects"):
            if j == i:
                continue
            overlap = poly_i.intersection(polygons[j])
            if getattr(overlap, "area", 0.0) > min_area_threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited = set()
    groups = []

    def dfs(node: int, acc: List[int]):
        visited.add(node)
        acc.append(node)
        for neighbor in adjacency.get(node, ()):
            if neighbor not in visited:
                dfs(neighbor, acc)

    for idx in range(len(polygons)):
        if idx not in visited:
            component: List[int] = []
            dfs(idx, component)
            groups.append(sorted(component))

    return groups


__all__ = [
    "remove_overlaps",
    "count_overlaps",
    "find_overlapping_groups",
    "resolve_overlap_pair",
]
