"""Functions for handling multiple overlapping polygons.

This module provides efficient algorithms for resolving overlaps in large
collections of polygons using spatial indexing.
"""

from typing import List, Literal, Union
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from .split import split_overlap
from .core.types import OverlapStrategy


def remove_overlaps(
    polygons: List[Polygon],
    overlap_strategy: OverlapStrategy = OverlapStrategy.SPLIT,
    max_iterations: int = 100
) -> List[Polygon]:
    """Remove all overlaps from a list of polygons efficiently.

    This function processes a list of potentially overlapping polygons and returns
    a new list where all overlaps have been resolved. It uses spatial indexing
    (STRtree) to efficiently find overlapping pairs, avoiding O(n²) comparisons.

    The algorithm works in iterations:
    1. Build a spatial index of all polygons
    2. Find all overlapping pairs using the index
    3. Select independent pairs (no polygon appears in multiple pairs)
    4. Resolve all independent pairs in parallel using split_overlap
    5. Repeat until no overlaps remain or max_iterations is reached

    This approach minimizes the number of split_overlap calls and handles the case
    where multiple polygons overlap the same polygon by processing them iteratively.

    Args:
        polygons: List of polygons (potentially overlapping)
        overlap_strategy: How to handle overlaps:
            - OverlapStrategy.SPLIT: Split overlap equally (50/50) between polygons
            - OverlapStrategy.LARGEST: Assign overlap to the larger polygon
            - OverlapStrategy.SMALLEST: Assign overlap to the smaller polygon
        max_iterations: Maximum number of iterations to prevent infinite loops
            (default: 100)

    Returns:
        List of polygons with all overlaps removed. The order and number of
        polygons is preserved.

    Examples:
        >>> from shapely.geometry import Polygon
        >>> from polyforge.core.types import OverlapStrategy
        >>> poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        >>> poly2 = Polygon([(1, 0), (3, 0), (3, 2), (1, 2)])
        >>> result = remove_overlaps([poly1, poly2])

        >>> # Using specific strategy
        >>> result = remove_overlaps([poly1, poly2], overlap_strategy=OverlapStrategy.LARGEST)
    """
    if not polygons:
        return []

    # Make a working copy
    result = list(polygons)
    changed = True
    iteration = 0

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1

        # Build spatial index for efficient overlap detection
        tree = STRtree(result)

        # Find all overlapping pairs using spatial index
        overlapping_pairs = []
        checked_pairs = set()

        for i in range(len(result)):
            poly_i = result[i]

            # Query spatial index for potential overlaps
            candidate_indices = tree.query(poly_i, predicate='intersects')

            for j in candidate_indices:
                if j > i:  # Only process each pair once
                    # Skip if already checked
                    if (i, j) in checked_pairs:
                        continue
                    checked_pairs.add((i, j))

                    poly_j = result[j]

                    # Check if they actually overlap (not just touch)
                    if poly_i.intersects(poly_j):
                        overlap = poly_i.intersection(poly_j)
                        if hasattr(overlap, 'area') and overlap.area > 1e-10:
                            overlapping_pairs.append((i, j))

        if not overlapping_pairs:
            # No overlaps found, we're done
            break

        # Find independent pairs (no polygon appears in multiple pairs)
        processed_indices = set()
        pairs_to_resolve = []

        for i, j in overlapping_pairs:
            if i not in processed_indices and j not in processed_indices:
                pairs_to_resolve.append((i, j))
                processed_indices.add(i)
                processed_indices.add(j)

        # Resolve all independent pairs in this iteration
        for i, j in pairs_to_resolve:
            new_i, new_j = split_overlap(
                result[i],
                result[j],
                overlap_strategy=overlap_strategy
            )
            result[i] = new_i
            result[j] = new_j
            changed = True

    return result


def count_overlaps(polygons: List[Polygon], tolerance: float = 1e-10) -> int:
    """Count the number of overlapping pairs in a list of polygons.

    Uses spatial indexing for efficient counting.

    Args:
        polygons: List of polygons to check
        tolerance: Minimum overlap area to count (default: 1e-10)

    Returns:
        Number of overlapping pairs
    """
    if not polygons:
        return 0

    # Build spatial index
    tree = STRtree(polygons)

    overlap_count = 0
    checked_pairs = set()

    for i, poly_i in enumerate(polygons):
        candidate_indices = tree.query(poly_i, predicate='intersects')

        for j in candidate_indices:
            if j > i:  # Only count each pair once
                if (i, j) in checked_pairs:
                    continue
                checked_pairs.add((i, j))

                poly_j = polygons[j]
                if poly_i.intersects(poly_j):
                    overlap = poly_i.intersection(poly_j)
                    if hasattr(overlap, 'area') and overlap.area > tolerance:
                        overlap_count += 1

    return overlap_count


def find_overlapping_groups(polygons: List[Polygon], tolerance: float = 1e-10) -> List[List[int]]:
    """Find groups of mutually overlapping polygons.

    Returns groups where all polygons in a group have at least one overlap
    with another polygon in the same group (connected components).

    Args:
        polygons: List of polygons to analyze
        tolerance: Minimum overlap area to consider (default: 1e-10)

    Returns:
        List of groups, where each group is a list of polygon indices
    """
    if not polygons:
        return []

    # Build spatial index
    tree = STRtree(polygons)

    # Build adjacency graph
    adjacency = {i: set() for i in range(len(polygons))}

    for i, poly_i in enumerate(polygons):
        candidate_indices = tree.query(poly_i, predicate='intersects')

        for j in candidate_indices:
            if j != i:
                poly_j = polygons[j]
                if poly_i.intersects(poly_j):
                    overlap = poly_i.intersection(poly_j)
                    if hasattr(overlap, 'area') and overlap.area > tolerance:
                        adjacency[i].add(j)
                        adjacency[j].add(i)

    # Find connected components using DFS
    visited = set()
    groups = []

    def dfs(node, current_group):
        visited.add(node)
        current_group.append(node)
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                dfs(neighbor, current_group)

    for i in range(len(polygons)):
        if i not in visited:
            group = []
            dfs(i, group)
            groups.append(sorted(group))

    return groups


__all__ = [
    'remove_overlaps',
    'count_overlaps',
    'find_overlapping_groups',
]
