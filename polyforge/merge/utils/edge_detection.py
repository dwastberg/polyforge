"""Edge detection utilities for finding parallel and close edges."""

from typing import List, Tuple
from collections import defaultdict
import numpy as np
from shapely.geometry import Polygon, LineString


def find_parallel_close_edges(
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
        parallel_pairs = filter_redundant_parallel_pairs(parallel_pairs)

    return parallel_pairs


def filter_redundant_parallel_pairs(
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


__all__ = ['find_parallel_close_edges', 'filter_redundant_parallel_pairs']
