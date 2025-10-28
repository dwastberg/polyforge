"""Boundary extension merge strategy - extend parallel edges."""

from typing import List, Union
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union

from ..utils.edge_detection import find_parallel_close_edges
from .selective_buffer import merge_selective_buffer


def merge_boundary_extension(
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
    parallel_edges = find_parallel_close_edges(group_polygons, margin)

    if not parallel_edges:
        # No parallel edges, fall back to selective buffer
        return merge_selective_buffer(group_polygons, margin, preserve_holes)

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


__all__ = ['merge_boundary_extension']
