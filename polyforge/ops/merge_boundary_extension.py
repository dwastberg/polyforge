"""Boundary extension merge strategy - extend parallel edges."""

from typing import List, Optional, Tuple, Union
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union

from polyforge.core.geometry_utils import remove_holes
from polyforge.ops.merge_edge_detection import find_parallel_close_edges
from polyforge.ops.merge_selective_buffer import merge_selective_buffer


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

    parallel_edges = _collect_parallel_pairs(group_polygons, margin)
    if not parallel_edges:
        return merge_selective_buffer(group_polygons, margin, preserve_holes)

    bridges = _build_bridges(parallel_edges)
    if not bridges:
        return merge_selective_buffer(group_polygons, margin, preserve_holes)

    return _merge_with_bridges(group_polygons, bridges, preserve_holes)


def _collect_parallel_pairs(
    polygons: List[Polygon],
    margin: float,
):
    """Return parallel edge pairs within the requested margin."""
    try:
        return find_parallel_close_edges(polygons, margin)
    except Exception:
        return []


def _build_bridges(
    parallel_edges,
    min_overlap: float = 1e-6,
) -> List[Polygon]:
    """Convert parallel edge pairs into bridge polygons."""
    bridges: List[Polygon] = []
    for edge1, edge2, distance in parallel_edges:
        bridge = _bridge_from_edge_pair(edge1, edge2, min_overlap)
        if bridge is None:
            continue
        if bridge.is_valid and bridge.area > min_overlap:
            bridges.append(bridge)
    return bridges


def _bridge_from_edge_pair(edge1: LineString, edge2: LineString, min_overlap: float) -> Optional[Polygon]:
    """Create a bridge polygon for a pair of edges if overlap exists."""
    coords1 = np.array(edge1.coords)
    coords2 = np.array(edge2.coords)

    if abs(coords1[1][0] - coords1[0][0]) < min_overlap:
        return _bridge_for_vertical_edges(coords1, coords2, min_overlap)
    if abs(coords1[1][1] - coords1[0][1]) < min_overlap:
        return _bridge_for_horizontal_edges(coords1, coords2, min_overlap)
    return _bridge_for_angled_edges(coords1, coords2)


def _bridge_for_vertical_edges(coords1, coords2, min_overlap):
    range1 = sorted([coords1[0][1], coords1[1][1]])
    range2 = sorted([coords2[0][1], coords2[1][1]])
    overlap = _interval_overlap(range1, range2)
    if overlap is None or overlap[1] - overlap[0] < min_overlap:
        return None
    x1 = coords1[0][0]
    x2 = coords2[0][0]
    bridge_coords = [
        (x1, overlap[0]),
        (x2, overlap[0]),
        (x2, overlap[1]),
        (x1, overlap[1]),
    ]
    return Polygon(bridge_coords)


def _bridge_for_horizontal_edges(coords1, coords2, min_overlap):
    range1 = sorted([coords1[0][0], coords1[1][0]])
    range2 = sorted([coords2[0][0], coords2[1][0]])
    overlap = _interval_overlap(range1, range2)
    if overlap is None or overlap[1] - overlap[0] < min_overlap:
        return None
    y1 = coords1[0][1]
    y2 = coords2[0][1]
    bridge_coords = [
        (overlap[0], y1),
        (overlap[0], y2),
        (overlap[1], y2),
        (overlap[1], y1),
    ]
    return Polygon(bridge_coords)


def _bridge_for_angled_edges(coords1, coords2):
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

    return Polygon(bridge_coords)


def _interval_overlap(range1, range2) -> Optional[Tuple[float, float]]:
    """Return the overlap between two ranges, or None if disjoint."""
    overlap_start = max(range1[0], range2[0])
    overlap_end = min(range1[1], range2[1])
    if overlap_end <= overlap_start:
        return None
    return overlap_start, overlap_end


def _merge_with_bridges(
    polygons: List[Polygon],
    bridges: List[Polygon],
    preserve_holes: bool,
) -> Union[Polygon, MultiPolygon]:
    """Union polygons with bridges and post-process holes."""
    all_geoms = list(polygons) + bridges
    merged = unary_union(all_geoms)
    return remove_holes(merged, preserve_holes)


__all__ = ['merge_boundary_extension']
