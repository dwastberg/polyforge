"""Merge-related geometry helpers shared across strategies."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

from ..core.spatial_utils import (
    SegmentIndex,
    build_segment_index,
    find_polygon_pairs,
    query_close_segments,
)


def find_close_boundary_pairs(
    polygons: List[Polygon],
    margin: float,
    segment_length: Optional[float] = None,
) -> List[Tuple[LineString, LineString, float]]:
    """Return pairs of boundary segments that sit within ``margin`` of each other."""
    if segment_length is None:
        segment_length = margin * 2.0 if margin > 0 else 1.0

    index = build_segment_index(polygons, segment_length)
    close_pairs: List[Tuple[LineString, LineString, float]] = []
    seen: set[Tuple[int, int]] = set()

    for seg_idx, segment in enumerate(index.segments):
        owner_i = index.owners[seg_idx][0]
        for cand_idx in query_close_segments(index, seg_idx, margin):
            if cand_idx <= seg_idx:
                continue
            owner_j = index.owners[cand_idx][0]
            if owner_i == owner_j:
                continue
            pair_key = (seg_idx, cand_idx)
            if pair_key in seen:
                continue
            distance = segment.distance(index.segments[cand_idx])
            if distance <= margin:
                close_pairs.append((segment, index.segments[cand_idx], distance))
                seen.add(pair_key)

    return close_pairs


def get_boundary_points_near(
    polygon: Polygon,
    point: Point,
    radius: float,
) -> List[Tuple[float, float]]:
    """Extract boundary points within ``radius`` of ``point``."""
    coords = list(polygon.exterior.coords)
    close_points = []

    for coord in coords[:-1]:
        coord_point = Point(coord)
        if coord_point.distance(point) <= radius:
            close_points.append(coord)

    if len(close_points) < 3:
        boundary = polygon.exterior
        num_samples = max(10, int(boundary.length / 2))

        for i in range(num_samples):
            t = i / num_samples
            sampled_point = boundary.interpolate(t, normalized=True)
            if sampled_point.distance(point) <= radius:
                close_points.append((sampled_point.x, sampled_point.y))

    return close_points


def insert_connection_vertices(
    polygons: List[Polygon],
    margin: float,
    tolerance: float = 0.01,
) -> List[Polygon]:
    """Insert vertices at optimal connection points between close polygons."""
    if len(polygons) < 2:
        return polygons

    modified_coords = {}
    candidate_pairs = find_polygon_pairs(
        polygons,
        margin=margin,
        predicate="intersects",
        validate_func=None,
    )

    for i, j in candidate_pairs:
        poly_i = polygons[i]
        poly_j = polygons[j]

        if poly_i.distance(poly_j) > margin:
            continue

        if i not in modified_coords:
            modified_coords[i] = list(poly_i.exterior.coords)
        if j not in modified_coords:
            modified_coords[j] = list(poly_j.exterior.coords)

        pt_i, pt_j = nearest_points(poly_i.boundary, poly_j.boundary)
        _plan_insertion(i, pt_i, modified_coords, tolerance)
        _plan_insertion(j, pt_j, modified_coords, tolerance)

    return _rebuild_from_coords(polygons, modified_coords)


def _plan_insertion(
    poly_idx: int,
    point,
    modified_coords,
    tolerance: float,
) -> None:
    coords = modified_coords[poly_idx]
    pt_coords = point.coords[0]

    for coord in coords[:-1]:
        if np.hypot(coord[0] - pt_coords[0], coord[1] - pt_coords[1]) < tolerance:
            return

    for edge_idx in range(len(coords) - 1):
        seg_start = coords[edge_idx]
        seg_end = coords[edge_idx + 1]
        seg = np.array(seg_end[:2]) - np.array(seg_start[:2])
        if np.linalg.norm(seg) < 1e-12:
            continue
        projection = LineString([seg_start, seg_end]).distance(point)
        if projection > 1e-6:
            continue

        new_vertex = _interpolate_vertex(coords[edge_idx], coords[edge_idx + 1], point)
        coords.insert(edge_idx + 1, new_vertex)
        modified_coords[poly_idx] = coords
        return


def _interpolate_vertex(start, end, point) -> tuple:
    if len(start) == 3:
        seg_2d = LineString([(start[0], start[1]), (end[0], end[1])])
        dist_along = seg_2d.project(point)
        total = seg_2d.length
        if total > 1e-10:
            t = dist_along / total
            z = start[2] + t * (end[2] - start[2])
            return (point.x, point.y, z)
        return start
    return (point.x, point.y)


def _rebuild_from_coords(polygons: List[Polygon], modified_coords: dict) -> List[Polygon]:
    result = []
    for idx, poly in enumerate(polygons):
        if idx not in modified_coords:
            result.append(poly)
            continue
        coords = modified_coords[idx]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        holes = [list(hole.coords) for hole in poly.interiors] if poly.interiors else None
        result.append(Polygon(coords, holes=holes))
    return result


__all__ = [
    "find_close_boundary_pairs",
    "get_boundary_points_near",
    "insert_connection_vertices",
]
