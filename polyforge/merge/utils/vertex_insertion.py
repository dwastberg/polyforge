"""Vertex insertion utilities for optimal merge connection points."""

from typing import List

import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import nearest_points

from ...core.spatial_utils import find_polygon_pairs


def insert_connection_vertices(
    polygons: List[Polygon],
    margin: float,
    tolerance: float = 0.01
) -> List[Polygon]:
    """Insert vertices at optimal connection points between close polygons.

    For each pair of polygons within margin distance, finds the closest points
    on their boundaries. If a closest point lies on an edge (not at an existing
    vertex), inserts a new vertex at that location. This gives subsequent merge
    strategies optimal anchor points for creating minimal bridges.

    Args:
        polygons: List of polygons to process
        margin: Maximum distance for considering polygons close
        tolerance: Minimum distance from existing vertex to insert new one (default: 0.01)

    Returns:
        List of polygons with new vertices inserted at connection points

    Notes:
        - Only inserts vertices when closest point is > tolerance from existing vertices
        - Inserts at closest point per edge pair (one per close edge)
        - Preserves holes and Z-coordinates if present
    """
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
        line = np.array(pt_coords[:2])
        line_segment = np.array(seg_start[:2])
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


__all__ = ['insert_connection_vertices']
