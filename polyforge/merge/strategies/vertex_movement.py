"""Vertex movement merge strategy - move vertices toward each other."""

from typing import List, Union
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union, nearest_points


def merge_vertex_movement(
    group_polygons: List[Polygon],
    margin: float,
    preserve_holes: bool
) -> Union[Polygon, MultiPolygon]:
    """Merge by moving vertices of close polygons toward each other.

    Precise control, preserves overall structure.

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

    modified_polygons = []

    for poly_idx, poly in enumerate(group_polygons):
        coords = np.array(poly.exterior.coords)
        new_coords = coords.copy()

        # For each vertex, check if it's close to another polygon
        for i in range(len(coords) - 1):  # Exclude closing vertex
            vertex = Point(coords[i])

            # Find closest point on other polygons
            min_dist = float('inf')
            closest_point = None

            for other_idx, other_poly in enumerate(group_polygons):
                if other_idx == poly_idx:
                    continue

                # Use Shapely's distance for efficiency
                dist = vertex.distance(other_poly)

                if dist < min_dist and dist <= margin:
                    min_dist = dist
                    # Get actual closest point on boundary
                    _, closest_pt = nearest_points(vertex, other_poly.boundary)
                    closest_point = closest_pt

            # Move vertex toward closest point
            if closest_point is not None and min_dist <= margin:
                move_vector = np.array(closest_point.coords[0]) - coords[i][:2]
                # Move vertex all the way to create overlap
                new_coords[i][:2] = coords[i][:2] + move_vector

        # Create modified polygon, preserving holes if needed
        if preserve_holes and poly.interiors:
            holes = [np.array(hole.coords) for hole in poly.interiors]
            modified_polygons.append(Polygon(new_coords, holes=holes))
        else:
            modified_polygons.append(Polygon(new_coords))

    # Union the modified polygons
    result = unary_union(modified_polygons)

    # Validate and fix if needed
    if not result.is_valid:
        result = result.buffer(0)

    return result


__all__ = ['merge_vertex_movement']
