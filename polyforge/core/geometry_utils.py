from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, Point
from shapely.geometry.base import BaseGeometry
from shapely.errors import GEOSException


def to_single_polygon(geometry: BaseGeometry) -> Polygon:
    """Convert geometry to a single Polygon by taking the largest piece."""
    if isinstance(geometry, Polygon):
        return geometry
    elif isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda p: p.area)
    elif isinstance(geometry, GeometryCollection):
        # Extract polygons from collection
        polygons = [g for g in geometry.geoms if isinstance(g, Polygon)]
        if polygons:
            return max(polygons, key=lambda p: p.area)
        # Try multipolygons
        multipolygons = [g for g in geometry.geoms if isinstance(g, MultiPolygon)]
        if multipolygons:
            all_polys = []
            for mp in multipolygons:
                all_polys.extend(mp.geoms)
            return max(all_polys, key=lambda p: p.area)
    # Fallback to empty polygon
    return Polygon()


def remove_holes(
    geometry: Polygon | MultiPolygon, preserve_holes: bool
) -> Polygon | MultiPolygon:
    if preserve_holes:
        return geometry

    if isinstance(geometry, Polygon):
        if geometry.interiors:
            return Polygon(geometry.exterior)
        return geometry
    elif isinstance(geometry, MultiPolygon):
        return MultiPolygon([Polygon(p.exterior) for p in geometry.geoms])

    return geometry


def safe_buffer_fix(
    geometry: BaseGeometry,
    distance: float = 0.0,
    return_largest: bool = True,
    min_area: float = 0.0,
) -> BaseGeometry | None:
    try:
        buffered = geometry.buffer(distance)
    except GEOSException:
        return None

    if return_largest and isinstance(buffered, MultiPolygon) and buffered.geoms:
        buffered = max(buffered.geoms, key=lambda p: p.area)

    if (
        isinstance(buffered, (Polygon, MultiPolygon))
        and buffered.is_valid
        and not buffered.is_empty
    ):
        if min_area > 0 and hasattr(buffered, "area") and buffered.area < min_area:
            return None
        return buffered
    return None


def update_coord_preserve_z(coords: np.ndarray, index: int, new_xy: np.ndarray) -> None:
    if coords.shape[1] > 2:
        # 3D coordinates - preserve Z
        coords[index] = np.array([new_xy[0], new_xy[1], coords[index][2]])
    else:
        # 2D coordinates
        coords[index] = new_xy


def hole_shape_metrics(hole_polygon: Polygon) -> tuple[float, float]:
    obb = hole_polygon.minimum_rotated_rectangle
    coords = list(obb.exterior.coords)
    if len(coords) < 4:
        raise ValueError("degenerate hole")

    edge1 = Point(coords[0]).distance(Point(coords[1]))
    edge2 = Point(coords[1]).distance(Point(coords[2]))
    longer = max(edge1, edge2)
    shorter = min(edge1, edge2)
    if shorter <= 0:
        raise ValueError("degenerate hole")

    aspect_ratio = longer / shorter
    width = shorter
    return aspect_ratio, width


__all__ = [
    "to_single_polygon",
    "remove_holes",
    "safe_buffer_fix",
    "update_coord_preserve_z",
    "hole_shape_metrics",
]
