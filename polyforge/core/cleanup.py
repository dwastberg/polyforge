"""Shared geometry cleanup utilities used across simplify and repair modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .geometry_utils import to_single_polygon


@dataclass
class CleanupConfig:
    """Describes which cleanup operations should run."""

    min_zero_area: float = 1e-10
    hole_area_threshold: Optional[float] = None
    hole_aspect_ratio: Optional[float] = None
    hole_min_width: Optional[float] = None
    preserve_holes: bool = True


def remove_small_holes(
    geometry: Union[Polygon, MultiPolygon],
    min_area: float,
) -> Union[Polygon, MultiPolygon]:
    """Remove holes whose area is below the threshold."""
    if isinstance(geometry, Polygon):
        return _strip_holes_from_polygon(geometry, min_area)
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon(
            [_strip_holes_from_polygon(poly, min_area) for poly in geometry.geoms]
        )
    return geometry


def remove_narrow_holes(
    geometry: Union[Polygon, MultiPolygon],
    max_aspect_ratio: float = 50.0,
    min_width: Optional[float] = None,
) -> Union[Polygon, MultiPolygon]:
    """Remove holes that exceed aspect-ratio or width constraints."""
    if isinstance(geometry, Polygon):
        return _filter_holes_by_shape(geometry, max_aspect_ratio, min_width)
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon(
            [
                _filter_holes_by_shape(poly, max_aspect_ratio, min_width)
                for poly in geometry.geoms
            ]
        )
    return geometry


def cleanup_polygon(
    geometry: Union[Polygon, MultiPolygon],
    config: CleanupConfig,
) -> Union[Polygon, MultiPolygon]:
    """Apply the requested cleanup operations to the geometry."""
    result: Union[Polygon, MultiPolygon] = geometry
    result = remove_small_holes(result, min_area=config.min_zero_area)

    if config.hole_area_threshold and config.hole_area_threshold > 0:
        result = remove_small_holes(result, min_area=config.hole_area_threshold)

    if config.hole_aspect_ratio or (
        config.hole_min_width is not None and config.hole_min_width > 0
    ):
        result = remove_narrow_holes(
            result,
            max_aspect_ratio=config.hole_aspect_ratio or 50.0,
            min_width=config.hole_min_width,
        )

    if not config.preserve_holes:
        result = _strip_all_holes(result)

    return result


def _strip_holes_from_polygon(polygon: Polygon, min_area: float) -> Polygon:
    if min_area <= 0:
        return polygon
    valid_holes = [
        hole
        for hole in polygon.interiors
        if Polygon(hole).area >= min_area
    ]
    return Polygon(polygon.exterior, holes=[list(h.coords) for h in valid_holes])


def _filter_holes_by_shape(
    polygon: Polygon,
    max_aspect_ratio: float,
    min_width: Optional[float],
) -> Polygon:
    holes = []
    for interior in polygon.interiors:
        hole_poly = Polygon(interior)
        try:
            aspect, width = _hole_shape_metrics(hole_poly)
        except Exception:
            continue
        if aspect > max_aspect_ratio:
            continue
        if min_width is not None and width < min_width:
            continue
        holes.append(list(interior.coords))
    return Polygon(polygon.exterior, holes=holes)


def _hole_shape_metrics(hole: Polygon) -> Tuple[float, float]:
    rectangle = hole.minimum_rotated_rectangle
    coords = list(rectangle.exterior.coords)
    if len(coords) < 4:
        raise ValueError("degenerate hole")
    from shapely.geometry import Point

    edge1 = Point(coords[0]).distance(Point(coords[1]))
    edge2 = Point(coords[1]).distance(Point(coords[2]))
    longer = max(edge1, edge2)
    shorter = min(edge1, edge2)
    if shorter <= 0:
        raise ValueError("degenerate hole")
    return longer / shorter, shorter


def _strip_all_holes(
    geometry: Union[Polygon, MultiPolygon]
) -> Union[Polygon, MultiPolygon]:
    if isinstance(geometry, Polygon):
        return Polygon(geometry.exterior)
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon([Polygon(poly.exterior) for poly in geometry.geoms])
    return geometry


__all__ = [
    "CleanupConfig",
    "cleanup_polygon",
    "remove_small_holes",
    "remove_narrow_holes",
]
