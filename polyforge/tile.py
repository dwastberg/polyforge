import shapely
from shapely.geometry import Polygon, MultiPolygon, Box
from typing import Tuple, Optional, Union
from math import atan2


def tile_polygon(polygon: Polygon, tile_count: Optional[Union[Tuple[int, int], int]] = None,
                 tile_size: Optional[Tuple[float, float], float] = None, axis_oriented=False) -> Polygon:
    if axis_oriented:
        tiling_bbox = Box(*polygon.bounds)
        angle = 0.0
    else:
        tiling_bbox = shapely.oriented_envelope(polygon)
        angle = atan2(
            tiling_bbox.exterior.coords[1][1] - tiling_bbox.exterior.coords[0][1],
            tiling_bbox.exterior.coords[1][0] - tiling_bbox.exterior.coords[0][0]
        )
        centroid = tiling_bbox.centroid
        tiling_bbox = shapely.rotate(tiling_bbox, -angle, origin=centroid, use_radians=True)
    tiles = _tile_box(tiling_bbox, tile_count=tile_count, tile_size=tile_size)
    tiles=MultiPolygon(tiles)
    tiled_polygon = shapely.intersection(polygon, tiles)
    if not axis_oriented:
        tiled_polygon = shapely.rotate(tiled_polygon, angle, origin=centroid, use_radians=True)
    return tiled_polygon


def _tile_box(box: Polygon, tile_count: Optional[Union[Tuple[int, int], int]] = None,
              tile_size: Optional[Tuple[float, float], float] = None) -> List[Polygon]:
    minx, miny, maxx, maxy = box.bounds
    width = maxx - minx
    height = maxy - miny

    if tile_count is not None:
        if isinstance(tile_count, int):
            cols = rows = tile_count
        else:
            cols, rows = tile_count
        tile_width = width / cols
        tile_height = height / rows
    elif tile_size is not None:
        if isinstance(tile_size, (int, float)):
            tile_width = tile_height = tile_size
        else:
            tile_width, tile_height = tile_size
        cols = int(width // tile_width) + 1
        rows = int(height // tile_height) + 1
    else:
        raise ValueError("Either tile_count or tile_size must be provided.")

    tiles = []
    for i in range(cols):
        for j in range(rows):
            tile_minx = minx + i * tile_width
            tile_miny = miny + j * tile_height
            tile_maxx = min(tile_minx + tile_width, maxx)
            tile_maxy = min(tile_miny + tile_height, maxy)
            tile = Box(tile_minx, tile_miny, tile_maxx, tile_maxy)
            tiles.append(tile)

    return tiles


__all__ = [
    'tile_polygon',
]
