"""Simple buffer merge strategy - classic expand-contract method."""

from typing import List, Union
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from polyforge.simplify import simplify_vwp
from polyforge.core.geometry_utils import remove_holes


def merge_simple_buffer(
    group_polygons: List[Polygon],
    margin: float,
    preserve_holes: bool,
    simplify: bool = True
) -> Union[Polygon, MultiPolygon]:
    """Merge using classic expand-contract buffer method.

    Fast and simple, but changes polygon shape (rounds corners).

    Args:
        group_polygons: Polygons to merge
        margin: Distance for buffering
        preserve_holes: Whether to preserve holes
        simplify: Whether to simplify result to reduce complexity

    Returns:
        Merged polygon(s)
    """
    if len(group_polygons) == 1:
        return group_polygons[0]

    # For overlapping polygons (margin=0), just use unary_union
    if margin <= 0:
        return unary_union(group_polygons)

    # Expand all polygons by margin/2
    buffer_dist = margin / 2.0
    expanded = [p.buffer(buffer_dist, quad_segs=16) for p in group_polygons]

    # Union all expanded polygons
    merged = unary_union(expanded)

    # Contract back by margin/2
    result = merged.buffer(-buffer_dist, quad_segs=16)

    # Handle holes
    result = remove_holes(result, preserve_holes)

    if simplify:
        result = simplify_vwp(result, threshold=margin / 2)

    return result


__all__ = ['merge_simple_buffer']
