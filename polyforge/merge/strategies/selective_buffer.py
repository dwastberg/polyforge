"""Selective buffer merge strategy - buffer only near gaps."""

from typing import List, Union
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union

from ...simplify import simplify_vwp
from ..utils.boundary_analysis import find_close_boundary_pairs


def merge_selective_buffer(
    group_polygons: List[Polygon],
    margin: float,
    preserve_holes: bool,
    simplify: bool = True
) -> Union[Polygon, MultiPolygon]:
    """Merge by buffering only boundaries that are close to each other.

    Better shape preservation than simple buffer.

    Args:
        group_polygons: Polygons to merge
        margin: Distance threshold
        preserve_holes: Whether to preserve holes
        simplify: Whether to simplify result to reduce complexity

    Returns:
        Merged polygon(s)
    """
    if len(group_polygons) == 1:
        return group_polygons[0]

    # For overlapping polygons, just use unary_union
    if margin <= 0:
        return unary_union(group_polygons)

    # Find close boundary segment pairs
    close_segments = find_close_boundary_pairs(group_polygons, margin)

    if not close_segments:
        # No close segments, just union
        return unary_union(group_polygons)

    # Create minimal bridge zones between close segments
    buffer_zones = []

    # Filter to only the closest segment pairs to avoid over-bridging
    # Group by distance and only use segments within a tight threshold
    min_distance = min(dist for _, _, dist in close_segments)
    tolerance = min(margin * 0.2, 0.5)  # Only use segments very close to minimum
    close_segments_filtered = [
        (seg1, seg2, dist) for seg1, seg2, dist in close_segments
        if dist <= min_distance + tolerance
    ]

    for seg1, seg2, distance in close_segments_filtered:
        # Create a minimal rectangular bridge connecting the segments
        # Buffer distance should just span the gap, not the margin
        buffer_dist = distance / 2.0 + 0.1  # Just enough to overlap both sides

        # Create LineString connecting segment midpoints
        mid1 = seg1.centroid
        mid2 = seg2.centroid
        connector = LineString([mid1.coords[0], mid2.coords[0]])

        # Use minimal quad_segs for more rectangular bridges
        bridge = connector.buffer(buffer_dist, quad_segs=2)
        buffer_zones.append(bridge)

    # Union original polygons with buffer zones
    all_geoms = list(group_polygons) + buffer_zones
    result = unary_union(all_geoms)

    # Handle holes
    if not preserve_holes and isinstance(result, Polygon) and result.interiors:
        result = Polygon(result.exterior)
    elif not preserve_holes and isinstance(result, MultiPolygon):
        result = MultiPolygon([Polygon(p.exterior) for p in result.geoms])

    if simplify:
        result = simplify_vwp(result, threshold=margin / 2)

    return result


__all__ = ['merge_selective_buffer']
