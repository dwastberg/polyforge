"""Selective buffer merge strategy - buffer only near gaps."""

from typing import List, Union
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union

from polyforge.ops.merge_common import postprocess_merge_result
from polyforge.ops.merge_ops import find_close_boundary_pairs


def merge_selective_buffer(
    group_polygons: List[Polygon],
    margin: float,
    preserve_holes: bool,
    simplify: bool = True
) -> Union[Polygon, MultiPolygon]:
    """Merge by buffering only boundaries that are close to each other.

    Better shape preservation than simple buffer.

    Note: The orchestrator handles preprocessing (single polygon check,
    unary_union for overlapping polygons, margin=0 case). This function
    assumes len(group_polygons) >= 2 and margin > 0.

    Args:
        group_polygons: Polygons to merge (already processed by orchestrator)
        margin: Distance threshold (guaranteed > 0)
        preserve_holes: Whether to preserve holes
        simplify: Whether to simplify result to reduce complexity

    Returns:
        Merged polygon(s)
    """
    # Find close boundary segment pairs
    close_segments = find_close_boundary_pairs(group_polygons, margin)

    if not close_segments:
        # No close segments, just union the polygons
        result = unary_union(group_polygons)
        return postprocess_merge_result(
            result,
            preserve_holes=preserve_holes,
            simplify=simplify,
            simplify_threshold=margin / 2
        )

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

    # Common post-processing (hole removal and optional simplification)
    return postprocess_merge_result(
        result,
        preserve_holes=preserve_holes,
        simplify=simplify,
        simplify_threshold=margin / 2
    )


__all__ = ['merge_selective_buffer']
