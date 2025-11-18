"""Simple buffer merge strategy - classic expand-contract method."""

from typing import List, Union
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from polyforge.ops.merge_common import postprocess_merge_result


def merge_simple_buffer(
    group_polygons: List[Polygon],
    margin: float,
    preserve_holes: bool,
    simplify: bool = True
) -> Union[Polygon, MultiPolygon]:
    """Merge using classic expand-contract buffer method.

    Fast and simple, but changes polygon shape (rounds corners).

    Note: The orchestrator handles preprocessing (single polygon check,
    unary_union for overlapping polygons, margin=0 case). This function
    assumes len(group_polygons) >= 2 and margin > 0.

    Args:
        group_polygons: Polygons to merge (already processed by orchestrator)
        margin: Distance for buffering (guaranteed > 0)
        preserve_holes: Whether to preserve holes
        simplify: Whether to simplify result to reduce complexity

    Returns:
        Merged polygon(s)
    """
    # Expand all polygons by margin/2
    buffer_dist = margin / 2.0
    expanded = [p.buffer(buffer_dist, quad_segs=16) for p in group_polygons]

    # Union all expanded polygons
    merged = unary_union(expanded)

    # Contract back by margin/2
    result = merged.buffer(-buffer_dist, quad_segs=16)

    # Common post-processing (hole removal and optional simplification)
    return postprocess_merge_result(
        result,
        preserve_holes=preserve_holes,
        simplify=simplify,
        simplify_threshold=margin / 2
    )


__all__ = ['merge_simple_buffer']
