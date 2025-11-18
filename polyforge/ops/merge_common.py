"""Common preprocessing and postprocessing for merge strategies."""

from typing import List, Optional, Union
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from polyforge.simplify import simplify_vwp
from polyforge.core.geometry_utils import remove_holes


def preprocess_merge_group(
    group_polygons: List[Polygon],
    margin: float,
    preserve_holes: bool
) -> Optional[Union[Polygon, MultiPolygon]]:
    """Common preprocessing for all merge strategies.

    Handles:
    - Single polygon groups (return unchanged)
    - Unary union for overlapping/touching polygons
    - margin=0 case (no gap bridging needed)

    Args:
        group_polygons: Polygons to potentially merge
        margin: Distance threshold
        preserve_holes: Whether to preserve interior holes

    Returns:
        - Result geometry if early exit is possible
        - None if strategy-specific work is needed
    """
    if len(group_polygons) == 1:
        return group_polygons[0]

    base_union = unary_union(group_polygons)

    if isinstance(base_union, Polygon):
        return remove_holes(base_union, preserve_holes)
    if isinstance(base_union, MultiPolygon) and len(base_union.geoms) == 1:
        return remove_holes(base_union.geoms[0], preserve_holes)

    # For overlapping polygons (margin=0), just use unary_union
    if margin <= 0:
        return base_union

    # None signals "continue with strategy-specific logic"
    return None


def postprocess_merge_result(
    result: Union[Polygon, MultiPolygon],
    preserve_holes: bool,
    simplify: bool = False,
    simplify_threshold: Optional[float] = None
) -> Union[Polygon, MultiPolygon]:
    """Common post-processing for merge results.

    Args:
        result: Merged geometry
        preserve_holes: Whether to preserve interior holes
        simplify: Whether to simplify the result
        simplify_threshold: Simplification threshold (used if simplify=True)

    Returns:
        Post-processed geometry
    """
    result = remove_holes(result, preserve_holes)

    if simplify and simplify_threshold is not None:
        result = simplify_vwp(result, threshold=simplify_threshold)

    return result


def union_with_bridges(
    polygons: List[Polygon],
    bridges: List[Polygon],
    preserve_holes: bool
) -> Union[Polygon, MultiPolygon]:
    """Union polygons with bridge geometries.

    Common pattern used by boundary_extension and convex_bridges strategies.

    Args:
        polygons: Original polygons
        bridges: Bridge geometries to connect polygons
        preserve_holes: Whether to preserve interior holes

    Returns:
        Merged geometry
    """
    all_geoms = list(polygons) + bridges
    merged = unary_union(all_geoms)
    return remove_holes(merged, preserve_holes)


__all__ = [
    'preprocess_merge_group',
    'postprocess_merge_result',
    'union_with_bridges',
]
