"""Convex bridges merge strategy - connect with convex hull of close regions."""

from typing import List, Optional, Sequence, Tuple, Union

from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import nearest_points, unary_union

from polyforge.core.geometry_utils import remove_holes
from polyforge.core.spatial_utils import iterate_unique_pairs
from polyforge.ops.merge_ops import get_boundary_points_near
from polyforge.ops.merge_common import union_with_bridges


def merge_convex_bridges(
    group_polygons: List[Polygon],
    margin: float,
    preserve_holes: bool
) -> Union[Polygon, MultiPolygon]:
    """Merge using convex hull of close boundary regions.

    Creates smooth connections for irregular gaps.

    Note: The orchestrator handles preprocessing (single polygon check,
    unary_union for overlapping polygons, margin=0 case). This function
    assumes len(group_polygons) >= 2 and margin > 0.

    Args:
        group_polygons: Polygons to merge (already processed by orchestrator)
        margin: Distance threshold (guaranteed > 0)
        preserve_holes: Whether to preserve holes

    Returns:
        Merged polygon(s)
    """
    bridges = _build_convex_bridges(group_polygons, margin)

    if not bridges:
        # No bridges found, just union the polygons
        merged = unary_union(group_polygons)
        return remove_holes(merged, preserve_holes)

    # Use common bridge merging pattern
    return union_with_bridges(group_polygons, bridges, preserve_holes)


def _build_convex_bridges(
    polygons: List[Polygon],
    margin: float,
) -> List[Polygon]:
    """Return a list of convex bridge polygons for all close pairs."""
    bridges: List[Polygon] = []

    for i, j in iterate_unique_pairs(polygons):
        poly1 = polygons[i]
        poly2 = polygons[j]
        bridge = _bridge_for_polygon_pair(poly1, poly2, margin)
        if bridge is not None:
            bridges.append(bridge)

    return bridges


def _bridge_for_polygon_pair(
    poly1: Polygon,
    poly2: Polygon,
    margin: float,
) -> Optional[Polygon]:
    """Build a bridge polygon between two polygons if they are close."""
    distance = poly1.distance(poly2)
    if distance > margin:
        return None

    pt1, pt2 = nearest_points(poly1, poly2)
    boundary1 = get_boundary_points_near(poly1, pt1, _bridge_search_radius(margin, distance))
    boundary2 = get_boundary_points_near(poly2, pt2, _bridge_search_radius(margin, distance))

    if len(boundary1) < 2 or len(boundary2) < 2:
        return _fallback_point_bridge(pt1, pt2, margin)

    bridge_points = boundary1 + boundary2 + [pt1.coords[0], pt2.coords[0]]
    return _convex_hull_bridge(bridge_points, margin, distance)


def _bridge_search_radius(margin: float, distance: float) -> float:
    """Compute search radius for boundary sampling."""
    return min(margin * 0.75, distance * 2.0 + 0.5)


def _fallback_point_bridge(pt1: Point, pt2: Point, margin: float) -> Optional[Polygon]:
    """Create a simple buffered line between two points."""
    try:
        bridge_line = pt1.buffer(0).union(pt2.buffer(0)).envelope
        buffer_dist = max(margin * 0.5, pt1.distance(pt2) * 0.5 + 0.1)
        bridge = bridge_line.buffer(buffer_dist, quad_segs=4)
        return bridge if bridge.is_valid and bridge.area > 1e-10 else None
    except Exception:
        return None


def _convex_hull_bridge(
    points: Sequence[Tuple[float, float]],
    margin: float,
    distance: float,
) -> Optional[Polygon]:
    """Create a convex hull bridge buffered to ensure overlap."""
    if len(points) < 3:
        return None
    try:
        hull = MultiPoint(points).convex_hull
    except Exception:
        return None

    buffer_dist = max(0.1, distance * 0.05 + 0.05)
    if isinstance(hull, Polygon):
        buffered = hull.buffer(buffer_dist, quad_segs=4)
    else:
        buffered = hull.buffer(max(margin * 0.3, distance * 0.5 + 0.1), quad_segs=4)

    if isinstance(buffered, Polygon) and buffered.is_valid and buffered.area > 1e-10:
        return buffered
    return None


__all__ = ['merge_convex_bridges']
