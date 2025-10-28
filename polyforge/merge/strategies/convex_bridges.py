"""Convex bridges merge strategy - connect with convex hull of close regions."""

from typing import List, Union
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, MultiPoint
from shapely.ops import unary_union, nearest_points

from ..utils.boundary_analysis import get_boundary_points_near


def merge_convex_bridges(
    group_polygons: List[Polygon],
    margin: float,
    preserve_holes: bool
) -> Union[Polygon, MultiPolygon]:
    """Merge using convex hull of close boundary regions.

    Creates smooth connections for irregular gaps.

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

    bridges = []

    # Find close pairs and create convex bridges
    for i, poly1 in enumerate(group_polygons):
        for j, poly2 in enumerate(group_polygons[i + 1:], i + 1):
            distance = poly1.distance(poly2)
            if distance > margin:
                continue

            # Get closest points between the two polygons
            pt1, pt2 = nearest_points(poly1, poly2)

            # Extract boundary points near the gap on each polygon
            # Key: use a very tight search radius to avoid collecting distant points
            # that would create diagonal bridges
            search_radius = min(margin * 0.75, distance * 2.0 + 0.5)
            boundary1_close = get_boundary_points_near(poly1, pt1, search_radius)
            boundary2_close = get_boundary_points_near(poly2, pt2, search_radius)

            # Need at least 2 points from each polygon
            if len(boundary1_close) < 2 or len(boundary2_close) < 2:
                # Fall back to simple point-to-point bridge
                bridge_line = LineString([pt1.coords[0], pt2.coords[0]])
                bridge = bridge_line.buffer(max(margin * 0.5, 0.1), quad_segs=4)
                if bridge.is_valid and bridge.area > 1e-10:
                    bridges.append(bridge)
                continue

            # Create bridge by finding the convex hull, but only of nearby points
            # The tight search_radius ensures we don't get distant corners
            try:
                bridge_points = []

                # Add boundary points from both polygons
                bridge_points.extend(boundary1_close)
                bridge_points.extend(boundary2_close)

                # Always include the actual closest points
                bridge_points.append(pt1.coords[0])
                bridge_points.append(pt2.coords[0])

                if len(bridge_points) >= 3:
                    # Create convex hull of the nearby points only
                    bridge = MultiPoint(bridge_points).convex_hull

                    # Buffer the bridge slightly to ensure it overlaps with both polygons
                    # This is critical - without overlap, union won't merge them
                    if isinstance(bridge, LineString):
                        # LineString needs more buffering
                        buffer_dist = max(margin * 0.3, distance * 0.5 + 0.1)
                        bridge = bridge.buffer(buffer_dist, quad_segs=4)
                    elif isinstance(bridge, Polygon):
                        # Small polygon bridge still needs buffering to ensure overlap
                        buffer_dist = max(0.1, distance * 0.05 + 0.05)
                        bridge = bridge.buffer(buffer_dist, quad_segs=4)

                    if isinstance(bridge, Polygon) and bridge.is_valid and bridge.area > 1e-10:
                        bridges.append(bridge)
            except Exception:
                # Skip if bridge creation fails
                continue

    # Union all
    all_geoms = list(group_polygons) + bridges
    result = unary_union(all_geoms)

    # Handle holes
    if not preserve_holes and isinstance(result, Polygon) and result.interiors:
        result = Polygon(result.exterior)
    elif not preserve_holes and isinstance(result, MultiPolygon):
        result = MultiPolygon([Polygon(p.exterior) for p in result.geoms])

    return result


__all__ = ['merge_convex_bridges']
