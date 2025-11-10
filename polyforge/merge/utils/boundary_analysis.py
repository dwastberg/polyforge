"""Boundary analysis utilities for identifying close polygon segments."""

from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from ...core.spatial_utils import (
    SegmentIndex,
    build_segment_index,
    query_close_segments,
)


def find_close_boundary_pairs(
    polygons: List[Polygon],
    margin: float,
    segment_length: Optional[float] = None
) -> List[Tuple[LineString, LineString, float]]:
    """Find pairs of boundary segments that are within margin distance.

    Args:
        polygons: List of polygons
        margin: Maximum distance threshold
        segment_length: Optional length for discretizing boundaries (auto if None)

    Returns:
        List of (segment1, segment2, distance) tuples
    """
    if segment_length is None:
        segment_length = margin * 2.0 if margin > 0 else 1.0

    index = build_segment_index(polygons, segment_length)
    close_pairs: List[Tuple[LineString, LineString, float]] = []
    seen: set[Tuple[int, int]] = set()

    for seg_idx, segment in enumerate(index.segments):
        owner_i = index.owners[seg_idx][0]
        for cand_idx in query_close_segments(index, seg_idx, margin):
            if cand_idx <= seg_idx:
                continue
            owner_j = index.owners[cand_idx][0]
            if owner_i == owner_j:
                continue
            pair_key = (seg_idx, cand_idx)
            if pair_key in seen:
                continue
            distance = segment.distance(index.segments[cand_idx])
            if distance <= margin:
                close_pairs.append((segment, index.segments[cand_idx], distance))
                seen.add(pair_key)

    return close_pairs


def get_boundary_points_near(
    polygon: Polygon,
    point: Point,
    radius: float
) -> List[Tuple[float, float]]:
    """Extract boundary points within radius of a given point.

    Args:
        polygon: Input polygon
        point: Reference point
        radius: Search radius

    Returns:
        List of coordinate tuples
    """
    coords = list(polygon.exterior.coords)
    close_points = []

    for coord in coords[:-1]:  # Exclude duplicate closing point
        coord_point = Point(coord)
        if coord_point.distance(point) <= radius:
            close_points.append(coord)

    # If we didn't find enough points, also sample along the boundary
    if len(close_points) < 3:
        # Sample points along boundary near the reference point
        boundary = polygon.exterior
        num_samples = max(10, int(boundary.length / 2))

        for i in range(num_samples):
            t = i / num_samples
            sampled_point = boundary.interpolate(t, normalized=True)
            if sampled_point.distance(point) <= radius:
                close_points.append((sampled_point.x, sampled_point.y))

    return close_points


__all__ = ['find_close_boundary_pairs', 'get_boundary_points_near']
