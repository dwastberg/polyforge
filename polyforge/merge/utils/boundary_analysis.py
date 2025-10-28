"""Boundary analysis utilities for identifying close polygon segments."""

from typing import List, Tuple, Optional
import numpy as np
from shapely.geometry import Polygon, LineString, Point


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
    # Auto-determine segment length based on margin
    if segment_length is None:
        segment_length = margin * 2.0

    # Extract boundary segments from all polygons
    all_segments = []
    for poly_idx, poly in enumerate(polygons):
        coords = list(poly.exterior.coords)

        # Discretize boundary into segments
        for i in range(len(coords) - 1):
            seg = LineString([coords[i], coords[i + 1]])

            # Further subdivide long segments
            seg_len = seg.length
            if seg_len > segment_length:
                # Split into smaller segments
                num_splits = int(np.ceil(seg_len / segment_length))
                for j in range(num_splits):
                    start = j / num_splits
                    end = (j + 1) / num_splits
                    subseg = LineString([
                        seg.interpolate(start, normalized=True).coords[0],
                        seg.interpolate(end, normalized=True).coords[0]
                    ])
                    all_segments.append((poly_idx, subseg))
            else:
                all_segments.append((poly_idx, seg))

    # Find close segment pairs from different polygons
    close_pairs = []
    for i, (poly_idx_i, seg_i) in enumerate(all_segments):
        for j, (poly_idx_j, seg_j) in enumerate(all_segments[i + 1:], i + 1):
            # Only consider segments from different polygons
            if poly_idx_i != poly_idx_j:
                distance = seg_i.distance(seg_j)
                if distance <= margin:
                    close_pairs.append((seg_i, seg_j, distance))

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
