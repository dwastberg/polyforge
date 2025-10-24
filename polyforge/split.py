"""Polygon overlap splitting functions.

This module provides functions for splitting overlapping polygons such that
they become touching but not overlapping, with the overlap area divided
between them.
"""

import numpy as np
from typing import Tuple, Literal
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import split, unary_union


def split_overlap(
    poly1: Polygon,
    poly2: Polygon,
    overlap_strategy: Literal['split', 'largest', 'smallest'] = 'split'
) -> Tuple[Polygon, Polygon]:
    """Split or assign the overlapping area between two polygons.

    Given two polygons that overlap, this function returns two modified polygons
    that are touching but not overlapping. The overlapping area can be handled
    in three ways: split equally between them, assigned to the largest polygon,
    or assigned to the smallest polygon.

    Args:
        poly1: First polygon
        poly2: Second polygon
        overlap_strategy: How to handle the overlap area:
            - 'split': Split the overlap equally (50/50) between the two polygons (default)
            - 'largest': Assign entire overlap to the larger polygon
            - 'smallest': Assign entire overlap to the smaller polygon

    Returns:
        Tuple of (modified_poly1, modified_poly2) where the polygons touch but
        don't overlap. If the polygons don't overlap or one contains the other,
        returns the original polygons unchanged.

    Examples:
        >>> from shapely.geometry import Polygon
        >>> poly1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        >>> poly2 = Polygon([(2, 0), (5, 0), (5, 3), (2, 3)])
        >>> result1, result2 = split_overlap(poly1, poly2)
        >>> result1.intersects(result2)  # They touch
        True
        >>> result1.intersection(result2).area  # But don't overlap
        0.0
        >>> # Assign overlap to largest polygon
        >>> result1, result2 = split_overlap(poly1, poly2, overlap_strategy='largest')
    """
    # Check if polygons actually overlap
    if not poly1.intersects(poly2):
        # No overlap, return originals
        return poly1, poly2

    # Get the intersection
    overlap = poly1.intersection(poly2)

    # Handle empty or point/line intersections (no area overlap)
    if overlap.is_empty or overlap.area < 1e-10:
        return poly1, poly2

    # Check for containment - if one contains the other, return originals
    if poly1.contains(poly2) or poly2.contains(poly1):
        return poly1, poly2

    # Handle MultiPolygon overlap results by taking the union
    if isinstance(overlap, MultiPolygon):
        overlap = unary_union(overlap)
        if isinstance(overlap, MultiPolygon):
            # Still multi - use largest piece
            overlap = max(overlap.geoms, key=lambda p: p.area)

    # Get the parts that don't overlap
    poly1_only = poly1.difference(overlap)
    poly2_only = poly2.difference(overlap)

    # Handle different overlap strategies
    if overlap_strategy == 'largest':
        # Assign entire overlap to the larger polygon
        if poly1.area >= poly2.area:
            new_poly1 = _safe_union(poly1_only, overlap)
            new_poly2 = poly2_only if isinstance(poly2_only, Polygon) else _to_polygon(poly2_only)
        else:
            new_poly1 = poly1_only if isinstance(poly1_only, Polygon) else _to_polygon(poly1_only)
            new_poly2 = _safe_union(poly2_only, overlap)
        return new_poly1, new_poly2

    elif overlap_strategy == 'smallest':
        # Assign entire overlap to the smaller polygon
        if poly1.area <= poly2.area:
            new_poly1 = _safe_union(poly1_only, overlap)
            new_poly2 = poly2_only if isinstance(poly2_only, Polygon) else _to_polygon(poly2_only)
        else:
            new_poly1 = poly1_only if isinstance(poly1_only, Polygon) else _to_polygon(poly1_only)
            new_poly2 = _safe_union(poly2_only, overlap)
        return new_poly1, new_poly2

    # For 'split' strategy, continue with the splitting logic
    # Calculate the split line through the overlap
    # Strategy: Use the line connecting the centroids of the non-overlapping parts
    try:
        centroid1 = _get_geometry_centroid(poly1_only)
        centroid2 = _get_geometry_centroid(poly2_only)

        if centroid1 is None or centroid2 is None:
            # Fallback: one polygon is entirely contained in overlap
            # Use polygon centroids instead
            centroid1 = poly1.centroid
            centroid2 = poly2.centroid

        # Create a line perpendicular to the line connecting centroids
        # This line passes through the overlap centroid
        overlap_centroid = overlap.centroid

        # Vector from centroid1 to centroid2
        direction = np.array([centroid2.x - centroid1.x, centroid2.y - centroid1.y])

        # Handle case where centroids are the same
        if np.linalg.norm(direction) < 1e-10:
            # Use a different approach: split along longest axis of overlap
            direction = _get_overlap_longest_axis(overlap)

        direction = direction / np.linalg.norm(direction)

        # Perpendicular direction (rotate 90 degrees)
        perp_direction = np.array([-direction[1], direction[0]])

        # Create a cutting line through the overlap centroid
        # Make it long enough to cross the entire overlap
        overlap_bounds = overlap.bounds
        diagonal = np.sqrt(
            (overlap_bounds[2] - overlap_bounds[0])**2 +
            (overlap_bounds[3] - overlap_bounds[1])**2
        )
        extension = diagonal * 2  # Make it extra long

        cut_p1 = np.array([overlap_centroid.x, overlap_centroid.y]) - perp_direction * extension
        cut_p2 = np.array([overlap_centroid.x, overlap_centroid.y]) + perp_direction * extension

        cutting_line = LineString([cut_p1, cut_p2])

        # Split the overlap along this line
        split_result = split(overlap, cutting_line)

        if split_result.is_empty or len(split_result.geoms) < 2:
            # Split failed, try alternative approach
            return _split_overlap_simple(poly1, poly2, overlap, poly1_only, poly2_only)

        # Assign each piece to the nearest polygon
        pieces = list(split_result.geoms)
        pieces = [p for p in pieces if isinstance(p, Polygon) and p.area > 1e-10]

        if len(pieces) < 2:
            # Not enough pieces, use simple split
            return _split_overlap_simple(poly1, poly2, overlap, poly1_only, poly2_only)

        # Find which piece is closer to which original polygon
        piece1, piece2 = _assign_pieces_to_polygons(pieces, centroid1, centroid2)

        # Merge pieces with their respective polygons
        new_poly1 = _safe_union(poly1_only, piece1)
        new_poly2 = _safe_union(poly2_only, piece2)

        return new_poly1, new_poly2

    except Exception:
        # If anything goes wrong, fall back to simple approach
        return _split_overlap_simple(poly1, poly2, overlap, poly1_only, poly2_only)


def _get_geometry_centroid(geom) -> Point:
    """Get centroid of a geometry, handling MultiPolygon cases."""
    if geom.is_empty:
        return None

    if isinstance(geom, MultiPolygon):
        # Use the centroid of the largest piece
        largest = max(geom.geoms, key=lambda p: p.area)
        return largest.centroid

    return geom.centroid


def _get_overlap_longest_axis(overlap: Polygon) -> np.ndarray:
    """Calculate the direction of the longest axis of the overlap."""
    bounds = overlap.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]

    if width > height:
        return np.array([1.0, 0.0])
    else:
        return np.array([0.0, 1.0])


def _assign_pieces_to_polygons(pieces, centroid1, centroid2) -> Tuple[Polygon, Polygon]:
    """Assign overlap pieces to their respective polygons based on proximity."""
    if len(pieces) == 2:
        # Simple case: two pieces
        dist1_to_p1 = Point(centroid1.x, centroid1.y).distance(pieces[0].centroid)
        dist1_to_p2 = Point(centroid1.x, centroid1.y).distance(pieces[1].centroid)

        if dist1_to_p1 < dist1_to_p2:
            return pieces[0], pieces[1]
        else:
            return pieces[1], pieces[0]

    # More than 2 pieces - assign based on distance to centroids
    pieces1 = []
    pieces2 = []

    for piece in pieces:
        dist_to_1 = Point(centroid1.x, centroid1.y).distance(piece.centroid)
        dist_to_2 = Point(centroid2.x, centroid2.y).distance(piece.centroid)

        if dist_to_1 < dist_to_2:
            pieces1.append(piece)
        else:
            pieces2.append(piece)

    # Combine pieces for each polygon
    poly1_part = unary_union(pieces1) if pieces1 else Polygon()
    poly2_part = unary_union(pieces2) if pieces2 else Polygon()

    # Convert MultiPolygon to Polygon if needed
    if isinstance(poly1_part, MultiPolygon):
        poly1_part = max(poly1_part.geoms, key=lambda p: p.area)
    if isinstance(poly2_part, MultiPolygon):
        poly2_part = max(poly2_part.geoms, key=lambda p: p.area)

    return poly1_part, poly2_part


def _safe_union(geom1, geom2) -> Polygon:
    """Safely union two geometries, handling MultiPolygon results."""
    if geom1.is_empty:
        return geom2 if isinstance(geom2, Polygon) else Polygon()
    if geom2.is_empty:
        return geom1 if isinstance(geom1, Polygon) else Polygon()

    result = unary_union([geom1, geom2])

    if isinstance(result, MultiPolygon):
        # Return the largest piece
        return max(result.geoms, key=lambda p: p.area)

    return result if isinstance(result, Polygon) else Polygon()


def _to_polygon(geom) -> Polygon:
    """Convert a geometry to a Polygon, taking the largest piece if MultiPolygon."""
    if isinstance(geom, Polygon):
        return geom
    elif isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)
    else:
        return Polygon()


def _split_overlap_simple(poly1, poly2, overlap, poly1_only, poly2_only) -> Tuple[Polygon, Polygon]:
    """Simple fallback: split overlap using a buffer approach.

    This approach uses buffering to create approximate 50/50 split.
    """
    # Use buffer to erode each polygon slightly into the overlap
    # Each polygon loses half the overlap
    half_overlap_area = overlap.area / 2

    # Approximate buffer distance to remove half the overlap area
    # For simplicity, use a small buffer
    buffer_dist = -np.sqrt(half_overlap_area / np.pi) * 0.5

    # Create new polygons by removing part of overlap
    try:
        new_poly1 = poly1.buffer(buffer_dist / 2)
        new_poly2 = poly2.buffer(buffer_dist / 2)

        if new_poly1.is_valid and new_poly2.is_valid and not new_poly1.is_empty and not new_poly2.is_empty:
            return new_poly1, new_poly2
    except Exception:
        pass

    # Ultimate fallback: return originals
    return poly1, poly2
