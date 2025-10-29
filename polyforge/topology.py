"""Topological operations for polygon boundaries.

This module provides functions for ensuring topological consistency between
polygons, such as conforming boundaries and vertex alignment.
"""

from typing import Tuple, List, Dict
import numpy as np
from shapely.geometry import Polygon, LinearRing


def align_boundaries(
    poly1: Polygon,
    poly2: Polygon,
    distance_tolerance: float = 1e-10
) -> Tuple[Polygon, Polygon]:
    """Align two touching polygons to have conforming boundaries.

    When two polygons share a boundary, this function ensures that any vertex
    from one polygon that lies on an edge of the other polygon is added as a
    vertex to that edge. This creates a "conforming" mesh where boundaries
    align perfectly at all vertices, eliminating T-junctions.

    Args:
        poly1: First polygon
        poly2: Second polygon
        distance_tolerance: Distance tolerance for considering a point on an edge (default: 1e-10)

    Returns:
        Tuple of (modified_poly1, modified_poly2) with aligned boundaries

    Examples:
        >>> # Two squares sharing an edge, one has vertex at midpoint
        >>> poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        >>> poly2 = Polygon([(2, 0), (4, 0), (4, 2), (2, 1), (2, 2)])  # vertex at (2, 1)
        >>> result1, result2 = align_boundaries(poly1, poly2)
        >>> # result1 now has vertex at (2, 1) on its right edge

    Notes:
        - Polygons must be valid and should be touching (but not overlapping)
        - Works with polygons containing holes
        - Preserves winding order
        - Vertices already existing (within tolerance) are not duplicated
    """
    # Get vertices that need to be added to each polygon
    vertices_for_poly1 = _find_vertices_on_edges(poly2, poly1, distance_tolerance)
    vertices_for_poly2 = _find_vertices_on_edges(poly1, poly2, distance_tolerance)

    # Insert vertices into polygon exteriors
    new_poly1_exterior = _insert_vertices_into_ring(
        poly1.exterior,
        vertices_for_poly1.get('exterior', []),
        distance_tolerance
    )
    new_poly2_exterior = _insert_vertices_into_ring(
        poly2.exterior,
        vertices_for_poly2.get('exterior', []),
        distance_tolerance
    )

    # Handle holes if present
    new_poly1_holes = []
    if poly1.interiors:
        for hole_idx, hole in enumerate(poly1.interiors):
            hole_vertices = vertices_for_poly1.get(f'hole_{hole_idx}', [])
            new_hole = _insert_vertices_into_ring(hole, hole_vertices, distance_tolerance)
            new_poly1_holes.append(new_hole)

    new_poly2_holes = []
    if poly2.interiors:
        for hole_idx, hole in enumerate(poly2.interiors):
            hole_vertices = vertices_for_poly2.get(f'hole_{hole_idx}', [])
            new_hole = _insert_vertices_into_ring(hole, hole_vertices, distance_tolerance)
            new_poly2_holes.append(new_hole)

    # Create new polygons
    result_poly1 = Polygon(new_poly1_exterior, holes=new_poly1_holes)
    result_poly2 = Polygon(new_poly2_exterior, holes=new_poly2_holes)

    return result_poly1, result_poly2


def _find_vertices_on_edges(
    source_poly: Polygon,
    target_poly: Polygon,
    tolerance: float
) -> Dict[str, List[Tuple[float, float]]]:
    """Find vertices from source_poly that lie on edges of target_poly.

    Args:
        source_poly: Polygon whose vertices to check
        target_poly: Polygon whose edges to check against
        tolerance: Distance tolerance

    Returns:
        Dictionary mapping ring names to lists of vertices to insert:
        - 'exterior': [(x, y), ...]
        - 'hole_0', 'hole_1', etc.: [(x, y), ...]
    """
    result = {}

    # Check source vertices against target exterior
    source_coords = np.array(source_poly.exterior.coords[:-1])  # Exclude closing point
    target_exterior_coords = np.array(target_poly.exterior.coords)

    vertices_for_exterior = []
    for source_pt in source_coords:
        if _point_on_any_edge(source_pt, target_exterior_coords, tolerance):
            vertices_for_exterior.append(tuple(source_pt))

    if vertices_for_exterior:
        result['exterior'] = vertices_for_exterior

    # Check source vertices against target holes
    if target_poly.interiors:
        for hole_idx, hole in enumerate(target_poly.interiors):
            hole_coords = np.array(hole.coords)
            vertices_for_hole = []

            for source_pt in source_coords:
                if _point_on_any_edge(source_pt, hole_coords, tolerance):
                    vertices_for_hole.append(tuple(source_pt))

            if vertices_for_hole:
                result[f'hole_{hole_idx}'] = vertices_for_hole

    # Also check source holes against target exterior and holes
    if source_poly.interiors:
        for source_hole in source_poly.interiors:
            source_hole_coords = np.array(source_hole.coords[:-1])

            # Against target exterior
            for source_pt in source_hole_coords:
                if _point_on_any_edge(source_pt, target_exterior_coords, tolerance):
                    if 'exterior' not in result:
                        result['exterior'] = []
                    if tuple(source_pt) not in result['exterior']:
                        result['exterior'].append(tuple(source_pt))

            # Against target holes
            if target_poly.interiors:
                for hole_idx, target_hole in enumerate(target_poly.interiors):
                    target_hole_coords = np.array(target_hole.coords)
                    for source_pt in source_hole_coords:
                        if _point_on_any_edge(source_pt, target_hole_coords, tolerance):
                            key = f'hole_{hole_idx}'
                            if key not in result:
                                result[key] = []
                            if tuple(source_pt) not in result[key]:
                                result[key].append(tuple(source_pt))

    return result


def _point_on_any_edge(
    point: np.ndarray,
    ring_coords: np.ndarray,
    tolerance: float
) -> bool:
    """Check if a point lies on any edge of a ring (excluding vertices).

    Args:
        point: Point to check (2D)
        ring_coords: Ring coordinates (Nx2 or Nx3, including closing point)
        tolerance: Distance tolerance

    Returns:
        True if point is on any edge (not at existing vertices)
    """
    point_2d = point[:2] if len(point) > 2 else point

    for i in range(len(ring_coords) - 1):
        edge_start = ring_coords[i][:2] if len(ring_coords[i]) > 2 else ring_coords[i]
        edge_end = ring_coords[i + 1][:2] if len(ring_coords[i + 1]) > 2 else ring_coords[i + 1]

        # Skip if point is already a vertex (within tolerance)
        if np.linalg.norm(point_2d - edge_start) < tolerance:
            return False
        if np.linalg.norm(point_2d - edge_end) < tolerance:
            return False

        # Check if point is on this edge
        if _point_on_segment(point_2d, edge_start, edge_end, tolerance):
            return True

    return False


def _point_on_segment(
    point: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    tolerance: float
) -> bool:
    """Check if a point lies on a line segment (excluding endpoints).

    Uses parametric representation: P = A + t(B-A) where 0 < t < 1

    Args:
        point: Point to check (2D)
        seg_start: Segment start point (2D)
        seg_end: Segment end point (2D)
        tolerance: Distance tolerance

    Returns:
        True if point is on segment (not at endpoints)
    """
    # Vector from start to end
    seg_vec = seg_end - seg_start
    seg_length_sq = np.dot(seg_vec, seg_vec)

    if seg_length_sq < tolerance * tolerance:
        # Degenerate segment
        return False

    # Project point onto line
    t = np.dot(point - seg_start, seg_vec) / seg_length_sq

    # Check if projection is within segment (excluding endpoints)
    if t <= tolerance or t >= 1.0 - tolerance:
        return False

    # Calculate closest point on segment
    closest_point = seg_start + t * seg_vec

    # Check distance from point to segment
    distance = np.linalg.norm(point - closest_point)

    return distance < tolerance


def _insert_vertices_into_ring(
    ring: LinearRing,
    vertices_to_insert: List[Tuple[float, float]],
    tolerance: float
) -> List[Tuple[float, ...]]:
    """Insert vertices into a ring while maintaining edge topology.

    Args:
        ring: Original ring
        vertices_to_insert: List of (x, y) coordinates to insert
        tolerance: Distance tolerance for comparisons

    Returns:
        New coordinate list with inserted vertices
    """
    if not vertices_to_insert:
        # No vertices to insert, return original
        return list(ring.coords)

    coords = np.array(ring.coords)
    new_coords = []

    # For each edge, collect vertices that belong on that edge
    for i in range(len(coords) - 1):
        edge_start = coords[i][:2] if len(coords[i]) > 2 else coords[i]
        edge_end = coords[i + 1][:2] if len(coords[i + 1]) > 2 else coords[i + 1]

        # Add start vertex
        new_coords.append(tuple(coords[i]))

        # Find vertices that lie on this edge
        edge_vertices = []
        for vertex in vertices_to_insert:
            vertex_2d = np.array(vertex[:2])
            if _point_on_segment(vertex_2d, edge_start, edge_end, tolerance):
                # Calculate parameter t for sorting
                seg_vec = edge_end - edge_start
                seg_length_sq = np.dot(seg_vec, seg_vec)
                if seg_length_sq > tolerance * tolerance:
                    t = np.dot(vertex_2d - edge_start, seg_vec) / seg_length_sq
                    edge_vertices.append((t, vertex))

        # Sort by parameter t and insert in order
        edge_vertices.sort(key=lambda x: x[0])
        for _, vertex in edge_vertices:
            new_coords.append(vertex)

    # Add closing point (should match first point)
    new_coords.append(new_coords[0])

    return new_coords


__all__ = [
    'align_boundaries',
]
