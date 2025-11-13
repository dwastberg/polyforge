"""Function for removing narrow protrusions.

This module provides a function to identify and remove narrow spike-like
protrusions from polygons based on aspect ratio.
"""

import numpy as np
from shapely.geometry import Polygon
from typing import List, Optional, Tuple

from .utils import _point_to_line_perpendicular_distance


class _ProtrusionCandidate(Tuple[int, float]): ...


def remove_narrow_protrusions(
    geometry: Polygon,
    aspect_ratio_threshold: float = 5.0,
    min_iterations: int = 1,
    max_iterations: int = 10,
) -> Polygon:
    """Remove narrow protrusions by identifying high aspect ratio triangles.

    A narrow protrusion is defined as three consecutive vertices forming a triangle
    with a very high aspect ratio (length >> width). This function identifies such
    protrusions and removes the middle vertex, effectively cutting off the spike.

    Args:
        geometry: Input polygon
        aspect_ratio_threshold: Minimum aspect ratio to consider a protrusion
            (default: 5.0). Higher values = only remove very narrow spikes.
        min_iterations: Minimum number of iterations even if no protrusions found
            (default: 1)
        max_iterations: Maximum iterations to prevent infinite loops (default: 10)

    Returns:
        Polygon with narrow protrusions removed

    Examples:
        >>> # Polygon with narrow horizontal spike
        >>> coords = [(0, 0), (10, 0), (10, 4), (10, 4.9), (12, 5), (10, 5.1),
        ...           (10, 6), (0, 6)]
        >>> poly = Polygon(coords)
        >>> fixed = remove_narrow_protrusions(poly, aspect_ratio_threshold=5.0)
        >>> # Spike vertices removed, base vertices connected directly

    Notes:
        - Aspect ratio = max_edge_length / height_to_base
        - Only removes protrusions where the tip vertex extends significantly
        - Preserves interior rings (holes)
        - Multiple iterations handle cases where removing one protrusion reveals another
    """
    if not isinstance(geometry, Polygon):
        raise TypeError("Geometry must be a Polygon")

    result = geometry
    iteration = 0

    while iteration < max_iterations:
        coords = np.array(result.exterior.coords)
        if len(coords) - 1 < 4:
            break

        candidate = _collect_protrusion_candidate(coords, aspect_ratio_threshold)
        if candidate is None:
            if iteration >= min_iterations:
                break
            iteration += 1
            continue

        new_poly = _remove_candidate_vertex(result, coords, candidate[0])
        if new_poly is None:
            break

        result = new_poly
        iteration += 1

    return result


def _collect_protrusion_candidate(
    coords: np.ndarray,
    threshold: float,
) -> Optional[Tuple[int, float]]:
    """Return the vertex index with highest aspect ratio beyond the threshold."""
    n = len(coords) - 1
    best_idx: Optional[int] = None
    best_ratio = threshold

    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        prev_pt = coords[prev_idx][:2]
        curr_pt = coords[i][:2]
        next_pt = coords[next_idx][:2]

        aspect_ratio = _calculate_triangle_aspect_ratio(prev_pt, curr_pt, next_pt)
        if aspect_ratio > best_ratio:
            best_ratio = aspect_ratio
            best_idx = i

    if best_idx is None:
        return None
    return best_idx, best_ratio


def _remove_candidate_vertex(
    geometry: Polygon,
    coords: np.ndarray,
    index: int,
) -> Optional[Polygon]:
    """Remove the vertex at index and rebuild a polygon."""
    new_coords = np.delete(coords, index, axis=0)
    if not np.allclose(new_coords[0], new_coords[-1]):
        new_coords[-1] = new_coords[0]

    holes = [list(interior.coords) for interior in geometry.interiors]
    try:
        new_poly = Polygon(new_coords, holes=holes) if holes else Polygon(new_coords)
    except Exception:
        return None

    if not new_poly.is_valid or new_poly.is_empty:
        return None

    return new_poly


def _calculate_triangle_aspect_ratio(pt1: np.ndarray, pt2: np.ndarray, pt3: np.ndarray) -> float:
    """Calculate aspect ratio of a triangle formed by three points.

    Aspect ratio is defined as the ratio of the longest edge to the
    perpendicular distance from the opposite vertex to that edge.

    High aspect ratio indicates a long, narrow triangle (like a spike).

    Args:
        pt1: First point [x, y]
        pt2: Middle point (potential spike tip) [x, y]
        pt3: Third point [x, y]

    Returns:
        Aspect ratio (length / width). Higher = narrower triangle.
    """
    # Calculate all three edge lengths
    edge1 = np.linalg.norm(pt2 - pt1)  # pt1 to pt2
    edge2 = np.linalg.norm(pt3 - pt2)  # pt2 to pt3
    edge3 = np.linalg.norm(pt3 - pt1)  # pt1 to pt3 (base)

    # Find the longest edge (this will be considered the "length")
    max_edge = max(edge1, edge2, edge3)

    # Calculate height from the opposite vertex to the longest edge
    if max_edge == edge3:
        # Base is pt1-pt3, measure distance from pt2 to this line
        height = _point_to_line_perpendicular_distance(pt2, pt1, pt3)
    elif max_edge == edge1:
        # Base is pt1-pt2, measure distance from pt3 to this line
        height = _point_to_line_perpendicular_distance(pt3, pt1, pt2)
    else:  # max_edge == edge2
        # Base is pt2-pt3, measure distance from pt1 to this line
        height = _point_to_line_perpendicular_distance(pt1, pt2, pt3)

    # Avoid division by zero
    if height < 1e-10:
        return 0.0

    # Aspect ratio = length / width
    aspect_ratio = max_edge / height

    return aspect_ratio


__all__ = [
    'remove_narrow_protrusions',
]
