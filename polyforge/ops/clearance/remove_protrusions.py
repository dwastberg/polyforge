"""Function for removing narrow protrusions.

This module provides a function to identify and remove narrow spike-like
protrusions from polygons based on aspect ratio.
"""

import numpy as np
from shapely.geometry import Polygon
from typing import Optional

from .utils import _point_to_line_perpendicular_distance


def remove_narrow_protrusions(
    geometry: Polygon,
    aspect_ratio_threshold: float = 5.0,
    min_iterations: int = 1,
    max_iterations: int = 10
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
        n = len(coords) - 1  # Exclude closing vertex

        if n < 4:  # Need at least 4 vertices for a valid polygon
            break

        # Find the most problematic narrow protrusion
        best_protrusion = None
        best_aspect_ratio = aspect_ratio_threshold

        for i in range(n):
            # Get three consecutive vertices forming a potential protrusion
            prev_idx = (i - 1) % n
            curr_idx = i
            next_idx = (i + 1) % n

            prev_pt = coords[prev_idx][:2]
            curr_pt = coords[curr_idx][:2]
            next_pt = coords[next_idx][:2]

            # Calculate aspect ratio of this triangle
            aspect_ratio = _calculate_triangle_aspect_ratio(prev_pt, curr_pt, next_pt)

            # Check if this is a narrow protrusion
            if aspect_ratio > best_aspect_ratio:
                best_aspect_ratio = aspect_ratio
                best_protrusion = curr_idx

        # If we found a protrusion to remove
        if best_protrusion is not None:
            # Remove the middle vertex
            new_coords = np.delete(coords, best_protrusion, axis=0)

            # Ensure the ring is still closed
            if not np.allclose(new_coords[0], new_coords[-1]):
                new_coords[-1] = new_coords[0]

            # Create new polygon (preserve holes)
            try:
                if len(result.interiors) > 0:
                    holes = [list(interior.coords) for interior in result.interiors]
                    new_poly = Polygon(new_coords, holes=holes)
                else:
                    new_poly = Polygon(new_coords)

                # Validate the result
                if new_poly.is_valid and not new_poly.is_empty:
                    result = new_poly
                    iteration += 1
                else:
                    # Invalid result, stop trying
                    break
            except Exception:
                # Failed to create valid polygon
                break
        else:
            # No more protrusions found
            if iteration >= min_iterations:
                break
            iteration += 1

    return result


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
