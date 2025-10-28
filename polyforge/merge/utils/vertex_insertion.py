"""Vertex insertion utilities for optimal merge connection points."""

from typing import List
from shapely.geometry import Polygon, LineString
from shapely.ops import nearest_points


def insert_connection_vertices(
    polygons: List[Polygon],
    margin: float,
    tolerance: float = 0.01
) -> List[Polygon]:
    """Insert vertices at optimal connection points between close polygons.

    For each pair of polygons within margin distance, finds the closest points
    on their boundaries. If a closest point lies on an edge (not at an existing
    vertex), inserts a new vertex at that location. This gives subsequent merge
    strategies optimal anchor points for creating minimal bridges.

    Args:
        polygons: List of polygons to process
        margin: Maximum distance for considering polygons close
        tolerance: Minimum distance from existing vertex to insert new one (default: 0.01)

    Returns:
        List of polygons with new vertices inserted at connection points

    Notes:
        - Only inserts vertices when closest point is > tolerance from existing vertices
        - Inserts at closest point per edge pair (one per close edge)
        - Preserves holes and Z-coordinates if present
    """
    if len(polygons) < 2:
        return polygons

    # Build mapping of which polygons to modify
    modified_coords = {}  # poly_idx -> new exterior coords

    # Find close edge pairs between polygons
    # This approach ensures symmetry - both polygons in a pair get vertices
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            poly_i = polygons[i]
            poly_j = polygons[j]

            distance = poly_i.distance(poly_j)
            if distance > margin:
                continue

            # Initialize coordinate lists if not already done
            if i not in modified_coords:
                modified_coords[i] = list(poly_i.exterior.coords)
            if j not in modified_coords:
                modified_coords[j] = list(poly_j.exterior.coords)

            # Get closest points between the two polygons
            pt_i, pt_j = nearest_points(poly_i.boundary, poly_j.boundary)

            # Process both polygons to ensure symmetry
            insertions = []  # List of (poly_idx, edge_idx, new_vertex)

            for poly_idx, poly, pt in [(i, poly_i, pt_i), (j, poly_j, pt_j)]:
                coords = modified_coords[poly_idx]
                pt_coords = pt.coords[0]

                # Check if point is already at a vertex (within tolerance)
                is_at_vertex = False
                for coord in coords[:-1]:  # Exclude closing vertex
                    dist_to_vertex = ((coord[0] - pt_coords[0])**2 +
                                     (coord[1] - pt_coords[1])**2)**0.5
                    if dist_to_vertex < tolerance:
                        is_at_vertex = True
                        break

                if is_at_vertex:
                    continue  # Skip insertion, already at vertex

                # Find which edge the point lies on
                for k in range(len(coords) - 1):
                    seg = LineString([coords[k], coords[k + 1]])
                    dist_to_seg = seg.distance(pt)

                    if dist_to_seg < 1e-6:  # Point is on this edge
                        # Prepare vertex insertion
                        # Determine 2D or 3D
                        if len(pt_coords) == 2 and len(coords[k]) == 3:
                            # 3D coords, interpolate Z
                            seg_2d = LineString([(coords[k][0], coords[k][1]),
                                                 (coords[k+1][0], coords[k+1][1])])
                            dist_along = seg_2d.project(pt)
                            total_length = seg_2d.length
                            if total_length > 1e-10:
                                t = dist_along / total_length
                                z = coords[k][2] + t * (coords[k+1][2] - coords[k][2])
                                new_vertex = (pt_coords[0], pt_coords[1], z)
                            else:
                                new_vertex = coords[k]
                        elif len(pt_coords) == 3:
                            new_vertex = pt_coords
                        else:
                            new_vertex = pt_coords

                        insertions.append((poly_idx, k, new_vertex))
                        break

            # Apply insertions (sorted by poly_idx, then edge_idx descending to avoid index shifts)
            for poly_idx, edge_idx, new_vertex in sorted(insertions, key=lambda x: (x[0], -x[1])):
                coords = modified_coords[poly_idx]
                coords.insert(edge_idx + 1, new_vertex)
                modified_coords[poly_idx] = coords

    # Reconstruct polygons with new vertices
    result = []
    for i, poly in enumerate(polygons):
        if i in modified_coords:
            # Create new polygon with modified exterior
            new_coords = modified_coords[i]
            # Ensure closed ring
            if new_coords[0] != new_coords[-1]:
                new_coords.append(new_coords[0])

            # Preserve holes
            if poly.interiors:
                holes = [list(hole.coords) for hole in poly.interiors]
                result.append(Polygon(new_coords, holes=holes))
            else:
                result.append(Polygon(new_coords))
        else:
            # No modification needed
            result.append(poly)

    return result


__all__ = ['insert_connection_vertices']
