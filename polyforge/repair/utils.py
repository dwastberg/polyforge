from typing import List, Tuple
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.geometry.base import BaseGeometry


def clean_coordinates(
    geometry: BaseGeometry,
    tolerance: float
) -> BaseGeometry:
    """Clean coordinate sequences: remove duplicates, close rings."""
    geom_type = geometry.geom_type

    if geom_type == 'Polygon':
        # Clean exterior
        exterior_coords = np.array(geometry.exterior.coords)
        clean_exterior = remove_duplicate_coords(exterior_coords, tolerance)
        clean_exterior = ensure_closed_ring(clean_exterior)

        # Clean holes
        clean_holes = []
        for hole in geometry.interiors:
            hole_coords = np.array(hole.coords)
            clean_hole = remove_duplicate_coords(hole_coords, tolerance)
            clean_hole = ensure_closed_ring(clean_hole)
            if len(clean_hole) >= 4:  # Valid ring needs at least 4 points
                clean_holes.append(clean_hole)

        return Polygon(clean_exterior, holes=clean_holes)

    elif geom_type == 'LineString':
        coords = np.array(geometry.coords)
        clean_coords = remove_duplicate_coords(coords, tolerance)
        return LineString(clean_coords)

    elif geom_type == 'MultiPolygon':
        clean_polys = []
        for poly in geometry.geoms:
            try:
                clean_poly = clean_coordinates(poly, tolerance)
                if clean_poly.is_valid and not clean_poly.is_empty:
                    clean_polys.append(clean_poly)
            except Exception:
                pass
        return MultiPolygon(clean_polys) if clean_polys else geometry

    return geometry


def remove_duplicate_coords(coords: np.ndarray, tolerance: float) -> np.ndarray:
    """Remove consecutive duplicate coordinates."""
    if len(coords) < 2:
        return coords

    unique_coords = [coords[0]]
    for i in range(1, len(coords)):
        distance = np.linalg.norm(coords[i] - unique_coords[-1])
        if distance > tolerance:
            unique_coords.append(coords[i])

    return np.array(unique_coords)


def ensure_closed_ring(coords: np.ndarray) -> np.ndarray:
    """Ensure coordinate ring is closed (first == last)."""
    if len(coords) < 3:
        return coords

    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0:1]])

    return coords


def extract_all_coords(geometry: BaseGeometry) -> List[Tuple[float, ...]]:
    """Extract all coordinates from any geometry."""
    coords = []

    if hasattr(geometry, 'coords'):
        coords.extend(list(geometry.coords))
    elif hasattr(geometry, 'exterior'):
        coords.extend(list(geometry.exterior.coords))
        for interior in geometry.interiors:
            coords.extend(list(interior.coords))
    elif hasattr(geometry, 'geoms'):
        for geom in geometry.geoms:
            coords.extend(extract_all_coords(geom))

    return coords


__all__ = [
    'clean_coordinates',
    'remove_duplicate_coords',
    'ensure_closed_ring',
    'extract_all_coords',
]
