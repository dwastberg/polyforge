"""Reconstruct repair strategy - rebuilds geometry from points."""

from shapely.geometry import MultiPoint
from shapely.geometry.base import BaseGeometry

from ...core.errors import RepairError
from ..utils import extract_all_coords


def fix_with_reconstruct(
    geometry: BaseGeometry,
    tolerance: float,
    verbose: bool
) -> BaseGeometry:
    """Fix geometry by reconstructing from points.

    Uses convex hull or point-based reconstruction.
    """
    try:
        geom_type = geometry.geom_type

        if geom_type in ('Polygon', 'MultiPolygon'):
            # Try convex hull
            hull = geometry.convex_hull
            if hull.is_valid:
                return hull

        # Extract all coordinates and rebuild
        coords = extract_all_coords(geometry)
        if len(coords) < 3:
            raise RepairError("Not enough coordinates to reconstruct")

        # Build polygon from points
        points = MultiPoint(coords)
        hull = points.convex_hull

        if hull.is_valid:
            return hull

        raise RepairError("Reconstruction failed")

    except Exception as e:
        raise RepairError(f"Reconstruct repair failed: {e}")


__all__ = ['fix_with_reconstruct']
