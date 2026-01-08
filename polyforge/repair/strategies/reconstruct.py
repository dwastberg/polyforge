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
    """Fix geometry by reconstructing from its convex hull.

    This is a last-resort strategy that reconstructs the geometry using
    its convex hull. The result is always valid but loses concave features
    and interior holes.

    Parameters
    ----------
    geometry : BaseGeometry
        The potentially invalid geometry to repair.
    tolerance : float
        Tolerance parameter (currently unused, kept for API consistency).
    verbose : bool
        If True, print diagnostic messages (currently unused).

    Returns
    -------
    BaseGeometry
        The convex hull of the input geometry.

    Raises
    ------
    RepairError
        If the geometry has fewer than 3 coordinates or convex hull fails.

    Warnings
    --------
    This strategy produces significant shape changes:

    - All concave features are removed
    - All interior holes are removed
    - Result is always convex

    Only use when other strategies have failed.
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
