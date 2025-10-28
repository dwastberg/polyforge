"""Buffer repair strategy - uses buffer(0) trick."""

from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.geometry.base import BaseGeometry

from ...core.errors import RepairError


def fix_with_buffer(
    geometry: BaseGeometry,
    buffer_distance: float,
    verbose: bool
) -> BaseGeometry:
    """Fix geometry using the buffer(0) trick.

    The buffer(0) operation often fixes many topology errors.
    """
    try:
        # Try buffer(0) first
        fixed = geometry.buffer(buffer_distance)

        # Handle MultiPolygon results
        if isinstance(fixed, MultiPolygon) and isinstance(geometry, Polygon):
            # Return largest piece
            fixed = max(fixed.geoms, key=lambda p: p.area)

        # Handle GeometryCollection
        if isinstance(fixed, GeometryCollection):
            # Extract polygons
            polygons = [g for g in fixed.geoms if g.geom_type == 'Polygon']
            if polygons:
                fixed = max(polygons, key=lambda p: p.area)
            else:
                raise RepairError("Buffer produced no valid polygons")

        return fixed

    except Exception as e:
        raise RepairError(f"Buffer repair failed: {e}")


__all__ = ['fix_with_buffer']
