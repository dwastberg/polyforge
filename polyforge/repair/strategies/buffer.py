"""Buffer repair strategy - uses buffer(0) trick."""

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from ...core.errors import RepairError
from ...core.geometry_utils import to_single_polygon


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

        # Convert to single polygon if needed
        if isinstance(geometry, Polygon):
            fixed = to_single_polygon(fixed)

        return fixed

    except Exception as e:
        raise RepairError(f"Buffer repair failed: {e}")


__all__ = ['fix_with_buffer']
