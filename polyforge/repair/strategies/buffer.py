"""Buffer repair strategy - uses buffer(0) trick."""

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from ...core.errors import RepairError
from ...core.geometry_utils import safe_buffer_fix


def fix_with_buffer(
    geometry: BaseGeometry,
    buffer_distance: float,
    verbose: bool
) -> BaseGeometry:
    """Fix geometry using the buffer(0) trick.

    The buffer(0) operation often fixes many topology errors.
    """
    fixed = safe_buffer_fix(
        geometry,
        distance=buffer_distance,
        return_largest=isinstance(geometry, Polygon),
    )
    if fixed is None:
        raise RepairError("Buffer repair failed")
    return fixed


__all__ = ['fix_with_buffer']
