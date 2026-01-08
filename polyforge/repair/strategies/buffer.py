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
    """Fix geometry using the buffer operation.

    The buffer(0) operation is a well-known technique that often fixes
    self-intersections and other topological errors. It works by rebuilding
    the geometry from its boundary representation.

    For Polygon inputs, returns the largest polygon if buffering produces
    a MultiPolygon (which can happen with complex self-intersections).

    Parameters
    ----------
    geometry : BaseGeometry
        The potentially invalid geometry to repair.
    buffer_distance : float
        Distance for buffer operation. Typically 0 for repair purposes.
        Non-zero values can help with very complex geometries.
    verbose : bool
        If True, print diagnostic messages (currently unused).

    Returns
    -------
    BaseGeometry
        A valid geometry. For Polygon input, returns Polygon (largest if
        buffering produces MultiPolygon).

    Raises
    ------
    RepairError
        If buffer operation fails or produces invalid/empty geometry.

    Notes
    -----
    The buffer(0) trick works because Shapely/GEOS rebuilds the geometry
    from scratch during the buffer operation, resolving many topological
    inconsistencies in the process.
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
