"""Strict repair strategy - only conservative fixes."""

from shapely.geometry.base import BaseGeometry

from ...core.errors import RepairError
from ..utils import clean_coordinates


def fix_strict(
    geometry: BaseGeometry,
    tolerance: float,
    verbose: bool
) -> BaseGeometry:
    """Apply only conservative fixes that preserve geometric intent.

    This strategy only applies minimal coordinate cleaning:

    - Remove duplicate consecutive vertices
    - Ensure rings are properly closed

    If these conservative fixes don't produce a valid geometry, the
    function raises an error rather than applying aggressive changes.

    Parameters
    ----------
    geometry : BaseGeometry
        The potentially invalid geometry to repair.
    tolerance : float
        Tolerance for identifying duplicate vertices.
    verbose : bool
        If True, print diagnostic messages (currently unused).

    Returns
    -------
    BaseGeometry
        The cleaned geometry if it becomes valid after cleaning.

    Raises
    ------
    RepairError
        If conservative cleaning doesn't produce a valid geometry.
        The error indicates that more aggressive repair strategies are needed.

    Notes
    -----
    Use this strategy when shape preservation is critical and you prefer
    explicit failure over silent shape modifications. This allows callers
    to decide whether to accept shape changes or handle the geometry
    differently.
    """
    cleaned = clean_coordinates(geometry, tolerance)

    if cleaned.is_valid:
        return cleaned

    raise RepairError(
        "Strict mode: geometry cannot be repaired without aggressive changes"
    )


__all__ = ['fix_strict']
