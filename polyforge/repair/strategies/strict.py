"""Strict repair strategy - only conservative fixes."""

from shapely.geometry.base import BaseGeometry

from ...core.errors import RepairError
from ..utils import clean_coordinates


def fix_strict(
    geometry: BaseGeometry,
    tolerance: float,
    verbose: bool
) -> BaseGeometry:
    """Apply only conservative fixes that preserve intent.

    Only applies coordinate cleaning and closing rings.
    """
    cleaned = clean_coordinates(geometry, tolerance)

    if cleaned.is_valid:
        return cleaned

    raise RepairError(
        "Strict mode: geometry cannot be repaired without aggressive changes"
    )


__all__ = ['fix_strict']
