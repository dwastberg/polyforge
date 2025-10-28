"""Simplify repair strategy - uses simplification to fix geometry."""

from shapely.geometry.base import BaseGeometry

from ...core.errors import RepairError
from ..utils import clean_coordinates


def fix_with_simplify(
    geometry: BaseGeometry,
    tolerance: float,
    verbose: bool
) -> BaseGeometry:
    """Fix geometry using simplification.

    Simplification can remove problematic vertices causing invalidity.
    """
    try:
        # Clean coordinates first
        cleaned = clean_coordinates(geometry, tolerance)

        # Apply simplification with increasing tolerance
        for epsilon in [tolerance * 10, tolerance * 100, tolerance * 1000]:
            simplified = cleaned.simplify(epsilon, preserve_topology=True)
            if simplified.is_valid and not simplified.is_empty:
                return simplified

        # Last resort: non-topology-preserving simplification
        simplified = cleaned.simplify(tolerance * 1000, preserve_topology=False)
        if simplified.is_valid:
            return simplified

        raise RepairError("Simplification did not produce valid geometry")

    except Exception as e:
        raise RepairError(f"Simplify repair failed: {e}")


__all__ = ['fix_with_simplify']
