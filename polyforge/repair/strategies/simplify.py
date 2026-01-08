"""Simplify repair strategy - uses simplification to fix geometry."""

from shapely.geometry.base import BaseGeometry

from ...core.errors import RepairError
from ..utils import clean_coordinates


def fix_with_simplify(
    geometry: BaseGeometry,
    tolerance: float,
    verbose: bool
) -> BaseGeometry:
    """Fix geometry using progressive simplification.

    Simplification can remove problematic vertices that cause invalidity.
    This function tries increasingly aggressive simplification until a
    valid result is achieved.

    The simplification sequence is:

    1. Clean coordinates (remove duplicates, close rings)
    2. Simplify with tolerance × 10 (topology preserving)
    3. Simplify with tolerance × 100 (topology preserving)
    4. Simplify with tolerance × 1000 (topology preserving)
    5. Simplify with tolerance × 1000 (non-topology preserving, last resort)

    Parameters
    ----------
    geometry : BaseGeometry
        The potentially invalid geometry to repair.
    tolerance : float
        Base tolerance. Actual simplification uses 10x to 1000x this value.
    verbose : bool
        If True, print diagnostic messages (currently unused).

    Returns
    -------
    BaseGeometry
        A valid, simplified geometry.

    Raises
    ------
    RepairError
        If no simplification level produces a valid geometry.

    Notes
    -----
    This strategy reduces vertex count and may significantly change
    the geometry shape. For shape-preserving repairs, prefer BUFFER
    or STRICT strategies.
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
