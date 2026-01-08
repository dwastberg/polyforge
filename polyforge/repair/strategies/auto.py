"""Auto repair strategy - tries multiple approaches."""

from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

from ...core.errors import RepairError
from ..utils import clean_coordinates
from .buffer import fix_with_buffer
from .simplify import fix_with_simplify
from .reconstruct import fix_with_reconstruct


def auto_fix_geometry(
    geometry: BaseGeometry,
    buffer_distance: float,
    tolerance: float,
    verbose: bool
) -> BaseGeometry:
    """Automatically detect and fix geometry issues using multiple strategies.

    This function tries repair strategies in order of least to most aggressive:

    1. **Clean coordinates** - Remove duplicate vertices, ensure rings are closed
    2. **Buffer(0) trick** - Often fixes self-intersections and topology errors
    3. **Simplification** - Remove problematic vertices with increasing tolerance
    4. **Reconstruction** - Rebuild from convex hull (last resort)

    The first strategy that produces a valid geometry is returned.

    Parameters
    ----------
    geometry : BaseGeometry
        The potentially invalid geometry to repair.
    buffer_distance : float
        Distance for buffer operations. Typically 0 for the buffer(0) trick.
    tolerance : float
        Base tolerance for coordinate cleaning and simplification.
        Simplification tries 10x, 100x, and 1000x this value progressively.
    verbose : bool
        If True, print progress messages for each strategy attempted.

    Returns
    -------
    BaseGeometry
        A valid geometry, preserving the original type where possible.

    Raises
    ------
    RepairError
        If all strategies fail to produce a valid geometry.

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> invalid = Polygon([(0, 0), (1, 1), (0, 1), (1, 0)])  # Self-intersecting
    >>> fixed = auto_fix_geometry(invalid, 0.0, 1e-6, verbose=False)
    >>> fixed.is_valid
    True
    """
    geom_type = geometry.geom_type

    # Strategy 1: Clean coordinates
    if verbose:
        print("Trying strategy: Clean coordinates")
    try:
        cleaned = clean_coordinates(geometry, tolerance)
        if cleaned.is_valid:
            if verbose:
                print("Fixed with coordinate cleaning")
            return cleaned
    except Exception as e:
        if verbose:
            print(f"   Failed: {e}")

    # Strategy 2: Buffer(0)
    if verbose:
        print("Trying strategy: Buffer(0)")
    try:
        buffered = fix_with_buffer(geometry, buffer_distance, verbose)
        if buffered.is_valid:
            if verbose:
                print("   Fixed with buffer")
            return buffered
    except Exception as e:
        if verbose:
            print(f"   Failed: {e}")

    # Strategy 3: Simplify
    if verbose:
        print("Trying strategy: Simplify")
    try:
        simplified = fix_with_simplify(geometry, tolerance, verbose)
        if simplified.is_valid:
            if verbose:
                print("   Fixed with simplification")
            return simplified
    except Exception as e:
        if verbose:
            print(f"   Failed: {e}")

    # Strategy 4: Reconstruct
    if verbose:
        print("Trying strategy: Reconstruct")
    try:
        reconstructed = fix_with_reconstruct(geometry, tolerance, verbose)
        if reconstructed.is_valid:
            if verbose:
                print("   Fixed with reconstruction")
            return reconstructed
    except Exception as e:
        if verbose:
            print(f"   Failed: {e}")

    # All strategies failed
    raise RepairError(
        f"Could not repair {geom_type}: {explain_validity(geometry)}"
    )


__all__ = ['auto_fix_geometry']
