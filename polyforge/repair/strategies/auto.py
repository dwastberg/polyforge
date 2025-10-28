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
    """Automatically detect and fix geometry issues.

    Tries multiple strategies in order of preference:
    1. Clean coordinates (remove duplicates, close rings)
    2. Buffer(0) trick
    3. Simplification
    4. Reconstruction
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
