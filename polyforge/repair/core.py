from typing import List, Optional, Tuple
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

from ..core.types import RepairStrategy

from .strategies.auto import auto_fix_geometry
from .strategies.buffer import fix_with_buffer
from .strategies.simplify import fix_with_simplify
from .strategies.reconstruct import fix_with_reconstruct
from .strategies.strict import fix_strict


def repair_geometry(
    geometry: BaseGeometry,
    repair_strategy: RepairStrategy = RepairStrategy.AUTO,
    buffer_distance: float = 0.0,
    tolerance: float = 1e-10,
    verbose: bool = False
) -> BaseGeometry:
    """Repair invalid geometries using various strategies.

    This function identifies different types of geometry invalidity and applies
    appropriate repair strategies. It can handle:
    - Self-intersections
    - Ring self-intersections
    - Duplicate vertices
    - Unclosed rings
    - Invalid coordinate sequences
    - Topology errors
    - Bow-tie polygons
    - Overlapping holes

    Args:
        geometry: The geometry to repair
        repair_strategy: Repair strategy to use:
            - RepairStrategy.AUTO: Automatically detect and repair (default)
            - RepairStrategy.BUFFER: Use buffer(0) trick
            - RepairStrategy.SIMPLIFY: Simplify and rebuild
            - RepairStrategy.RECONSTRUCT: Reconstruct from scratch
            - RepairStrategy.STRICT: Only repair if guaranteed to preserve intent
        buffer_distance: Small buffer distance for buffer-based repairs (default: 0.0)
        tolerance: Tolerance for coordinate comparisons (default: 1e-10)
        verbose: Print diagnostic information (default: False)

    Returns:
        Repaired geometry (same type as input if possible)

    Raises:
        RepairError: If geometry cannot be repaired

    Examples:
        >>> # Repair self-intersecting polygon
        >>> from polyforge.core.types import RepairStrategy
        >>> poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])  # Bow-tie
        >>> repaired = repair_geometry(poly)
        >>> repaired.is_valid
        True

        >>> # Using specific strategy
        >>> repaired = repair_geometry(poly, repair_strategy=RepairStrategy.BUFFER)

    Notes:
        - Multiple strategies are tried if AUTO is selected
        - Original geometry is returned if already valid
        - Some repairs may slightly modify geometry shape
    """
    # Quick check: if valid, return as-is
    if geometry.is_valid:
        if verbose:
            print("Geometry is already valid")
        return geometry

    if verbose:
        reason = explain_validity(geometry)
        print(f"Invalid geometry: {reason}")

    # Determine strategy
    if repair_strategy == RepairStrategy.AUTO:
        return auto_fix_geometry(geometry, buffer_distance, tolerance, verbose)
    elif repair_strategy == RepairStrategy.BUFFER:
        return fix_with_buffer(geometry, buffer_distance, verbose)
    elif repair_strategy == RepairStrategy.SIMPLIFY:
        return fix_with_simplify(geometry, tolerance, verbose)
    elif repair_strategy == RepairStrategy.RECONSTRUCT:
        return fix_with_reconstruct(geometry, tolerance, verbose)
    elif repair_strategy == RepairStrategy.STRICT:
        return fix_strict(geometry, tolerance, verbose)
    else:
        raise ValueError(f"Unknown repair_strategy: {repair_strategy}")


def batch_repair_geometries(
    geometries: List[BaseGeometry],
    repair_strategy: RepairStrategy = RepairStrategy.AUTO,
    on_error: str = 'keep',
    verbose: bool = False
) -> Tuple[List[Optional[BaseGeometry]], List[int]]:
    """Repair multiple geometries in batch.

    The returned list always has the same length as the input list, so
    results can be mapped back to their original indices.

    Args:
        geometries: List of geometries to repair
        repair_strategy: Repair strategy (see repair_geometry)
        on_error: What to do on error:
            - 'keep': Keep original invalid geometry (default)
            - 'skip': Replace failed geometry with None
            - 'raise': Raise exception
        verbose: Print progress information

    Returns:
        Tuple of (repaired_geometries, failed_indices).  The repaired list
        always has the same length as *geometries*.  When *on_error='skip'*,
        failed entries are set to ``None``.

    Examples:
        >>> from polyforge.core.types import RepairStrategy
        >>> geometries = [poly1, poly2, poly3]
        >>> repaired, failed = batch_repair_geometries(geometries)
        >>> print(f"Repaired {len(repaired)}, failed {len(failed)}")

        >>> # Using specific strategy
        >>> repaired, failed = batch_repair_geometries(geometries, RepairStrategy.BUFFER)
    """
    repaired: List[Optional[BaseGeometry]] = []
    failed_indices: List[int] = []

    for i, geom in enumerate(geometries):
        try:
            if verbose and i % 100 == 0:
                print(f"Processing geometry {i}/{len(geometries)}...")

            repaired_geom = repair_geometry(geom, repair_strategy=repair_strategy, verbose=False)
            repaired.append(repaired_geom)

        except Exception as e:
            if on_error == 'raise':
                raise
            elif on_error == 'keep':
                repaired.append(geom)
            else:  # skip
                repaired.append(None)
                failed_indices.append(i)

            if verbose:
                print(f"  Failed to repair geometry {i}: {e}")

    return repaired, failed_indices


__all__ = ['repair_geometry', 'batch_repair_geometries']
