from __future__ import annotations

from shapely import make_valid
from shapely.errors import GEOSException, TopologicalError
from shapely.geometry.base import BaseGeometry

from polyforge.core.errors import PolyforgeError, RepairError


def repair_geometry(
    geometry: BaseGeometry,
    method: str = "linework",
) -> BaseGeometry:
    """Repair an invalid geometry using Shapely's make_valid.

    Args:
        geometry: Shapely geometry to repair.
        method: Repair method passed to make_valid (default: "linework").

    Returns:
        Repaired geometry.
    """
    return make_valid(geometry, method=method)


def batch_repair_geometries(
    geometries: list[BaseGeometry],
    on_error: str = "keep",
    verbose: bool = False,
) -> tuple[list[BaseGeometry | None], list[int]]:
    """Repair multiple geometries in batch.

    Args:
        geometries: List of Shapely geometries to repair.
        on_error: How to handle failures - "keep" (keep original), "skip" (omit),
            or "raise" (raise exception).
        verbose: If True, print progress information.

    Returns:
        Tuple of (repaired_geometries, failed_indices).
    """
    repaired: list[BaseGeometry | None] = []
    failed_indices: list[int] = []

    for i, geom in enumerate(geometries):
        try:
            repaired_geom = repair_geometry(geom)
            repaired.append(repaired_geom)

        except (GEOSException, TopologicalError, ValueError, PolyforgeError) as e:
            if on_error == "raise":
                raise RepairError(f"Failed to repair geometry at index {i}: {e}")
            elif on_error == "skip":
                failed_indices.append(i)
                continue
            elif on_error == "keep":
                failed_indices.append(i)
                repaired.append(geom)
            else:
                repaired.append(None)

    return repaired, failed_indices


__all__ = ["repair_geometry", "batch_repair_geometries"]
