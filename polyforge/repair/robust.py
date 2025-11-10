"""
Robust constraint-aware geometry fixing orchestrated via composable stages.

The previous implementation embedded the entire fixing pipeline inside two
monolithic functions. This module now uses the stage helpers defined in
``polyforge.repair.stages`` so that each responsibility (validity repair,
clearance improvement, component merging, cleanup, etc.) is isolated, testable,
and can be extended without editing a thousand-line function.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import warnings

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from ..core.constraints import ConstraintStatus, GeometryConstraints, MergeConstraints
from ..core.errors import FixWarning
from ..core.types import OverlapStrategy
from ..overlap import remove_overlaps
from .transaction import FixTransaction
from .stages import (
    build_default_stages,
    execute_stage_pipeline,
    StageResult,
)


def robust_fix_geometry(
    geometry: BaseGeometry,
    constraints: GeometryConstraints,
    max_iterations: int = 20,
    raise_on_failure: bool = False,
    merge_constraints: Optional[MergeConstraints] = None,
    verbose: bool = False,
) -> Tuple[BaseGeometry, Optional[FixWarning]]:
    """
    Fix a single geometry while enforcing the supplied constraints.

    The geometry flows through a sequence of stages (validity repair, clearance
    improvement, optional component merging, cleanup). Each stage either commits
    its change through :class:`FixTransaction` or rolls back if constraints
    regress. The best result discovered during the pipeline is always returned.
    """
    transaction = FixTransaction(geometry, geometry, constraints)
    stage_history = execute_stage_pipeline(
        transaction=transaction,
        constraints=constraints,
        merge_constraints=merge_constraints,
        stages=build_default_stages(merge_constraints),
        max_iterations=max_iterations,
        verbose=verbose,
    )

    best_geometry, best_status = transaction.get_best_result()
    history = _stage_history_summary(stage_history) + transaction.get_history_summary()

    if best_status.all_satisfied():
        return best_geometry, None

    warning = _build_warning(best_geometry, best_status, history)
    if raise_on_failure:
        raise warning

    warnings.warn(str(warning), UserWarning, stacklevel=2)
    return best_geometry, warning


def robust_fix_batch(
    geometries: List[BaseGeometry],
    constraints: GeometryConstraints,
    max_iterations: int = 20,
    handle_overlaps: bool = True,
    merge_constraints: Optional[MergeConstraints] = None,
    properties: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False,
) -> Tuple[List[BaseGeometry], List[Optional[FixWarning]], Optional[List[Dict[str, Any]]]]:
    """
    Apply :func:`robust_fix_geometry` to multiple geometries and optionally
    resolve overlaps between the fixed results.
    """
    if not geometries:
        return [], [], None if properties is None else []

    stage_template = build_default_stages(merge_constraints)

    fixed: List[BaseGeometry] = []
    statuses: List[ConstraintStatus] = []
    histories: List[List[str]] = []

    for geom in geometries:
        transaction = FixTransaction(geom, geom, constraints)
        stage_history = execute_stage_pipeline(
            transaction=transaction,
            constraints=constraints,
            merge_constraints=merge_constraints,
            stages=stage_template,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        best_geom, best_status = transaction.get_best_result()
        history = _stage_history_summary(stage_history) + transaction.get_history_summary()

        fixed.append(best_geom)
        statuses.append(best_status)
        histories.append(history)

    overlap_notes: List[str] = [""] * len(fixed)
    if handle_overlaps and constraints.max_overlap_area == 0.0:
        fixed, statuses, overlap_notes = _resolve_batch_overlaps(
            fixed,
            statuses,
            constraints,
            originals=geometries,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    warnings_list: List[Optional[FixWarning]] = []
    for idx, (geom, status, history) in enumerate(zip(fixed, statuses, histories)):
        note = overlap_notes[idx]
        if note:
            history = history + [note]
            histories[idx] = history

        if status.all_satisfied():
            warnings_list.append(None)
            continue

        warning = _build_warning(geom, status, history)
        warnings_list.append(warning)
        warnings.warn(str(warning), UserWarning, stacklevel=2)

    return fixed, warnings_list, _copy_properties(properties)


def _copy_properties(
    props: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    if props is None:
        return None
    return [p.copy() if p else {} for p in props]


def _build_warning(
    geometry: BaseGeometry,
    status: ConstraintStatus,
    history: List[str],
) -> FixWarning:
    unmet = [v.constraint_type.name for v in status.violations]
    message = (
        f"Could not satisfy constraints ({', '.join(unmet)}) "
        f"after stage pipeline. Returning best-effort geometry."
    )
    return FixWarning(
        message=message,
        geometry=geometry,
        status=status,
        unmet_constraints=unmet,
        history=history,
    )


def _stage_history_summary(stage_history: List[StageResult]) -> List[str]:
    return [result.summary() for result in stage_history]


def _resolve_batch_overlaps(
    geometries: List[BaseGeometry],
    statuses: List[ConstraintStatus],
    constraints: GeometryConstraints,
    originals: List[BaseGeometry],
    max_iterations: int,
    verbose: bool,
) -> Tuple[List[BaseGeometry], List[ConstraintStatus], List[str]]:
    notes = [""] * len(geometries)

    try:
        polygon_indices = []
        polygons: List[Polygon] = []

        for idx, geom in enumerate(geometries):
            if isinstance(geom, Polygon):
                polygons.append(geom)
                polygon_indices.append(idx)
            elif isinstance(geom, MultiPolygon) and len(geom.geoms) > 0:
                largest = max(geom.geoms, key=lambda g: g.area)
                polygons.append(largest)
                polygon_indices.append(idx)
            else:
                if verbose:
                    print(f"[Overlap] Geometry {idx} is not a polygon; skipping overlap handling")
                return geometries, statuses, notes

        resolved = remove_overlaps(
            polygons,
            overlap_strategy=OverlapStrategy.SPLIT,
            max_iterations=max_iterations,
        )
    except Exception as exc:
        if verbose:
            print(f"[Overlap] Resolution failed: {exc}")
        return geometries, statuses, notes

    for poly, idx in zip(resolved, polygon_indices):
        status = constraints.check(poly, originals[idx])
        if status.is_better_or_equal(statuses[idx]):
            geometries[idx] = poly
            statuses[idx] = status
            notes[idx] = "overlap_resolution: applied"
        else:
            notes[idx] = "overlap_resolution: rolled back"

    return geometries, statuses, notes


__all__ = [
    "robust_fix_geometry",
    "robust_fix_batch",
]
