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
from shapely.ops import unary_union

from ..core.cleanup import CleanupConfig, cleanup_polygon
from ..core.constraints import ConstraintStatus, GeometryConstraints, MergeConstraints
from ..core.errors import FixWarning
from ..core.geometry_utils import validate_and_fix
from ..core.types import OverlapStrategy
from ..clearance.fix_clearance import fix_clearance
from ..overlap import remove_overlaps
from ..merge import merge_close_polygons
from .transaction import FixTransaction
from .stages import (
    build_default_stages,
    execute_stage_pipeline,
    StageResult,
    _cleanup_geometry,
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
    prepared = _prepare_geometry(geometry)
    transaction = FixTransaction(prepared, prepared, constraints)
    stage_history = execute_stage_pipeline(
        transaction=transaction,
        constraints=constraints,
        merge_constraints=merge_constraints,
        stages=build_default_stages(merge_constraints),
        max_iterations=max_iterations,
        verbose=verbose,
    )

    best_geometry, best_status = transaction.get_best_result()
    best_geometry = _finalize_geometry(best_geometry, constraints)
    best_status = constraints.check(best_geometry, prepared)
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

    working_geometries = [_prepare_geometry(geom) for geom in geometries]
    working_originals = list(working_geometries)
    working_properties = _copy_properties(properties) if properties is not None else None

    if merge_constraints and merge_constraints.enabled:
        working_geometries, working_originals, working_properties = _apply_initial_merge(
            working_geometries,
            working_originals,
            working_properties,
            merge_constraints,
            verbose,
        )

    stage_template = build_default_stages(merge_constraints)

    fixed: List[BaseGeometry] = []
    statuses: List[ConstraintStatus] = []
    histories: List[List[str]] = []

    for geom, orig in zip(working_geometries, working_originals):
        transaction = FixTransaction(geom, orig, constraints)
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
            originals=working_originals,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    finalized_geometries: List[BaseGeometry] = []
    finalized_statuses: List[ConstraintStatus] = []
    for geom, original in zip(fixed, working_originals):
        finalized = _finalize_geometry(geom, constraints)
        finalized_geometries.append(finalized)
        finalized_statuses.append(constraints.check(finalized, original))

    fixed = finalized_geometries
    statuses = finalized_statuses

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

    return fixed, warnings_list, working_properties


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

        # Apply cleanup to each resolved geometry to remove degenerate features
        # that may have been created during overlap resolution
        cleaned_resolved = []
        for poly in resolved:
            try:
                cleaned = _cleanup_geometry(poly, constraints)
                cleaned_resolved.append(cleaned if cleaned is not None else poly)
            except Exception:
                # If cleanup fails, use the uncleaned geometry
                cleaned_resolved.append(poly)
        resolved = cleaned_resolved

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


def _prepare_geometry(geometry: BaseGeometry) -> BaseGeometry:
    if geometry is None:
        return geometry
    geom = geometry
    if isinstance(geom, (Polygon, MultiPolygon)):
        cfg = CleanupConfig(min_zero_area=1e-10, preserve_holes=True)
        geom = cleanup_polygon(geom, cfg)
    fixed = validate_and_fix(geom)
    return fixed if fixed is not None else geom


def _finalize_geometry(geometry: BaseGeometry, constraints: GeometryConstraints) -> BaseGeometry:
    geom = _cleanup_geometry(geometry, constraints)
    return _apply_min_clearance(geom, constraints.min_clearance)


def _apply_min_clearance(geometry: BaseGeometry, min_clearance: Optional[float]) -> BaseGeometry:
    if min_clearance is None or min_clearance <= 0:
        return geometry

    if isinstance(geometry, Polygon):
        try:
            return fix_clearance(geometry, min_clearance)
        except Exception:
            return geometry

    if isinstance(geometry, MultiPolygon):
        unioned = unary_union(geometry)
        if isinstance(unioned, Polygon):
            return _apply_min_clearance(unioned, min_clearance)
        if isinstance(unioned, MultiPolygon):
            parts = [_apply_min_clearance(part, min_clearance) for part in unioned.geoms]
            polys = [part for part in parts if isinstance(part, Polygon) and not part.is_empty]
            if polys:
                return MultiPolygon(polys)
    return geometry


def _apply_initial_merge(
    geometries: List[BaseGeometry],
    originals: List[BaseGeometry],
    properties: Optional[List[Dict[str, Any]]],
    merge_constraints: MergeConstraints,
    verbose: bool,
) -> Tuple[List[BaseGeometry], List[BaseGeometry], Optional[List[Dict[str, Any]]]]:
    try:
        polygons: List[Polygon] = []
        sources: List[int] = []
        for idx, geom in enumerate(geometries):
            if isinstance(geom, Polygon):
                polygons.append(geom)
                sources.append(idx)
            elif isinstance(geom, MultiPolygon) and len(geom.geoms) > 0:
                polygons.append(max(geom.geoms, key=lambda g: g.area))
                sources.append(idx)
        if not polygons:
            return geometries, originals, properties

        merged, mapping = merge_close_polygons(
            polygons,
            margin=merge_constraints.margin,
            merge_strategy=merge_constraints.merge_strategy,
            preserve_holes=merge_constraints.preserve_holes,
            insert_vertices=merge_constraints.insert_vertices,
            return_mapping=True,
        )

        new_originals: List[BaseGeometry] = []
        for group in mapping:
            source_indices = [sources[i] for i in group]
            group_originals = [originals[idx] for idx in source_indices]
            new_originals.append(unary_union(group_originals))

        new_properties = properties
        if properties is not None:
            aggregated: List[Dict[str, Any]] = []
            for group in mapping:
                source_idx = sources[group[0]]
                base = properties[source_idx].copy()
                base['merge_group'] = ','.join(str(sources[i]) for i in group)
                aggregated.append(base)
            new_properties = aggregated

        return merged, new_originals, new_properties
    except Exception as exc:
        if verbose:
            print(f"[Merge] Initial merge skipped: {exc}")
        return geometries, originals, properties


__all__ = [
    "robust_fix_geometry",
    "robust_fix_batch",
]
