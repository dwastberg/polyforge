"""
Robust constraint-aware geometry fixing orchestrated via the lightweight pipeline.

This module translates the legacy stage/transaction system into a much simpler
loop: a handful of deterministic steps run in order, each step keeps its result
only if it improves the current constraint status, and the pipeline exits once
the constraints are satisfied or progress stalls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import warnings

from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..core.cleanup import CleanupConfig, cleanup_polygon
from ..core.constraints import ConstraintStatus, GeometryConstraints, MergeConstraints
from ..core.errors import FixWarning
from ..core.geometry_utils import safe_buffer_fix, validate_and_fix
from ..core.types import OverlapStrategy, RepairStrategy
from ..metrics import overlap_area_by_geometry, total_overlap_area
from ..clearance.fix_clearance import fix_clearance
from ..merge import merge_close_polygons
from ..overlap import remove_overlaps
from ..pipeline import (
    FixConfig,
    PipelineContext,
    PipelineStep,
    StepResult,
    config_from_constraints,
    run_steps,
)
from .core import repair_geometry


def robust_fix_geometry(
    geometry: BaseGeometry,
    constraints: GeometryConstraints,
    max_iterations: int = 20,
    raise_on_failure: bool = False,
    merge_constraints: Optional[MergeConstraints] = None,
    verbose: bool = False,
) -> Tuple[BaseGeometry, Optional[FixWarning]]:
    """Fix a single geometry using the lightweight pipeline."""
    prepared = _prepare_geometry(geometry)
    config = config_from_constraints(constraints)
    context = PipelineContext(
        original=prepared,
        constraints=constraints,
        config=config,
        merge_constraints=merge_constraints,
    )
    steps = _build_pipeline_steps(config, merge_constraints)

    fixed, status, history = run_steps(
        prepared,
        steps,
        context,
        max_passes=max_iterations,
    )

    finalized = _finalize_geometry(fixed, constraints)
    final_status = constraints.check(finalized, prepared, overlap_area=0.0)
    history_strings = _history_strings(history)

    if final_status.all_satisfied():
        return finalized, None

    warning = _build_warning(finalized, final_status, history_strings)
    if raise_on_failure:
        raise warning

    warnings.warn(str(warning), UserWarning, stacklevel=2)
    return finalized, warning


def robust_fix_batch(
    geometries: List[BaseGeometry],
    constraints: GeometryConstraints,
    max_iterations: int = 20,
    handle_overlaps: bool = True,
    merge_constraints: Optional[MergeConstraints] = None,
    properties: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False,
) -> Tuple[List[BaseGeometry], List[Optional[FixWarning]], Optional[List[Dict[str, Any]]]]:
    """Apply :func:`robust_fix_geometry` to multiple geometries."""
    if not geometries:
        return [], [], None if properties is None else []

    (
        working_geometries,
        working_originals,
        working_properties,
    ) = _prepare_batch_inputs(
        geometries,
        merge_constraints,
        properties,
        verbose,
    )

    fixed, statuses, histories = _run_batch_steps(
        working_geometries,
        working_originals,
        constraints,
        merge_constraints,
        max_iterations,
    )

    fixed, statuses, overlap_notes = _enforce_overlap_limits(
        fixed,
        statuses,
        constraints,
        working_originals,
        max_iterations,
        handle_overlaps,
        verbose,
    )

    finalized_geometries, finalized_statuses = _finalize_batch_results(
        fixed,
        working_originals,
        constraints,
    )

    warnings_list = _build_batch_warnings(
        finalized_geometries,
        finalized_statuses,
        histories,
        overlap_notes,
    )

    return finalized_geometries, warnings_list, working_properties


# ---------------------------------------------------------------------------
# Pipeline step construction
# ---------------------------------------------------------------------------

def _prepare_batch_inputs(
    geometries: List[BaseGeometry],
    merge_constraints: Optional[MergeConstraints],
    properties: Optional[List[Dict[str, Any]]],
    verbose: bool,
) -> Tuple[List[BaseGeometry], List[BaseGeometry], Optional[List[Dict[str, Any]]]]:
    working_geometries = [_prepare_geometry(geom) for geom in geometries]
    working_originals = list(working_geometries)
    working_properties = _copy_properties(properties) if properties is not None else None

    if merge_constraints and merge_constraints.enabled:
        (
            working_geometries,
            working_originals,
            working_properties,
        ) = _apply_initial_merge(
            working_geometries,
            working_originals,
            working_properties,
            merge_constraints,
            verbose,
        )

    return working_geometries, working_originals, working_properties


def _run_batch_steps(
    geometries: List[BaseGeometry],
    originals: List[BaseGeometry],
    constraints: GeometryConstraints,
    merge_constraints: Optional[MergeConstraints],
    max_iterations: int,
) -> Tuple[List[BaseGeometry], List[ConstraintStatus], List[List[str]]]:
    config = config_from_constraints(constraints)
    steps = _build_pipeline_steps(config, merge_constraints)

    fixed: List[BaseGeometry] = []
    statuses: List[ConstraintStatus] = []
    histories: List[List[str]] = []

    for geom, orig in zip(geometries, originals):
        context = PipelineContext(
            original=orig,
            constraints=constraints,
            config=config,
            merge_constraints=merge_constraints,
        )
        result_geom, status, history = run_steps(
            geom,
            steps,
            context,
            max_passes=max_iterations,
        )
        fixed.append(result_geom)
        statuses.append(status)
        histories.append(_history_strings(history))

    return fixed, statuses, histories


def _enforce_overlap_limits(
    geometries: List[BaseGeometry],
    statuses: List[ConstraintStatus],
    constraints: GeometryConstraints,
    originals: List[BaseGeometry],
    max_iterations: int,
    handle_overlaps: bool,
    verbose: bool,
) -> Tuple[List[BaseGeometry], List[ConstraintStatus], List[str]]:
    overlap_notes: List[str] = [""] * len(geometries)
    if not handle_overlaps:
        return geometries, statuses, overlap_notes

    max_overlap_allowed = constraints.max_overlap_area
    if max_overlap_allowed is None or math.isinf(max_overlap_allowed):
        return geometries, statuses, overlap_notes

    current_overlap = total_overlap_area(geometries)
    if current_overlap <= max_overlap_allowed + 1e-9:
        return geometries, statuses, overlap_notes

    return _resolve_batch_overlaps(
        geometries,
        statuses,
        constraints,
        originals=originals,
        max_iterations=max_iterations,
        verbose=verbose,
    )


def _finalize_batch_results(
    geometries: List[BaseGeometry],
    originals: List[BaseGeometry],
    constraints: GeometryConstraints,
) -> Tuple[List[BaseGeometry], List[ConstraintStatus]]:
    finalized_geometries: List[BaseGeometry] = []
    for geom in geometries:
        finalized = _finalize_geometry(geom, constraints)
        finalized_geometries.append(finalized)

    overlap_by_geometry = overlap_area_by_geometry(finalized_geometries)
    finalized_statuses: List[ConstraintStatus] = []
    for idx, (geom, original) in enumerate(zip(finalized_geometries, originals)):
        overlap_value = overlap_by_geometry[idx] if idx < len(overlap_by_geometry) else 0.0
        finalized_statuses.append(constraints.check(geom, original, overlap_area=overlap_value))

    return finalized_geometries, finalized_statuses


def _build_batch_warnings(
    geometries: List[BaseGeometry],
    statuses: List[ConstraintStatus],
    histories: List[List[str]],
    overlap_notes: List[str],
) -> List[Optional[FixWarning]]:
    warnings_list: List[Optional[FixWarning]] = []

    for idx, (geom, status, history) in enumerate(zip(geometries, statuses, histories)):
        note = overlap_notes[idx] if idx < len(overlap_notes) else ""
        history_with_notes = history + ([note] if note else [])

        if status.all_satisfied():
            warnings_list.append(None)
            continue

        warning = _build_warning(geom, status, history_with_notes)
        warnings_list.append(warning)
        warnings.warn(str(warning), UserWarning, stacklevel=2)

    return warnings_list


def _build_pipeline_steps(
    config: FixConfig,
    merge_constraints: Optional[MergeConstraints],
) -> List[PipelineStep]:
    steps: List[PipelineStep] = [
        _validity_step,
        _clearance_step,
    ]

    if merge_constraints and merge_constraints.enabled:
        steps.append(_merge_components_step)

    if config.cleanup:
        steps.append(_cleanup_step)

    return steps


def _validity_step(geometry: BaseGeometry, ctx: PipelineContext) -> StepResult:
    if not ctx.config.must_be_valid or geometry.is_valid:
        return StepResult("validity", geometry, False, "already valid")

    repaired = repair_geometry(geometry, repair_strategy=RepairStrategy.AUTO)
    return _maybe_accept("validity", geometry, repaired, ctx, "repaired invalid geometry")


def _clearance_step(geometry: BaseGeometry, ctx: PipelineContext) -> StepResult:
    target = ctx.config.min_clearance
    if target is None or target <= 0:
        return StepResult("clearance", geometry, False, "no target")

    current_clearance = _safe_clearance(geometry)
    if current_clearance is not None and current_clearance + 1e-9 >= target:
        return StepResult("clearance", geometry, False, "meets target")

    improved = _apply_clearance_fix(geometry, target)
    if improved is geometry:
        return StepResult("clearance", geometry, False, "no-op")

    return _maybe_accept("clearance", geometry, improved, ctx, "clearance improved")


def _merge_components_step(geometry: BaseGeometry, ctx: PipelineContext) -> StepResult:
    config = ctx.merge_constraints
    if not config or not config.enabled:
        return StepResult("merge_components", geometry, False, "disabled")

    if isinstance(geometry, Polygon):
        polygons = [geometry]
    elif isinstance(geometry, MultiPolygon):
        polygons = list(geometry.geoms)
    else:
        return StepResult("merge_components", geometry, False, "not a polygon")

    merged = merge_close_polygons(
        polygons,
        margin=config.margin,
        merge_strategy=config.merge_strategy,
        preserve_holes=config.preserve_holes,
        insert_vertices=config.insert_vertices,
    )

    if not merged:
        return StepResult("merge_components", geometry, False, "no merge result")

    if len(merged) == 1:
        candidate: BaseGeometry = merged[0]
    else:
        candidate = MultiPolygon(merged)

    return _maybe_accept("merge_components", geometry, candidate, ctx, "components merged")


def _cleanup_step(geometry: BaseGeometry, ctx: PipelineContext) -> StepResult:
    cleaned = _cleanup_geometry(geometry, ctx.constraints)
    if cleaned is None or cleaned.equals(geometry):
        return StepResult("cleanup", geometry, False, "cleanup not needed")
    return _maybe_accept("cleanup", geometry, cleaned, ctx, "cleanup applied")


def _maybe_accept(
    name: str,
    current: BaseGeometry,
    candidate: BaseGeometry,
    ctx: PipelineContext,
    success_message: str,
) -> StepResult:
    """Accept candidate if it satisfies constraints better than current.

    Performance optimizations:
    - Early exit if geometries are equal (no point comparing)
    - Uses cached metrics from PipelineContext
    - Only calculates clearance if min_clearance constraint exists
    """
    # Early exit: if geometries are equal, no change occurred
    if candidate.equals(current):
        return StepResult(name, current, False, "no change")

    # Get cached metrics (automatically handles clearance caching)
    current_metrics = ctx.get_metrics(current)
    candidate_metrics = ctx.get_metrics(candidate)

    # Check constraints with pre-computed metrics
    current_status = ctx.constraints.check(current, ctx.original, metrics=current_metrics)
    candidate_status = ctx.constraints.check(candidate, ctx.original, metrics=candidate_metrics)

    if candidate_status.is_better_or_equal(current_status):
        return StepResult(name, candidate, True, success_message)

    return StepResult(name, current, False, "candidate rejected")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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
                polygons.append(max(geom.geoms, key=lambda g: g.area))
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

        cleaned_resolved = []
        for poly in resolved:
            try:
                cleaned = _cleanup_geometry(poly, constraints)
                cleaned_resolved.append(cleaned if cleaned is not None else poly)
            except Exception:
                cleaned_resolved.append(poly)
        resolved = cleaned_resolved

    except Exception as exc:
        if verbose:
            print(f"[Overlap] Resolution failed: {exc}")
        return geometries, statuses, notes

    for poly, idx in zip(resolved, polygon_indices):
        status = constraints.check(poly, originals[idx], overlap_area=0.0)
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
    geom = _apply_min_clearance(geometry, constraints.min_clearance)
    geom = _cleanup_geometry(geom, constraints)

    if not constraints.allow_multipolygon and isinstance(geom, MultiPolygon):
        candidates = [
            poly for poly in geom.geoms
            if isinstance(poly, Polygon) and not poly.is_empty
        ]
        if candidates:
            return max(candidates, key=lambda g: g.area)
        return Polygon()
    return geom


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
                base["merge_group"] = ",".join(str(sources[i]) for i in group)
                aggregated.append(base)
            new_properties = aggregated

        return merged, new_originals, new_properties
    except Exception as exc:
        if verbose:
            print(f"[Merge] Initial merge skipped: {exc}")
        return geometries, originals, properties


def _cleanup_geometry(
    geometry: BaseGeometry,
    constraints: GeometryConstraints,
) -> BaseGeometry:
    """Apply constraint-aware cleanup heuristics to polygonal geometries."""
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry

    cleaned = _apply_hole_constraints(geometry, constraints)
    cleaned = _smooth_low_clearance(cleaned, constraints.min_clearance)
    cleaned = _heal_geometry(cleaned)
    return cleaned


def _apply_clearance_fix(geometry: BaseGeometry, min_clearance: Optional[float]) -> BaseGeometry:
    if min_clearance is None or min_clearance <= 0:
        return geometry

    if isinstance(geometry, Polygon):
        return fix_clearance(geometry, min_clearance)

    if isinstance(geometry, MultiPolygon):
        united = unary_union(geometry)
        if isinstance(united, Polygon):
            return fix_clearance(united, min_clearance)
        if isinstance(united, MultiPolygon):
            fixed_parts = [fix_clearance(poly, min_clearance) for poly in united.geoms]
            valid_parts = [poly for poly in fixed_parts if isinstance(poly, Polygon) and not poly.is_empty]
            if not valid_parts:
                return geometry
            return MultiPolygon(valid_parts)
        return geometry

    return geometry


def _safe_clearance(geometry: BaseGeometry) -> Optional[float]:
    try:
        return geometry.minimum_clearance
    except Exception:
        return None


def _copy_properties(
    props: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    if props is None:
        return None
    return [p.copy() if p else {} for p in props]


def _apply_hole_constraints(
    geometry: BaseGeometry,
    constraints: GeometryConstraints,
) -> BaseGeometry:
    """Remove degenerate holes/slivers based on constraint thresholds."""
    config = CleanupConfig(
        min_zero_area=1e-10,
        hole_area_threshold=(
            constraints.min_hole_area
            if constraints.min_hole_area and constraints.min_hole_area > 0
            else None
        ),
        hole_aspect_ratio=constraints.max_hole_aspect_ratio,
        hole_min_width=constraints.min_hole_width,
        preserve_holes=True,
    )
    return cleanup_polygon(geometry, config)


def _smooth_low_clearance(
    geometry: BaseGeometry,
    min_clearance: Optional[float],
) -> BaseGeometry:
    """
    Remove thin features that violate minimum clearance using scaled buffer operations.

    Uses negative buffer (erosion) followed by positive buffer (dilation) to remove
    thin slivers and protrusions that are narrower than the target clearance.

    The buffer distance is scaled to min_clearance * 0.5, which removes features
    narrower than half the target clearance while preserving the overall shape.

    Args:
        geometry: Input polygon or multipolygon
        min_clearance: Target minimum clearance (None or <= 0 returns geometry unchanged)

    Returns:
        Cleaned geometry with thin features removed, or original if operation fails
    """
    if (
        min_clearance is None
        or min_clearance <= 0
        or not isinstance(geometry, (Polygon, MultiPolygon))
        or geometry.is_empty
    ):
        return geometry

    clearance = _safe_clearance(geometry)
    if clearance is None or clearance >= min_clearance:
        return geometry

    # Scale buffer distance to half the target clearance
    # This removes features narrower than half the target
    buffer_dist = min_clearance * 0.5

    # Calculate expected area loss based on buffer distance
    # Approximate as area of features with width < buffer_dist
    # Use a generous tolerance: allow losing ~10% or area proportional to buffer
    original_area = getattr(geometry, "area", 0.0) or 0.0
    max_area_loss = min(0.1, (buffer_dist * 2) ** 2 * 3.14159 / max(original_area, 1.0))

    try:
        eroded = geometry.buffer(-buffer_dist, join_style=2)
    except GEOSException:
        return geometry

    if not isinstance(eroded, (Polygon, MultiPolygon)) or eroded.is_empty or not eroded.is_valid:
        return geometry

    try:
        dilated = eroded.buffer(buffer_dist, join_style=2)
    except GEOSException:
        return geometry

    if not isinstance(dilated, (Polygon, MultiPolygon)) or dilated.is_empty or not dilated.is_valid:
        return geometry

    if original_area > 0.0:
        cleaned_area = getattr(dilated, "area", 0.0) or 0.0
        area_loss = max(0.0, (original_area - cleaned_area) / original_area)
        if area_loss > max_area_loss:
            return geometry

    return dilated


def _heal_geometry(geometry: BaseGeometry) -> BaseGeometry:
    """Attempt to fix minor topological issues via buffer(0) while preserving validity."""
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    healed = safe_buffer_fix(geometry, distance=0.0, return_largest=False)
    return healed if healed is not None else geometry


def _build_warning(
    geometry: BaseGeometry,
    status: ConstraintStatus,
    history: List[str],
) -> FixWarning:
    unmet = [v.constraint_type.name for v in status.violations]
    message = (
        f"Could not satisfy constraints ({', '.join(unmet)}) "
        f"after pipeline. Returning best-effort geometry."
    )
    return FixWarning(
        message=message,
        geometry=geometry,
        status=status,
        unmet_constraints=unmet,
        history=history,
    )


def _history_strings(results: List[StepResult]) -> List[str]:
    history: List[str] = []
    for result in results:
        status = "changed" if result.changed else "skipped"
        if result.message:
            history.append(f"{result.name}: {result.message}")
        else:
            history.append(f"{result.name}: {status}")
    return history


__all__ = [
    "robust_fix_geometry",
    "robust_fix_batch",
]
