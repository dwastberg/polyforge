from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import shapely
import numpy as np
from shapely.geometry import Polygon
from shapely.errors import GEOSException

from polyforge.ops.clearance import (
    fix_hole_too_close,
    fix_narrow_protrusion,
    remove_narrow_protrusions,
    remove_narrow_wedges,
    fix_narrow_passage,
    fix_near_self_intersection,
    fix_parallel_close_edges,
)
from polyforge.ops.clearance.passages import _erode_dilate_fix
from polyforge.ops.clearance.hole_rings import try_hole_ring_fix
from polyforge.ops.clearance.utils import _find_nearest_vertex_index, _find_nearest_edge_index
from polyforge.ops.clearance.courtyard import try_close_passage_to_hole
from polyforge.core.types import HoleStrategy, PassageStrategy, IntersectionStrategy
from polyforge.core.iterative_utils import iterative_improve
from polyforge.metrics import _safe_clearance

from polyforge.clearance._helpers import (
    clearance_or_zero,
    is_usable,
    normalize_polygon,
)
from polyforge.clearance._diagnosis import (
    ClearanceIssue,
    ClearanceContext,
    ClearanceDiagnosis,
    _ACUTE_TIP_ANGLE,
    diagnose_clearance,
    _build_clearance_context,
    RECOMMENDED_FIXES,
)

# Maximum allowed area growth factor. Clearance fixes should not significantly
# increase polygon area (e.g., by filling in concavities).  Set to 2% to allow
# minor growth from removing concave vertices at clearance bottlenecks (e.g.,
# vertex-to-edge issues where a single concave vertex is removed, slightly
# expanding the exterior).
_AREA_GROWTH_TOLERANCE = 1.05


@dataclass
class ClearanceFixSummary:
    """Metadata describing the fix process."""

    initial_clearance: float
    final_clearance: float
    area_ratio: float
    iterations: int
    issue: ClearanceIssue
    fixed: bool
    valid: bool
    history: list[ClearanceIssue]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def fix_clearance(
    geometry: Polygon,
    min_clearance: float,
    max_iterations: int = 10,
    min_area_ratio: float = 0.9,
    return_diagnosis: bool = False,
) -> Polygon | tuple[Polygon, ClearanceFixSummary]:
    """Automatically diagnose and fix low minimum clearance in a polygon.

    Args:
        geometry: Input polygon to fix.
        min_clearance: Target minimum clearance value.
        max_iterations: Maximum number of fix passes (default: 10).
        min_area_ratio: Minimum fraction of original area to retain (default: 0.9).
        return_diagnosis: If True, return (polygon, ClearanceFixSummary).

    Returns:
        Fixed polygon, or (polygon, ClearanceFixSummary) if return_diagnosis=True.

    Raises:
        TypeError: If geometry is not a Polygon.
        ValueError: If min_area_ratio is not in (0, 1].
    """
    if not isinstance(geometry, Polygon):
        raise TypeError(f"Expected Polygon, got {type(geometry).__name__}")

    if min_area_ratio <= 0 or min_area_ratio > 1.0:
        raise ValueError("min_area_ratio must be in (0, 1].")

    initial_clearance = geometry.minimum_clearance
    original_area = geometry.area

    if initial_clearance >= min_clearance:
        summary = ClearanceFixSummary(
            initial_clearance=initial_clearance,
            final_clearance=initial_clearance,
            area_ratio=1.0,
            iterations=0,
            issue=ClearanceIssue.NONE,
            fixed=True,
            valid=geometry.is_valid,
            history=[ClearanceIssue.NONE],
        )
        return (geometry, summary) if return_diagnosis else geometry

    # --- Phase 0: Deduplicate near-duplicate vertices ---
    geometry, dedup_done = _phase_dedup(geometry, min_clearance)
    if dedup_done:
        summary = _build_summary(initial_clearance, geometry, original_area, [ClearanceIssue.NONE], min_clearance)
        return (geometry, summary) if return_diagnosis else geometry

    original_exterior_area = Polygon(geometry.exterior).area
    issue_history: list[ClearanceIssue] = []

    # --- Phase 1: Region fix (erosion-dilation) ---
    region_result = _phase_region(geometry, min_clearance, min_area_ratio)
    if region_result is not None:
        issue_history.append(ClearanceIssue.PARALLEL_CLOSE_EDGES)
        summary = _build_summary(initial_clearance, region_result, original_area, issue_history, min_clearance)
        return (region_result, summary) if return_diagnosis else region_result

    # --- Phase 1.5: Courtyard passage detection ---
    geometry, courtyard_issues = _phase_courtyard(geometry, min_clearance, initial_clearance)
    issue_history.extend(courtyard_issues)
    if courtyard_issues:
        # Courtyard may have fully fixed the clearance
        courtyard_clearance = clearance_or_zero(geometry)
        if courtyard_clearance >= min_clearance:
            summary = _build_summary(initial_clearance, geometry, original_area, issue_history, min_clearance)
            return (geometry, summary) if return_diagnosis else geometry
        original_exterior_area = Polygon(geometry.exterior).area

    # --- Phase 2: Point-based fixes (diagnosis-driven iteration) ---
    improved = _phase_iterative(
        geometry, min_clearance, original_area, original_exterior_area,
        min_area_ratio, max_iterations, issue_history,
    )

    # --- Phase 3: Cleanup pass (erosion-dilation on improved geometry) ---
    improved = _phase_cleanup(improved, min_clearance, min_area_ratio, issue_history)

    summary = _build_summary(initial_clearance, improved, original_area, issue_history, min_clearance)
    return (improved, summary) if return_diagnosis else improved


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------


def _phase_dedup(geometry: Polygon, min_clearance: float) -> tuple[Polygon, bool]:
    """Phase 0: Remove near-duplicate vertices.

    Returns (geometry, True) if dedup fully resolved clearance, else
    (possibly-deduped geometry, False).
    """
    dedup_tolerance = min_clearance * 0.1
    dedup_candidate = _dedup_ring_vertices(geometry, dedup_tolerance)
    if not is_usable(dedup_candidate):
        return geometry, False
    dedup_clearance = clearance_or_zero(dedup_candidate)
    if dedup_clearance >= min_clearance:
        return dedup_candidate, True
    # Continue with deduplicated geometry even if clearance not yet met
    return dedup_candidate, False


def _phase_region(
    geometry: Polygon, min_clearance: float, min_area_ratio: float
) -> Polygon | None:
    """Phase 1: Erosion-dilation for broad narrow features.

    Returns fixed polygon if clearance met, else None.
    """
    region_candidate = _erode_dilate_fix(geometry, min_clearance, min_area_ratio)
    if region_candidate is None:
        return None
    region_clearance = clearance_or_zero(region_candidate)
    if region_clearance >= min_clearance:
        return region_candidate
    return None


def _phase_courtyard(
    geometry: Polygon, min_clearance: float, initial_clearance: float
) -> tuple[Polygon, list[ClearanceIssue]]:
    """Phase 1.5: Close narrow passages to courtyard holes.

    Returns (possibly-updated geometry, list of issues found).
    """
    courtyard_candidate = try_close_passage_to_hole(geometry, min_clearance)
    if courtyard_candidate is None or not courtyard_candidate.is_valid:
        return geometry, []
    courtyard_clearance = clearance_or_zero(courtyard_candidate)
    if courtyard_clearance > initial_clearance:
        return courtyard_candidate, [ClearanceIssue.NEAR_SELF_INTERSECTION]
    return geometry, []


def _is_same_hole_issue(context: ClearanceContext | None) -> bool:
    """Check if the clearance bottleneck is a same-hole self-clearance issue."""
    if context is None:
        return False
    return (
        context.ring_types == ("hole", "hole")
        and context.hole_indices[0] is not None
        and context.hole_indices[0] == context.hole_indices[1]
    )


def _is_inter_hole_pinch_issue(context: ClearanceContext | None) -> bool:
    """Check if the clearance bottleneck is between vertices on two different holes."""
    if context is None:
        return False
    return (
        context.ring_types == ("hole", "hole")
        and context.hole_indices[0] is not None
        and context.hole_indices[1] is not None
        and context.hole_indices[0] != context.hole_indices[1]
    )


def _move_hole_pinch_vertices(
    geometry: Polygon,
    min_clearance: float,
    context: ClearanceContext,
    max_passes: int = 5,
) -> Polygon | None:
    """Move near-coincident vertex pairs on a hole ring apart along clearance lines.

    For same-hole clearance issues where two non-consecutive vertices are
    nearly coincident (pinch points), moves each vertex away from the other
    by half the required clearance along the clearance line direction.

    Iterates to fix multiple pinch points on the same hole ring.
    """
    hole_index = context.hole_indices[0]
    if hole_index is None or hole_index < 0:
        return None

    candidate = geometry
    best_candidate = geometry
    best_clearance = clearance_or_zero(geometry)
    # Small tolerance to avoid extra passes from floating-point near-misses
    target_with_tolerance = min_clearance * (1 - 1e-4)
    for _ in range(max_passes):
        result = _move_single_hole_pinch(candidate, min_clearance, hole_index)
        if result is None:
            break
        try:
            new_clearance = result.minimum_clearance
        except (GEOSException, ValueError):
            break

        # Regression guard: stop if this pass made things worse
        if new_clearance <= best_clearance:
            break

        best_candidate = result
        best_clearance = new_clearance
        candidate = result

        if new_clearance >= target_with_tolerance:
            return result

    # Return improved geometry even if target not fully met
    if best_candidate is not geometry:
        return best_candidate
    return None


def _move_single_hole_pinch(
    geometry: Polygon,
    min_clearance: float,
    hole_index: int,
) -> Polygon | None:
    """Move a single pair of near-coincident vertices apart on a hole ring."""
    holes = list(geometry.interiors)
    if hole_index >= len(holes):
        return None

    try:
        clearance_line = shapely.minimum_clearance_line(geometry)
    except (GEOSException, ValueError):
        return None
    if clearance_line.is_empty:
        return None

    cl_pts = np.array(clearance_line.coords)
    if len(cl_pts) < 2:
        return None

    # Verify that the clearance bottleneck is on this hole ring
    hole_coords = np.array(holes[hole_index].coords)
    idx1 = _find_nearest_vertex_index(hole_coords, cl_pts[0])
    idx2 = _find_nearest_vertex_index(hole_coords, cl_pts[1])
    if idx1 == idx2:
        return None

    direction = cl_pts[1] - cl_pts[0]
    dist = float(np.linalg.norm(direction))
    if dist >= min_clearance:
        return None  # This pair already meets the target
    if dist < 1e-10:
        direction_norm = direction  # Will be overridden below
    else:
        direction_norm = direction / dist

    # Confirm endpoints are actually on this hole (not exterior or another hole)
    snap_dist_0 = float(np.linalg.norm(hole_coords[idx1][:2] - cl_pts[0][:2]))
    snap_dist_1 = float(np.linalg.norm(hole_coords[idx2][:2] - cl_pts[1][:2]))
    if dist > 1e-10 and (snap_dist_0 > dist * 10 or snap_dist_1 > dist * 10):
        return None

    # For near-zero gaps, compute direction from vertex positions directly
    if dist < 1e-10:
        v_direction = hole_coords[idx2][:2] - hole_coords[idx1][:2]
        v_dist = float(np.linalg.norm(v_direction))
        if v_dist < 1e-10:
            return None
        direction_norm = v_direction / v_dist

    # Move each vertex apart by half the remaining deficit
    move_dist = (min_clearance - dist) / 2
    new_hole = hole_coords.copy()
    new_hole[idx1][:2] = hole_coords[idx1][:2] - direction_norm[:2] * move_dist
    new_hole[idx2][:2] = hole_coords[idx2][:2] + direction_norm[:2] * move_dist

    # Update closing vertex if needed
    if idx1 == 0 or idx2 == 0:
        new_hole[-1] = new_hole[0]

    # Reconstruct polygon with modified hole
    new_holes = []
    for i, h in enumerate(holes):
        if i == hole_index:
            new_holes.append(new_hole)
        else:
            new_holes.append(list(h.coords))

    try:
        candidate = Polygon(geometry.exterior, new_holes)
    except (GEOSException, ValueError):
        return None

    if not candidate.is_valid or candidate.is_empty:
        return None

    return candidate


def _move_inter_hole_pinch(
    geometry: Polygon,
    min_clearance: float,
    context: ClearanceContext,
    max_passes: int = 5,
) -> Polygon | None:
    """Move near-coincident vertices on two different hole rings apart.

    Iterates like _move_hole_pinch_vertices but for the inter-hole case.
    """
    hole_idx_a = context.hole_indices[0]
    hole_idx_b = context.hole_indices[1]
    if hole_idx_a is None or hole_idx_b is None:
        return None

    candidate = geometry
    best_candidate = geometry
    best_clearance = clearance_or_zero(geometry)
    target_with_tolerance = min_clearance * (1 - 1e-4)
    for _ in range(max_passes):
        result = _move_single_inter_hole_pinch(candidate, min_clearance, hole_idx_a, hole_idx_b)
        if result is None:
            break
        try:
            new_clearance = result.minimum_clearance
        except (GEOSException, ValueError):
            break

        # Regression guard: stop if this pass made things worse
        if new_clearance <= best_clearance:
            break

        best_candidate = result
        best_clearance = new_clearance
        candidate = result

        if new_clearance >= target_with_tolerance:
            return result

    if best_candidate is not geometry:
        return best_candidate
    return None


def _move_single_inter_hole_pinch(
    geometry: Polygon,
    min_clearance: float,
    hole_idx_a: int,
    hole_idx_b: int,
) -> Polygon | None:
    """Move a single pair of near-coincident vertices apart on two different hole rings."""
    holes = list(geometry.interiors)
    if hole_idx_a >= len(holes) or hole_idx_b >= len(holes):
        return None

    try:
        clearance_line = shapely.minimum_clearance_line(geometry)
    except (GEOSException, ValueError):
        return None
    if clearance_line.is_empty:
        return None

    cl_pts = np.array(clearance_line.coords)
    if len(cl_pts) < 2:
        return None

    coords_a = np.array(holes[hole_idx_a].coords)
    coords_b = np.array(holes[hole_idx_b].coords)

    # Find nearest vertex on each hole to each clearance endpoint,
    # and pick the assignment that minimizes total snap distance.
    idx_a0 = _find_nearest_vertex_index(coords_a, cl_pts[0])
    idx_b0 = _find_nearest_vertex_index(coords_b, cl_pts[0])
    idx_a1 = _find_nearest_vertex_index(coords_a, cl_pts[1])
    idx_b1 = _find_nearest_vertex_index(coords_b, cl_pts[1])

    # Option 1: endpoint 0 → hole A, endpoint 1 → hole B
    snap_1 = float(np.linalg.norm(coords_a[idx_a0][:2] - cl_pts[0][:2])) + \
             float(np.linalg.norm(coords_b[idx_b1][:2] - cl_pts[1][:2]))
    # Option 2: endpoint 0 → hole B, endpoint 1 → hole A
    snap_2 = float(np.linalg.norm(coords_b[idx_b0][:2] - cl_pts[0][:2])) + \
             float(np.linalg.norm(coords_a[idx_a1][:2] - cl_pts[1][:2]))

    if snap_1 <= snap_2:
        va_idx, vb_idx = idx_a0, idx_b1
    else:
        va_idx, vb_idx = idx_a1, idx_b0

    direction = cl_pts[1] - cl_pts[0]
    dist = float(np.linalg.norm(direction))
    if dist >= min_clearance:
        return None

    if dist < 1e-10:
        # Compute direction from vertex positions directly
        v_direction = coords_b[vb_idx][:2] - coords_a[va_idx][:2]
        v_dist = float(np.linalg.norm(v_direction))
        if v_dist < 1e-10:
            return None
        direction_norm = v_direction / v_dist
    else:
        direction_norm = direction / dist

    # Validate snap distances
    snap_a = float(np.linalg.norm(coords_a[va_idx][:2] - cl_pts[0 if snap_1 <= snap_2 else 1][:2]))
    snap_b = float(np.linalg.norm(coords_b[vb_idx][:2] - cl_pts[1 if snap_1 <= snap_2 else 0][:2]))
    if dist > 1e-10 and (snap_a > dist * 10 or snap_b > dist * 10):
        return None

    move_dist = (min_clearance - dist) / 2

    # Move vertex on hole A away from hole B
    new_coords_a = coords_a.copy()
    new_coords_a[va_idx][:2] = coords_a[va_idx][:2] - direction_norm[:2] * move_dist
    if va_idx == 0:
        new_coords_a[-1] = new_coords_a[0]

    # Move vertex on hole B away from hole A
    new_coords_b = coords_b.copy()
    new_coords_b[vb_idx][:2] = coords_b[vb_idx][:2] + direction_norm[:2] * move_dist
    if vb_idx == 0:
        new_coords_b[-1] = new_coords_b[0]

    # Reconstruct polygon with both modified holes
    new_holes = []
    for i, h in enumerate(holes):
        if i == hole_idx_a:
            new_holes.append(new_coords_a)
        elif i == hole_idx_b:
            new_holes.append(new_coords_b)
        else:
            new_holes.append(list(h.coords))

    try:
        candidate = Polygon(geometry.exterior, new_holes)
    except (GEOSException, ValueError):
        return None

    if not candidate.is_valid or candidate.is_empty:
        return None

    return candidate


def _phase_iterative(
    geometry: Polygon,
    min_clearance: float,
    original_area: float,
    original_exterior_area: float,
    min_area_ratio: float,
    max_iterations: int,
    issue_history: list[ClearanceIssue],
) -> Polygon:
    """Phase 2: Diagnosis-driven iterative fixes."""

    def improve(poly: Polygon, target: float) -> Polygon | None:
        diagnosis = diagnose_clearance(poly, target)
        issue_history.append(diagnosis.issue)
        if diagnosis.issue == ClearanceIssue.NONE:
            return None

        candidate = None

        # For same-hole pinch points, try moving vertices apart first.
        # This is a surgical fix that only touches the 2 bottleneck vertices.
        if _is_same_hole_issue(diagnosis.context):
            candidate = _move_hole_pinch_vertices(poly, target, diagnosis.context)

        # For inter-hole pinch points (different holes with near-coincident vertices)
        if not is_usable(candidate) and _is_inter_hole_pinch_issue(diagnosis.context):
            candidate = _move_inter_hole_pinch(poly, target, diagnosis.context)

        # Fall back to strategy chain if vertex movement didn't work
        if not is_usable(candidate):
            candidate = _apply_clearance_strategy(poly, target, diagnosis)

        # Last resort: try hole ring fix (erode-dilate on the hole)
        if not is_usable(candidate):
            candidate = try_hole_ring_fix(poly, target, context=diagnosis.context)
        if not is_usable(candidate):
            return None
        if candidate.area < min_area_ratio * original_area:
            return None
        # Reject candidates that grow the exterior ring beyond a small tolerance.
        if Polygon(candidate.exterior).area > original_exterior_area * _AREA_GROWTH_TOLERANCE:
            return None
        # Clean up near-duplicate vertices that may have become the new bottleneck.
        dedup_candidate = _dedup_ring_vertices(candidate, target * 0.5)
        if is_usable(dedup_candidate):
            candidate = dedup_candidate
        return candidate

    metric = lambda poly: clearance_or_zero(poly)  # noqa: E731
    improved = iterative_improve(
        geometry,
        target_value=min_clearance,
        improve_func=improve,
        metric_func=metric,
        max_iterations=max_iterations,
    )

    if not is_usable(improved):
        improved = geometry
    if improved.area < min_area_ratio * original_area:
        improved = geometry

    return improved


def _phase_cleanup(
    geometry: Polygon,
    min_clearance: float,
    min_area_ratio: float,
    issue_history: list[ClearanceIssue],
) -> Polygon:
    """Phase 3: Final erosion-dilation cleanup pass."""
    cleanup_clearance = clearance_or_zero(geometry)
    if cleanup_clearance >= min_clearance or not geometry.is_valid or geometry.is_empty:
        return geometry
    cleanup_candidate = _erode_dilate_fix(geometry, min_clearance, min_area_ratio)
    if cleanup_candidate is not None and cleanup_candidate.is_valid:
        cleanup_result_clearance = clearance_or_zero(cleanup_candidate)
        if cleanup_result_clearance > cleanup_clearance:
            issue_history.append(ClearanceIssue.PARALLEL_CLOSE_EDGES)
            return cleanup_candidate
    return geometry


def _build_summary(
    initial_clearance: float,
    geometry: Polygon,
    original_area: float,
    issue_history: list[ClearanceIssue],
    min_clearance: float,
) -> ClearanceFixSummary:
    """Build final ClearanceFixSummary."""
    final_clearance = clearance_or_zero(geometry)
    final_area_ratio = (
        geometry.area / original_area if original_area > 0 else float("inf")
    )
    final_diag = diagnose_clearance(geometry, min_clearance)
    return ClearanceFixSummary(
        initial_clearance=initial_clearance,
        final_clearance=final_clearance,
        area_ratio=final_area_ratio,
        iterations=len(issue_history),
        issue=final_diag.issue,
        fixed=final_diag.meets_requirement,
        valid=geometry.is_valid,
        history=issue_history or [final_diag.issue],
    )


# ---------------------------------------------------------------------------
# Strategy registry and dispatch
# ---------------------------------------------------------------------------

StrategyFunc = Callable[[Polygon, float, ClearanceDiagnosis], Polygon | None]

STRATEGY_REGISTRY: dict[ClearanceIssue, StrategyFunc] = {}


def _strategy(issue: ClearanceIssue) -> StrategyFunc:
    return STRATEGY_REGISTRY.get(issue, _strategy_default)


def _apply_clearance_strategy(
    geometry: Polygon,
    min_clearance: float,
    diagnosis: ClearanceDiagnosis,
) -> Polygon | None:
    """Apply the diagnosed strategy, falling back to alternatives on failure."""
    current_clearance = diagnosis.current_clearance
    strategies = _build_fallback_chain(diagnosis.issue)

    for strategy_func in strategies:
        candidate = strategy_func(geometry, min_clearance, diagnosis)
        normalized = normalize_polygon(candidate)
        if normalized is None:
            continue
        try:
            new_clearance = normalized.minimum_clearance
            if new_clearance > current_clearance:
                return normalized
        except (GEOSException, ValueError):
            continue

    return None


def _build_fallback_chain(issue: ClearanceIssue) -> list[StrategyFunc]:
    """Return an ordered list of strategy functions to try.

    The diagnosed strategy is tried first. If it fails, erosion-dilation
    is tried as a universal fallback. Finally, generic passage widening
    is tried as last resort.
    """
    primary = _strategy(issue)
    chain = [primary]

    # Erosion-dilation as universal fallback (not useful for hole issues)
    if issue != ClearanceIssue.HOLE_TOO_CLOSE:
        chain.append(_strategy_erode_dilate)

    # Generic passage widening as last resort
    if issue not in (ClearanceIssue.NARROW_PASSAGE, ClearanceIssue.UNKNOWN):
        chain.append(_strategy_default)

    return chain


def _strategy_erode_dilate(
    geometry: Polygon,
    min_clearance: float,
    _: ClearanceDiagnosis,
) -> Polygon | None:
    """Morphological approach: erode then dilate to remove narrow features."""
    return _erode_dilate_fix(geometry, min_clearance, min_area_ratio=0.85)


def _register_strategy(issue: ClearanceIssue):
    def decorator(func: StrategyFunc) -> StrategyFunc:
        STRATEGY_REGISTRY[issue] = func
        return func

    return decorator


@_register_strategy(ClearanceIssue.HOLE_TOO_CLOSE)
def _strategy_hole_too_close(
    geometry: Polygon,
    min_clearance: float,
    _: ClearanceDiagnosis,
) -> Polygon | None:
    fixed = fix_hole_too_close(geometry, min_clearance, strategy=HoleStrategy.REMOVE)
    return fixed


@_register_strategy(ClearanceIssue.NARROW_PROTRUSION)
def _strategy_narrow_protrusion(
    geometry: Polygon,
    min_clearance: float,
    diagnosis: ClearanceDiagnosis,
) -> Polygon | None:
    baseline = diagnosis.current_clearance
    # Try targeted fix first — moves vertices apart at the clearance bottleneck
    result = fix_narrow_protrusion(geometry, min_clearance)
    if is_usable(result) and clearance_or_zero(result) > baseline:
        return result
    # Fallback: blunt vertex removal for micro-features that targeted fix can't resolve.
    # When separation=0 (vertex-to-edge bottleneck), use targeted removal of
    # vertices on the offending edge rather than blunt aspect-ratio removal.
    context = diagnosis.context
    if context is not None and context.separation == 0:
        targeted = _try_remove_bottleneck_edge_vertex(geometry, min_clearance)
        if is_usable(targeted) and clearance_or_zero(targeted) > baseline:
            return targeted
        return None
    fallback = remove_narrow_protrusions(geometry, aspect_ratio_threshold=6.0)
    if (
        fallback.is_valid
        and fallback.area <= geometry.area * _AREA_GROWTH_TOLERANCE
        and clearance_or_zero(fallback) > baseline
    ):
        return fallback
    return None


@_register_strategy(ClearanceIssue.NARROW_WEDGE)
def _strategy_narrow_wedge(
    geometry: Polygon,
    min_clearance: float,
    diagnosis: ClearanceDiagnosis,
) -> Polygon | None:
    """Strategy for narrow wedge intrusions: trace and bridge the opening."""
    result = remove_narrow_wedges(geometry, angle_threshold=_ACUTE_TIP_ANGLE)
    if result is not None:
        return result
    # Fall back to regular protrusion handling
    return _strategy_narrow_protrusion(geometry, min_clearance, diagnosis)


@_register_strategy(ClearanceIssue.NARROW_PASSAGE)
def _strategy_narrow_passage(
    geometry: Polygon,
    min_clearance: float,
    _: ClearanceDiagnosis,
) -> Polygon | None:
    return fix_narrow_passage(geometry, min_clearance, strategy=PassageStrategy.WIDEN)


@_register_strategy(ClearanceIssue.NEAR_SELF_INTERSECTION)
def _strategy_near_self_intersection(
    geometry: Polygon,
    min_clearance: float,
    diagnosis: ClearanceDiagnosis,
) -> Polygon | None:
    # Try closing narrow passage to courtyard hole first
    courtyard_result = try_close_passage_to_hole(geometry, min_clearance)
    if courtyard_result is not None:
        return courtyard_result
    context = diagnosis.context
    if context is not None and context.separation <= 2:
        # Try targeted dedup for near-collinear vertices
        dedup_result = _dedup_ring_vertices(geometry, min_clearance * 0.5)
        if is_usable(dedup_result):
            dedup_clearance = clearance_or_zero(dedup_result)
            if dedup_clearance > diagnosis.current_clearance:
                return dedup_result
        # Try targeted vertex removal at the bottleneck edge
        targeted = _try_remove_bottleneck_edge_vertex(geometry, min_clearance)
        if is_usable(targeted) and clearance_or_zero(targeted) > diagnosis.current_clearance:
            return targeted
    return fix_near_self_intersection(
        geometry,
        min_clearance,
        strategy=IntersectionStrategy.SIMPLIFY,
    )


@_register_strategy(ClearanceIssue.PARALLEL_CLOSE_EDGES)
def _strategy_parallel_edges(
    geometry: Polygon,
    min_clearance: float,
    _: ClearanceDiagnosis,
) -> Polygon | None:
    # Try closing narrow passage to courtyard hole first
    courtyard_result = try_close_passage_to_hole(geometry, min_clearance)
    if courtyard_result is not None:
        return courtyard_result
    return fix_parallel_close_edges(
        geometry,
        min_clearance,
        strategy=IntersectionStrategy.SIMPLIFY,
    )


def _strategy_default(
    geometry: Polygon,
    min_clearance: float,
    _: ClearanceDiagnosis,
) -> Polygon | None:
    return fix_narrow_passage(geometry, min_clearance, strategy=PassageStrategy.WIDEN)


# ---------------------------------------------------------------------------
# Dedup helpers (tightly coupled to orchestration)
# ---------------------------------------------------------------------------


def _dedup_ring_coords(coords: np.ndarray, tolerance: float) -> np.ndarray | None:
    """Remove near-duplicate vertices from a closed ring, including across the seam.

    Returns deduplicated coordinates (closed ring) or None if the ring would
    become degenerate (< 4 coords including closing vertex).
    """
    n = len(coords) - 1  # exclude closing vertex
    if n < 3:
        return None

    # Mark vertices to keep (all True initially)
    keep = [True] * n
    prev_kept = 0
    for i in range(1, n):
        dist = float(np.linalg.norm(coords[i][:2] - coords[prev_kept][:2]))
        if dist <= tolerance:
            keep[i] = False
        else:
            prev_kept = i

    # Check seam: if last kept vertex is near-duplicate of first kept vertex
    last_kept = max(i for i in range(n) if keep[i])
    first_kept = min(i for i in range(n) if keep[i])
    if last_kept != first_kept:
        seam_dist = float(np.linalg.norm(coords[last_kept][:2] - coords[first_kept][:2]))
        if seam_dist <= tolerance:
            keep[last_kept] = False

    result = [coords[i] for i in range(n) if keep[i]]
    if len(result) < 3:
        return None

    # Close the ring
    result.append(result[0].copy())
    return np.array(result)


def _dedup_ring_vertices(geometry: Polygon, tolerance: float) -> Polygon | None:
    """Deduplicate near-duplicate vertices on all rings of a polygon.

    Handles the ring seam correctly (where last unique vertex meets first).
    Returns None if deduplication would create a degenerate geometry.
    """
    ext_coords = np.array(geometry.exterior.coords)
    new_ext = _dedup_ring_coords(ext_coords, tolerance)
    if new_ext is None or len(new_ext) < 4:
        return None

    new_holes = []
    for hole in geometry.interiors:
        hole_coords = np.array(hole.coords)
        new_hole = _dedup_ring_coords(hole_coords, tolerance)
        if new_hole is not None and len(new_hole) >= 4:
            new_holes.append(new_hole)
        # If a hole becomes degenerate, drop it (this is acceptable)

    try:
        result = Polygon(new_ext, new_holes)
        if result.is_valid and not result.is_empty:
            return result
    except (GEOSException, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Bottleneck vertex removal (used by 2 strategies)
# ---------------------------------------------------------------------------


def _try_remove_bottleneck_edge_vertex(
    geometry: Polygon, min_clearance: float
) -> Polygon | None:
    """Try removing a vertex on the edge causing a vertex-to-edge clearance bottleneck.

    When separation=0, both clearance endpoints map to the same vertex, and the
    second endpoint lies on a non-adjacent edge.  Removing a vertex on that edge
    changes its position, potentially resolving the bottleneck without touching
    unrelated parts of the polygon.
    """
    try:
        clearance_line = shapely.minimum_clearance_line(geometry)
    except (GEOSException, ValueError):
        return None
    if clearance_line.is_empty:
        return None

    pts = np.array(clearance_line.coords)
    if len(pts) < 2:
        return None

    coords = np.array(geometry.exterior.coords)
    n = len(coords) - 1
    if n < 5:  # need at least 4 vertices after removal
        return None

    # Find the edge that the second clearance endpoint lies on
    best_candidate: Polygon | None = None
    best_clearance = clearance_or_zero(geometry)

    for endpoint in pts:
        # Find the edge closest to this endpoint (vectorized numpy)
        edge_idx = _find_nearest_edge_index(coords, endpoint)
        edge_verts = [edge_idx, (edge_idx + 1) % n]

        # Try removing each endpoint of the closest edge
        for vi in edge_verts:
            new_coords = np.delete(coords, vi, axis=0)
            if len(new_coords) < 4:
                continue
            if not np.allclose(new_coords[0], new_coords[-1]):
                new_coords[-1] = new_coords[0]
            holes = [list(h.coords) for h in geometry.interiors]
            try:
                candidate = Polygon(new_coords, holes=holes) if holes else Polygon(new_coords)
            except (GEOSException, ValueError):
                continue
            if not candidate.is_valid or candidate.is_empty:
                continue
            cand_clearance = clearance_or_zero(candidate)
            if cand_clearance > best_clearance:
                best_clearance = cand_clearance
                best_candidate = candidate

    return best_candidate


__all__ = [
    "fix_clearance",
    "diagnose_clearance",
    "ClearanceIssue",
    "ClearanceDiagnosis",
    "ClearanceFixSummary",
]
