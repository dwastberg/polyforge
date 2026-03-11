from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, LinearRing
from shapely.geometry.base import BaseGeometry

from polyforge.ops.clearance import (
    fix_hole_too_close,
    fix_narrow_protrusion,
    remove_narrow_protrusions,
    remove_narrow_wedges,
    fix_narrow_passage,
    fix_near_self_intersection,
    fix_parallel_close_edges,
)
from polyforge.ops.clearance.utils import (
    _find_nearest_vertex_index,
    _find_nearest_edge_index,
    _calculate_curvature_at_vertex,
    _compute_wedge_tip_angle,
)
from polyforge.ops.clearance.passages import _erode_dilate_fix
from polyforge.core.geometry_utils import to_single_polygon
from polyforge.core.types import HoleStrategy, PassageStrategy, IntersectionStrategy
from polyforge.core.iterative_utils import iterative_improve
from shapely.errors import GEOSException
from polyforge.metrics import _safe_clearance

# --- Clearance diagnosis thresholds ---
# Turning angle (degrees) above which a vertex is considered a sharp reversal,
# characteristic of spike-like protrusions.  135 degrees means the path turns more
# than 45 degrees past a right-angle bend.
_PROTRUSION_SHARP_ANGLE = 135.0

# Turning angle threshold for detecting protrusions when vertices are very
# close together (separation <= _CLOSE_VERTEX_SEPARATION).  A 90 degree turn
# combined with close vertices typically indicates a narrow finger.
_PROTRUSION_MODERATE_ANGLE = 90.0

# Maximum vertex separation (ring-distance between the two clearance-line
# endpoints) that qualifies as "close".  Used to distinguish protrusions and
# near-self-intersections from parallel-edge issues.
_CLOSE_VERTEX_SEPARATION = 3

# Maximum turning angle for near-self-intersection detection.  When both
# endpoints have low curvature (smooth turns) but are very close in ring
# distance, the geometry likely doubles back on itself.
_SELF_INTERSECTION_MAX_ANGLE = 90.0

# Maximum angle (degrees) between edge directions at the two clearance-line
# endpoints for them to be considered "parallel".
_PARALLEL_EDGE_MAX_ANGLE = 30.0

# Maximum tip angle (degrees) for angle-based wedge detection.  Features with
# a tip angle below this threshold are classified as acute wedges, even when
# the opening width already exceeds min_clearance.
_ACUTE_TIP_ANGLE = 20.0

# Floating-point tolerance for clearance comparisons.  Shapely's
# minimum_clearance can return values like 0.9999999999999982 for what is
# geometrically 1.0.
_CLEARANCE_TOLERANCE = 1e-9

# Maximum allowed area growth factor. Clearance fixes should not significantly
# increase polygon area (e.g., by filling in concavities).  Set to 2% to allow
# minor growth from removing concave vertices at clearance bottlenecks (e.g.,
# vertex-to-edge issues where a single concave vertex is removed, slightly
# expanding the exterior).
_AREA_GROWTH_TOLERANCE = 1.02

# Minimum ratio of enclosed courtyard area to min_clearance² for courtyard
# passage detection.  A courtyard must be at least this many times larger than
# a min_clearance-sided square to be worth preserving as a hole.
_COURTYARD_AREA_THRESHOLD = 10.0


class ClearanceIssue(Enum):
    """Enumerates the types of clearance problems we can detect."""

    NONE = "none"
    HOLE_TOO_CLOSE = "hole_too_close"
    NARROW_PROTRUSION = "narrow_protrusion"
    NARROW_WEDGE = "narrow_wedge"
    NARROW_PASSAGE = "narrow_passage"
    NEAR_SELF_INTERSECTION = "near_self_intersection"
    PARALLEL_CLOSE_EDGES = "parallel_close_edges"
    UNKNOWN = "unknown"


@dataclass
class ClearanceContext:
    """Geometric context around a clearance bottleneck."""

    curvature: tuple[float, float]
    separation: int
    vertex_count: int
    ring_types: tuple[str, str]
    edge_angle_similarity: float | None = None
    narrow_extent: int = 0
    tip_angle: float | None = None
    hole_indices: tuple[int | None, int | None] = (None, None)


@dataclass
class ClearanceDiagnosis:
    """
    Result of analyzing a polygon's clearance.

    Attributes:
        issue: Detected issue type.
        meets_requirement: Whether min_clearance is already satisfied.
        current_clearance: Measured minimum clearance.
        clearance_ratio: current_clearance / min_clearance.
        clearance_line: Location of the bottleneck if available.
        recommended_fix: Name of the suggested fix function.
        context: Optional clearance context with geometric details of the bottleneck.
    """

    issue: ClearanceIssue
    meets_requirement: bool
    current_clearance: float
    clearance_ratio: float
    clearance_line: LineString | None
    recommended_fix: str
    context: ClearanceContext | None = None

    @property
    def has_issues(self) -> bool:
        return not self.meets_requirement


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
    original_exterior_area = Polygon(geometry.exterior).area
    summary = ClearanceFixSummary(
        initial_clearance=initial_clearance,
        final_clearance=initial_clearance,
        area_ratio=1.0,
        iterations=0,
        issue=ClearanceIssue.NONE,
        fixed=initial_clearance >= min_clearance,
        valid=geometry.is_valid,
        history=[ClearanceIssue.NONE],
    )

    if summary.fixed:
        return (geometry, summary) if return_diagnosis else geometry

    # --- Phase 0: Deduplicate near-duplicate vertices ---
    # Near-duplicate vertices on any ring are a data quality issue that causes
    # Shapely's minimum_clearance to report tiny values.  Deduplicating them
    # first often resolves the bottleneck entirely, avoiding misdiagnosis
    # (e.g., hole-ring self-clearance mistaken for hole-too-close).
    dedup_tolerance = min_clearance * 0.1
    dedup_candidate = _dedup_ring_vertices(geometry, dedup_tolerance)
    if dedup_candidate is not None and dedup_candidate.is_valid and not dedup_candidate.is_empty:
        dedup_clearance = _safe_clearance(dedup_candidate) or 0.0
        if dedup_clearance >= min_clearance:
            final_area_ratio = (
                dedup_candidate.area / original_area if original_area > 0 else float("inf")
            )
            summary = ClearanceFixSummary(
                initial_clearance=initial_clearance,
                final_clearance=dedup_clearance,
                area_ratio=final_area_ratio,
                iterations=1,
                issue=ClearanceIssue.NONE,
                fixed=True,
                valid=dedup_candidate.is_valid,
                history=[ClearanceIssue.NONE],
            )
            return (dedup_candidate, summary) if return_diagnosis else dedup_candidate
        # Continue with deduplicated geometry even if clearance not yet met
        geometry = dedup_candidate
        original_exterior_area = Polygon(geometry.exterior).area

    issue_history: list[ClearanceIssue] = []

    # --- Phase 1: Region fix (erosion-dilation) ---
    # Handles slivers, narrow peninsulas, and extended narrow features in O(1).
    region_candidate = _erode_dilate_fix(geometry, min_clearance, min_area_ratio)
    if region_candidate is not None:
        region_clearance = _safe_clearance(region_candidate) or 0.0
        if region_clearance >= min_clearance:
            issue_history.append(ClearanceIssue.PARALLEL_CLOSE_EDGES)
            final_area_ratio = (
                region_candidate.area / original_area
                if original_area > 0
                else float("inf")
            )
            summary = ClearanceFixSummary(
                initial_clearance=initial_clearance,
                final_clearance=region_clearance,
                area_ratio=final_area_ratio,
                iterations=1,
                issue=ClearanceIssue.NONE,
                fixed=True,
                valid=region_candidate.is_valid,
                history=issue_history,
            )
            return (region_candidate, summary) if return_diagnosis else region_candidate

    # --- Phase 1.5: Courtyard passage detection ---
    # If the exterior ring has a narrow passage enclosing a significant area,
    # close the passage and convert the courtyard to a hole.  This must run
    # before the iterative improve() loop because creating a hole legitimately
    # changes exterior area and polygon area, which would be rejected by the
    # area guards in improve().
    courtyard_candidate = _try_close_passage_to_hole(geometry, min_clearance)
    if courtyard_candidate is not None and courtyard_candidate.is_valid:
        courtyard_clearance = _safe_clearance(courtyard_candidate) or 0.0
        if courtyard_clearance >= min_clearance:
            issue_history.append(ClearanceIssue.NEAR_SELF_INTERSECTION)
            final_area_ratio = (
                courtyard_candidate.area / original_area
                if original_area > 0
                else float("inf")
            )
            summary = ClearanceFixSummary(
                initial_clearance=initial_clearance,
                final_clearance=courtyard_clearance,
                area_ratio=final_area_ratio,
                iterations=1,
                issue=ClearanceIssue.NONE,
                fixed=True,
                valid=courtyard_candidate.is_valid,
                history=issue_history,
            )
            return (courtyard_candidate, summary) if return_diagnosis else courtyard_candidate
        # Courtyard created but clearance not yet met — continue with it
        # as starting point for Phase 2 (remaining clearance at pinch point).
        if courtyard_clearance > initial_clearance:
            geometry = courtyard_candidate
            original_exterior_area = Polygon(geometry.exterior).area
            issue_history.append(ClearanceIssue.NEAR_SELF_INTERSECTION)

    # --- Phase 2: Point-based fixes (diagnosis-driven iteration) ---
    # For cases where erosion-dilation is too aggressive (would lose too much area)
    # or doesn't fully solve the problem (e.g., holes, single protrusions).
    def improve(poly: Polygon, target: float) -> Polygon | None:
        diagnosis = diagnose_clearance(poly, target)
        issue_history.append(diagnosis.issue)
        if diagnosis.issue == ClearanceIssue.NONE:
            return None
        candidate = _apply_clearance_strategy(poly, target, diagnosis)
        # If standard strategy failed, try hole ring simplification for
        # same-hole self-clearance issues.
        if candidate is None or not candidate.is_valid or candidate.is_empty:
            candidate = _try_hole_ring_fix(poly, target, context=diagnosis.context)
        if candidate is None or not candidate.is_valid or candidate.is_empty:
            return None
        if candidate.area < min_area_ratio * original_area:
            return None
        # Reject candidates that grow the exterior ring beyond a small tolerance.
        # We check exterior area (not polygon area) so that hole fixes—which
        # legitimately increase polygon area by shrinking holes—are not blocked.
        if Polygon(candidate.exterior).area > original_exterior_area * _AREA_GROWTH_TOLERANCE:
            return None
        # Clean up near-duplicate vertices that may have become the new bottleneck.
        # Uses a larger tolerance than Phase 0 dedup to catch vertices exposed by
        # the fix (e.g., near-collinear seam vertices unmasked after a hole fix).
        dedup_candidate = _dedup_ring_vertices(candidate, target * 0.5)
        if dedup_candidate is not None and dedup_candidate.is_valid and not dedup_candidate.is_empty:
            candidate = dedup_candidate
        return candidate

    metric = lambda poly: _safe_clearance(poly) or 0.0  # noqa: E731
    improved = iterative_improve(
        geometry,
        target_value=min_clearance,
        improve_func=improve,
        metric_func=metric,
        max_iterations=max_iterations,
    )

    if improved is None or not improved.is_valid or improved.is_empty:
        improved = geometry
    if improved.area < min_area_ratio * original_area:
        improved = geometry

    # --- Phase 3: Cleanup pass (erosion-dilation on improved geometry) ---
    # Point fixes may have partially resolved slivers, making erosion viable
    # where it wasn't in Phase 1.
    cleanup_clearance = _safe_clearance(improved) or 0.0
    if (
        cleanup_clearance < min_clearance
        and improved.is_valid
        and not improved.is_empty
    ):
        cleanup_candidate = _erode_dilate_fix(improved, min_clearance, min_area_ratio)
        if cleanup_candidate is not None and cleanup_candidate.is_valid:
            cleanup_result_clearance = _safe_clearance(cleanup_candidate) or 0.0
            if cleanup_result_clearance > cleanup_clearance:
                improved = cleanup_candidate
                issue_history.append(ClearanceIssue.PARALLEL_CLOSE_EDGES)

    final_clearance = _safe_clearance(improved) or 0.0
    final_area_ratio = (
        improved.area / original_area if original_area > 0 else float("inf")
    )
    final_diag = diagnose_clearance(improved, min_clearance)
    summary = ClearanceFixSummary(
        initial_clearance=initial_clearance,
        final_clearance=final_clearance,
        area_ratio=final_area_ratio,
        iterations=len(issue_history),
        issue=final_diag.issue,
        fixed=final_diag.meets_requirement,
        valid=improved.is_valid,
        history=issue_history or [final_diag.issue],
    )

    return (improved, summary) if return_diagnosis else improved


StrategyFunc = Callable[[Polygon, float, ClearanceDiagnosis], Polygon | None]

RECOMMENDED_FIXES: dict[ClearanceIssue, str] = {
    ClearanceIssue.NONE: "none",
    ClearanceIssue.HOLE_TOO_CLOSE: "fix_hole_too_close",
    ClearanceIssue.NARROW_PROTRUSION: "fix_narrow_protrusion",
    ClearanceIssue.NARROW_WEDGE: "fill_narrow_wedge",
    ClearanceIssue.NARROW_PASSAGE: "fix_narrow_passage",
    ClearanceIssue.NEAR_SELF_INTERSECTION: "fix_near_self_intersection",
    ClearanceIssue.PARALLEL_CLOSE_EDGES: "fix_parallel_close_edges",
    ClearanceIssue.UNKNOWN: "fix_narrow_passage",
}

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
        normalized = _normalize_polygon(candidate)
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
    if result is not None and result.is_valid and (_safe_clearance(result) or 0.0) > baseline:
        return result
    # Fallback: blunt vertex removal for micro-features that targeted fix can't resolve.
    # When separation=0 (vertex-to-edge bottleneck), use targeted removal of
    # vertices on the offending edge rather than blunt aspect-ratio removal.
    # remove_narrow_protrusions picks the highest aspect-ratio vertex globally,
    # which is often a nearly-collinear vertex far from the bottleneck — removing
    # it cascades into destructive removal of structural vertices.
    context = diagnosis.context
    if context is not None and context.separation == 0:
        targeted = _try_remove_bottleneck_edge_vertex(geometry, min_clearance)
        if (
            targeted is not None
            and targeted.is_valid
            and (_safe_clearance(targeted) or 0.0) > baseline
        ):
            return targeted
        return None
    fallback = remove_narrow_protrusions(geometry, aspect_ratio_threshold=6.0)
    if (
        fallback.is_valid
        and fallback.area <= geometry.area * _AREA_GROWTH_TOLERANCE
        and (_safe_clearance(fallback) or 0.0) > baseline
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
    courtyard_result = _try_close_passage_to_hole(geometry, min_clearance)
    if courtyard_result is not None:
        return courtyard_result
    context = diagnosis.context
    if context is not None and context.separation <= 2:
        # Try targeted dedup for near-collinear vertices
        dedup_result = _dedup_ring_vertices(geometry, min_clearance * 0.5)
        if dedup_result is not None and dedup_result.is_valid:
            dedup_clearance = _safe_clearance(dedup_result) or 0.0
            if dedup_clearance > diagnosis.current_clearance:
                return dedup_result
        # Try targeted vertex removal at the bottleneck edge
        targeted = _try_remove_bottleneck_edge_vertex(geometry, min_clearance)
        if (
            targeted is not None
            and targeted.is_valid
            and (_safe_clearance(targeted) or 0.0) > diagnosis.current_clearance
        ):
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
    courtyard_result = _try_close_passage_to_hole(geometry, min_clearance)
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


def _try_hole_ring_fix(
    geometry: Polygon,
    min_clearance: float,
    context: ClearanceContext | None = None,
) -> Polygon | None:
    """Try to fix clearance by reshaping the offending hole ring.

    Checks if the minimum_clearance bottleneck is on a single hole ring.
    Tries multiple strategies in order:
    1. Erosion-dilation on the hole ring (handles spikes and close passages)
    2. Spike removal (targeted fix for long narrow spikes)
    3. Simplification (general fallback)
    """
    if context is None:
        context = _build_clearance_context(geometry, min_clearance)
    if context is None:
        return None
    # Only applies to same-hole self-clearance
    if not (
        context.ring_types == ("hole", "hole")
        and context.hole_indices[0] is not None
        and context.hole_indices[0] == context.hole_indices[1]
    ):
        return None

    current_clearance = _safe_clearance(geometry) or 0.0
    hole_index = context.hole_indices[0]

    # Try all strategies and pick the best result.
    best_candidate = None
    best_clearance = current_clearance

    # Erosion-dilation on the hole ring — handles both spikes and
    # self-proximity in a single operation.
    erode_candidate = _erode_dilate_hole_ring(
        geometry, hole_index, min_clearance
    )
    if erode_candidate is not None and erode_candidate.is_valid:
        erode_clearance = _safe_clearance(erode_candidate) or 0.0
        if erode_clearance > best_clearance:
            best_clearance = erode_clearance
            best_candidate = erode_candidate

    # Spike removal — handles long, narrow spikes that erosion-dilation
    # might not fix (e.g., when area loss is too large).
    spike_candidate = _remove_hole_ring_spike(geometry, hole_index, min_clearance)
    if spike_candidate is not None and spike_candidate.is_valid:
        spike_clearance = _safe_clearance(spike_candidate) or 0.0
        if spike_clearance > best_clearance:
            best_clearance = spike_clearance
            best_candidate = spike_candidate

    # Simplification
    simp_candidate = _simplify_hole_ring(geometry, hole_index, min_clearance)
    if simp_candidate is not None and simp_candidate.is_valid:
        simp_clearance = _safe_clearance(simp_candidate) or 0.0
        if simp_clearance > best_clearance:
            best_clearance = simp_clearance
            best_candidate = simp_candidate

    return best_candidate


def _erode_dilate_hole_ring(
    geometry: Polygon,
    hole_index: int,
    min_clearance: float,
    min_area_ratio: float = 0.8,
) -> Polygon | None:
    """Apply erosion-dilation to a specific hole ring to fix self-clearance.

    Buffers the hole polygon inward (erode) then outward (dilate) to remove
    narrow features like spikes and near-self-intersections. The result is
    simplified to avoid vertex explosion from buffering.

    Args:
        geometry: Input polygon.
        hole_index: Index of the hole to fix.
        min_clearance: Target minimum clearance.
        min_area_ratio: Minimum ratio of fixed hole area to original (default 0.8).

    Returns:
        Fixed polygon or None if the fix fails or loses too much hole area.
    """
    holes = list(geometry.interiors)
    if hole_index < 0 or hole_index >= len(holes):
        return None

    hole_ring = holes[hole_index]
    hole_poly = Polygon(hole_ring)
    original_hole_area = hole_poly.area
    if original_hole_area <= 0:
        return None

    # Try increasing buffer distances until clearance is resolved
    for scale in (1.0, 2.0, 4.0):
        d = min_clearance * scale
        half_d = d / 2

        try:
            smoothed = hole_poly.buffer(-half_d).buffer(half_d)
        except (GEOSException, ValueError):
            continue

        if smoothed.is_empty or not smoothed.is_valid:
            continue

        # Take the largest polygon if buffer split the hole
        if isinstance(smoothed, MultiPolygon):
            smoothed = max(smoothed.geoms, key=lambda p: p.area)

        # Check hole area preservation
        if smoothed.area < original_hole_area * min_area_ratio:
            continue

        # Simplify to reduce vertex count from buffering
        smoothed = smoothed.simplify(d * 0.25)
        if smoothed.is_empty or not smoothed.is_valid:
            continue

        # Reconstruct polygon with the smoothed hole
        new_holes = list(holes)
        new_holes[hole_index] = smoothed.exterior

        try:
            candidate = Polygon(geometry.exterior, new_holes)
        except (GEOSException, ValueError):
            continue

        if candidate.is_valid and not candidate.is_empty:
            return candidate

    return None


def _remove_hole_ring_spike(
    geometry: Polygon, hole_index: int, min_clearance: float
) -> Polygon | None:
    """Remove spike features from a hole ring that cause near-self-intersection.

    A spike is a pattern where the ring goes from vertex A to a distant vertex B
    and then back to vertex C near A, creating a long narrow protrusion.
    The fix removes the spike vertex B and merges A and C into their midpoint.

    This handles cases that simplification cannot fix: when the spike tip
    deviates far from the baseline (e.g., 10 units) but the base width is
    tiny (e.g., 0.15 units), RDP/VW won't remove the tip vertex.
    """
    holes = list(geometry.interiors)
    if hole_index < 0 or hole_index >= len(holes):
        return None

    hole_coords = np.array(holes[hole_index].coords)
    n = len(hole_coords) - 1  # exclude closing vertex
    if n < 5:  # need at least 4 vertices after spike removal
        return None

    # Find the clearance bottleneck location on this hole ring
    try:
        clearance_line = shapely.minimum_clearance_line(geometry)
    except (GEOSException, ValueError):
        return None
    if clearance_line.is_empty:
        return None

    cl_pts = np.array(clearance_line.coords)
    if len(cl_pts) < 2:
        return None

    # Find the nearest vertex on the hole ring to each clearance endpoint
    idx1 = _find_nearest_vertex_index(hole_coords, cl_pts[0])
    idx2 = _find_nearest_vertex_index(hole_coords, cl_pts[1])

    # Look for spike patterns around the clearance bottleneck.
    # A spike has two close base vertices with a far tip vertex between them.
    best_candidate = None
    best_clearance = _safe_clearance(geometry) or 0.0

    # Search vertices near the clearance endpoints for spike patterns
    search_indices = set()
    for idx in (idx1, idx2):
        for offset in range(-2, 3):
            search_indices.add((idx + offset) % n)

    for vi in search_indices:
        prev_i = (vi - 1) % n
        next_i = (vi + 1) % n

        base_dist = float(np.linalg.norm(
            hole_coords[prev_i][:2] - hole_coords[next_i][:2]
        ))
        tip_dist_prev = float(np.linalg.norm(
            hole_coords[vi][:2] - hole_coords[prev_i][:2]
        ))
        tip_dist_next = float(np.linalg.norm(
            hole_coords[vi][:2] - hole_coords[next_i][:2]
        ))

        # Spike pattern: base is narrow, both arms are much longer than base
        if base_dist >= min_clearance:
            continue
        min_arm = min(tip_dist_prev, tip_dist_next)
        if min_arm < base_dist * 3:
            continue

        # Remove the spike: delete tip vertex, merge base vertices to midpoint
        midpoint = (hole_coords[prev_i][:2] + hole_coords[next_i][:2]) / 2
        new_coords = []
        merged = False
        for j in range(n):
            if j == vi:
                continue  # skip spike tip
            if j == prev_i and not merged:
                new_coords.append(midpoint)
                merged = True
            elif j == next_i and merged:
                continue  # skip second base vertex (already merged)
            else:
                new_coords.append(hole_coords[j][:2])

        if len(new_coords) < 3:
            continue

        # Close the ring
        new_coords.append(new_coords[0].copy())
        new_ring_coords = np.array(new_coords)

        # Reconstruct polygon with modified hole
        new_holes = list(holes)
        try:
            new_hole_ring = LinearRing(new_ring_coords)
            new_holes[hole_index] = new_hole_ring
            candidate = Polygon(geometry.exterior, new_holes)
        except (GEOSException, ValueError):
            continue

        if not candidate.is_valid or candidate.is_empty:
            continue

        cand_clearance = _safe_clearance(candidate) or 0.0
        if cand_clearance > best_clearance:
            best_clearance = cand_clearance
            best_candidate = candidate

    return best_candidate


def _simplify_hole_ring(
    geometry: Polygon, hole_index: int, min_clearance: float
) -> Polygon | None:
    """Simplify a specific hole ring to resolve self-clearance issues.

    Uses Shapely's simplify on the offending hole ring with a tolerance
    derived from min_clearance, then reconstructs the polygon.
    """
    holes = list(geometry.interiors)
    if hole_index < 0 or hole_index >= len(holes):
        return None

    hole_ring = holes[hole_index]
    simplified = Polygon(hole_ring).simplify(min_clearance * 0.5)
    if simplified.is_empty or not simplified.is_valid:
        # If simplification destroys the hole, remove it
        new_holes = [h for i, h in enumerate(holes) if i != hole_index]
    else:
        new_holes = list(holes)
        new_holes[hole_index] = simplified.exterior

    try:
        result = Polygon(geometry.exterior, new_holes)
        if result.is_valid and not result.is_empty:
            return result
    except (GEOSException, ValueError):
        pass
    return None


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
    best_clearance = _safe_clearance(geometry) or 0.0

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
            cand_clearance = _safe_clearance(candidate) or 0.0
            if cand_clearance > best_clearance:
                best_clearance = cand_clearance
                best_candidate = candidate

    return best_candidate


def _try_close_passage_to_hole(
    geometry: Polygon, min_clearance: float
) -> Polygon | None:
    """Close a narrow exterior passage and convert the enclosed area to a hole.

    Detects when two nearby points on the exterior ring enclose a significant
    courtyard area behind a narrow passage.  Snaps the passage vertices together
    and uses make_valid() to split the polygon into exterior + hole.

    Args:
        geometry: Input polygon.
        min_clearance: Target minimum clearance.

    Returns:
        Polygon with courtyard as hole, or None if pattern not detected.
    """
    try:
        clearance_line = shapely.minimum_clearance_line(geometry)
    except (GEOSException, ValueError):
        return None
    if clearance_line.is_empty:
        return None

    cl_pts = np.array(clearance_line.coords)
    if len(cl_pts) < 2:
        return None

    ext_coords = np.array(geometry.exterior.coords[:-1])  # open ring
    n = len(ext_coords)
    if n < 6:
        # Need at least 6 vertices: 2 for passage + 3 for courtyard + 1 for exterior
        return None

    # Both endpoints must be on the exterior ring
    (rt1, _), (rt2, _) = _classify_ring_types(geometry, cl_pts[0], cl_pts[1])
    if rt1 != "exterior" or rt2 != "exterior":
        return None

    # Find nearest exterior vertices to each clearance endpoint
    raw_idx1 = _find_nearest_vertex_index(ext_coords, cl_pts[0])
    raw_idx2 = _find_nearest_vertex_index(ext_coords, cl_pts[1])
    if raw_idx1 == raw_idx2:
        # Both endpoints map to same vertex — use the nearest edge's closer endpoint
        edge_idx = _find_nearest_edge_index(ext_coords, cl_pts[1])
        cand_a, cand_b = edge_idx, (edge_idx + 1) % n
        # Pick the candidate that is different from raw_idx1; if both differ, pick closer
        if cand_a == raw_idx1:
            raw_idx2 = cand_b
        elif cand_b == raw_idx1:
            raw_idx2 = cand_a
        else:
            d_a = float(np.linalg.norm(ext_coords[cand_a] - cl_pts[1]))
            d_b = float(np.linalg.norm(ext_coords[cand_b] - cl_pts[1]))
            raw_idx2 = cand_a if d_a <= d_b else cand_b
    if raw_idx1 == raw_idx2:
        return None

    # Ensure idx1 < idx2
    idx1, idx2 = (raw_idx1, raw_idx2) if raw_idx1 < raw_idx2 else (raw_idx2, raw_idx1)

    # Two paths along the ring between idx1 and idx2
    path_a_indices = list(range(idx1, idx2 + 1))  # idx1 -> idx2 (forward)
    path_b_indices = list(range(idx2, n)) + list(range(0, idx1 + 1))  # idx2 -> idx1 (wrap)

    # Compute enclosed area of each path (closed with straight line between endpoints)
    area_a = _path_enclosed_area(ext_coords, path_a_indices)
    area_b = _path_enclosed_area(ext_coords, path_b_indices)

    # Courtyard is the smaller enclosed area
    if area_a <= area_b:
        courtyard_area = area_a
    else:
        courtyard_area = area_b

    # Check courtyard is significant
    if courtyard_area < _COURTYARD_AREA_THRESHOLD * min_clearance * min_clearance:
        return None

    # Need at least 4 vertices in courtyard path (to form a valid ring)
    courtyard_indices = path_a_indices if area_a <= area_b else path_b_indices
    if len(courtyard_indices) < 4:
        return None

    # --- Find passage mouth ---
    mouth = _find_passage_mouth(
        ext_coords, idx1, idx2, courtyard_indices, threshold=min_clearance * 2
    )

    if mouth is None:
        # Passage is only one vertex pair wide — use snap+make_valid fallback
        midpoint = (ext_coords[idx1] + ext_coords[idx2]) / 2
        new_coords = ext_coords.copy()
        new_coords[idx1] = midpoint
        new_coords[idx2] = midpoint
        ring = np.vstack([new_coords, [new_coords[0]]])
        holes = [list(h.coords) for h in geometry.interiors]
        try:
            self_touching = Polygon(ring, holes) if holes else Polygon(ring)
            candidate = shapely.make_valid(self_touching)
        except (GEOSException, ValueError):
            return None
        result = _normalize_polygon(candidate)
        if result is None or len(result.interiors) <= len(geometry.interiors):
            return None
        return result

    mouth1, mouth2 = mouth
    midpoint = (ext_coords[mouth1] + ext_coords[mouth2]) / 2

    # Identify arc to remove: mouth1 through passage+courtyard to mouth2
    path_fwd = list(range(mouth1, mouth2 + 1))
    path_bck = list(range(mouth2, n)) + list(range(0, mouth1 + 1))
    courtyard_set = set(courtyard_indices)
    if courtyard_set.issubset(set(path_fwd)):
        arc_to_remove = set(path_fwd)
    else:
        arc_to_remove = set(path_bck)

    # Build exterior ring: main body vertices in order, midpoint replaces arc
    ext_ring_coords = []
    midpoint_inserted = False
    for i in range(n):
        if i in arc_to_remove:
            if not midpoint_inserted:
                ext_ring_coords.append(midpoint)
                midpoint_inserted = True
        else:
            ext_ring_coords.append(ext_coords[i])

    if len(ext_ring_coords) < 3:
        return None
    ext_ring_coords.append(ext_ring_coords[0])  # close ring

    # Build hole ring from courtyard vertices
    hole_coords = [ext_coords[i].tolist() for i in courtyard_indices]
    if len(hole_coords) < 3:
        return None
    hole_coords.append(hole_coords[0])  # close ring

    # Preserve existing holes
    existing_holes = [list(h.coords) for h in geometry.interiors]
    all_holes = existing_holes + [hole_coords]

    try:
        result = Polygon(ext_ring_coords, all_holes)
    except (GEOSException, ValueError):
        return None

    if not result.is_valid or result.is_empty:
        try:
            result = _normalize_polygon(shapely.make_valid(result))
        except (GEOSException, ValueError):
            return None
        if result is None:
            return None

    # Must have gained at least one hole
    if len(result.interiors) <= len(geometry.interiors):
        return None

    return result


def _path_enclosed_area(coords: np.ndarray, indices: list[int]) -> float:
    """Compute the area enclosed by a path along the ring, closed by a straight line.

    Takes vertex indices along the ring and computes the area of the polygon
    formed by connecting them in order plus closing back to the first vertex.
    """
    if len(indices) < 3:
        return 0.0
    path_coords = coords[indices]
    # Close the ring
    closed = np.vstack([path_coords, [path_coords[0]]])
    try:
        return abs(Polygon(closed).area)
    except (GEOSException, ValueError):
        return 0.0


def _find_passage_mouth(
    ext_coords: np.ndarray,
    idx1: int,
    idx2: int,
    courtyard_indices: list[int],
    threshold: float,
) -> tuple[int, int] | None:
    """Walk outward from bottleneck to find passage mouth vertices.

    Starting from (idx1, idx2), walks along both passage walls away from
    the courtyard. Returns outermost pair still within threshold distance.

    Returns:
        (mouth_idx1, mouth_idx2) with mouth_idx1 < mouth_idx2, or None.
    """
    n = len(ext_coords)
    courtyard_set = set(courtyard_indices)

    # Determine walk direction: away from courtyard
    # From idx1, step to (idx1-1)%n; if that's in courtyard, reverse direction
    test_step = (idx1 - 1) % n
    if test_step in courtyard_set:
        dir1, dir2 = +1, -1
    else:
        dir1, dir2 = -1, +1

    mouth1, mouth2 = idx1, idx2
    max_steps = n // 2

    for _ in range(max_steps):
        next1 = (mouth1 + dir1) % n
        next2 = (mouth2 + dir2) % n

        if next1 in courtyard_set or next2 in courtyard_set:
            break
        if next1 == next2:
            break

        dist = float(np.linalg.norm(ext_coords[next1] - ext_coords[next2]))
        if dist > threshold:
            break

        mouth1, mouth2 = next1, next2

    if mouth1 == idx1 and mouth2 == idx2:
        return None  # No outward walk — passage is only at bottleneck

    m1, m2 = min(mouth1, mouth2), max(mouth1, mouth2)
    return (m1, m2)


def _normalize_polygon(candidate: BaseGeometry | None) -> Polygon | None:
    if candidate is None:
        return None
    if candidate.is_empty or not candidate.is_valid:
        return None
    polygon = to_single_polygon(candidate)
    return polygon if polygon.is_valid and not polygon.is_empty else None


def _diagnose_clearance_issue(
    geometry: Polygon, min_clearance: float
) -> tuple[ClearanceIssue, ClearanceContext | None]:
    """Diagnose the type of clearance issue in a polygon.

    Examines the geometry's minimum_clearance_line and surrounding geometry
    to determine what type of issue is causing low clearance.

    Args:
        geometry: Input polygon
        min_clearance: Target minimum clearance

    Returns:
        A tuple of (:class:`ClearanceIssue`, optional :class:`ClearanceContext`).
    """
    if _has_close_hole(geometry, min_clearance):
        return ClearanceIssue.HOLE_TOO_CLOSE, None

    context = _build_clearance_context(geometry, min_clearance)
    if context is None:
        return ClearanceIssue.UNKNOWN, None

    if "hole" in context.ring_types:
        # Distinguish same-hole self-clearance from genuine hole proximity.
        # If both endpoints are on the *same* hole ring, it's a ring geometry
        # issue (e.g., near-duplicate vertices, pinch) — let the heuristic
        # chain diagnose it.  Otherwise it's genuine hole-too-close.
        same_hole = (
            context.ring_types == ("hole", "hole")
            and context.hole_indices[0] is not None
            and context.hole_indices[0] == context.hole_indices[1]
        )
        if not same_hole:
            return ClearanceIssue.HOLE_TOO_CLOSE, context

    for heuristic in (
        _looks_like_narrow_wedge,
        _looks_like_protrusion,
        _looks_like_near_self_intersection,
        _looks_like_parallel_edges,
        _default_clearance_issue,
    ):
        issue = heuristic(context, min_clearance)
        if issue is not None:
            return issue, context
    return ClearanceIssue.UNKNOWN, context


def _classify_ring_types(
    geometry: Polygon, pt1: tuple[float, float], pt2: tuple[float, float]
) -> tuple[tuple[str, int | None], tuple[str, int | None]]:
    """Return (ring_type, hole_index) for each endpoint of the clearance line.

    ring_type is 'exterior' or 'hole'. hole_index is the index into
    geometry.interiors when ring_type == 'hole', else None.
    """

    def classify(point: tuple[float, float]) -> tuple[str, int | None]:
        p = Point(point)
        exterior_ring = LinearRing(geometry.exterior.coords)
        d_exterior = p.distance(exterior_ring)

        min_hole_dist = float("inf")
        hole_idx = None
        for idx, hole in enumerate(geometry.interiors):
            d = p.distance(LinearRing(hole.coords))
            if d < min_hole_dist:
                min_hole_dist = d
                hole_idx = idx

        if min_hole_dist < d_exterior:
            return "hole", hole_idx
        return "exterior", None

    c1 = classify(pt1)
    c2 = classify(pt2)
    return c1, c2



def _has_close_hole(geometry: Polygon, min_clearance: float) -> bool:
    """Check if any hole is closer than *min_clearance* to the exterior ring."""
    if not geometry.interiors:
        return False
    exterior_ring = LinearRing(geometry.exterior.coords)
    for hole in geometry.interiors:
        hole_ring = LinearRing(hole.coords)
        if hole_ring.distance(exterior_ring) < min_clearance:
            return True
    return False


def _compute_edge_angle_similarity(
    coords: np.ndarray, idx1: int, idx2: int
) -> float | None:
    """Compute the angle (degrees) between edge directions at two vertex indices.

    Returns the smallest angle between the two edge vectors, in [0, 180].
    Returns ``None`` if the edge vectors are degenerate (zero-length).
    """
    n = len(coords) - 1  # closed ring: last == first
    next1 = (idx1 + 1) % n
    next2 = (idx2 + 1) % n
    v1 = coords[next1] - coords[idx1]
    v2 = coords[next2] - coords[idx2]
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    if len1 < 1e-12 or len2 < 1e-12:
        return None
    cos_angle = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
    angle = np.degrees(np.arccos(abs(cos_angle)))  # abs -> [0, 90]
    return float(angle)


def _get_ring_coords_for_point(
    geometry: Polygon,
    point: np.ndarray,
    ring_type: str,
) -> np.ndarray:
    """Return the coordinate array for the ring containing the given point."""
    if ring_type == "exterior":
        return np.array(geometry.exterior.coords)

    # Find the closest hole
    p = Point(point)
    best_dist = float("inf")
    best_coords = np.array(geometry.exterior.coords)  # fallback

    for hole in geometry.interiors:
        ring = LinearRing(hole.coords)
        d = p.distance(ring)
        if d < best_dist:
            best_dist = d
            best_coords = np.array(hole.coords)

    return best_coords


def _build_clearance_context(
    geometry: Polygon, min_clearance: float = 0.0
) -> ClearanceContext | None:
    try:
        clearance_line = shapely.minimum_clearance_line(geometry)
    except (GEOSException, ValueError):
        return None

    if clearance_line.is_empty:
        return None

    coords_2d = np.array(clearance_line.coords)
    if len(coords_2d) < 2:
        return None

    pt1, pt2 = coords_2d[:2]
    classified = _classify_ring_types(geometry, pt1, pt2)
    (ring_type1, hole_idx1), (ring_type2, hole_idx2) = classified
    ring_types = (ring_type1, ring_type2)

    # Map endpoints to the correct ring for curvature analysis
    ring1_coords = _get_ring_coords_for_point(geometry, pt1, ring_type1)
    ring2_coords = _get_ring_coords_for_point(geometry, pt2, ring_type2)

    idx1 = _find_nearest_vertex_index(ring1_coords, pt1)
    idx2 = _find_nearest_vertex_index(ring2_coords, pt2)

    curvature1 = _calculate_curvature_at_vertex(ring1_coords, idx1)
    curvature2 = _calculate_curvature_at_vertex(ring2_coords, idx2)

    # Separation and edge angle only meaningful when both on the same ring
    exterior_coords = np.array(geometry.exterior.coords)
    n_exterior = len(exterior_coords) - 1

    # Check if both endpoints are on the same ring (exterior or same hole)
    same_ring = False
    shared_ring_coords = None
    if ring_type1 == ring_type2 == "exterior":
        same_ring = True
        shared_ring_coords = exterior_coords
    elif (
        ring_type1 == ring_type2 == "hole"
        and hole_idx1 is not None
        and hole_idx1 == hole_idx2
    ):
        # Both on the same hole ring — treat like same-ring analysis
        same_ring = True
        shared_ring_coords = ring1_coords

    if same_ring and shared_ring_coords is not None:
        n_ring = len(shared_ring_coords) - 1
        ring_idx1 = _find_nearest_vertex_index(shared_ring_coords, pt1)
        ring_idx2 = _find_nearest_vertex_index(shared_ring_coords, pt2)
        separation = min(abs(ring_idx2 - ring_idx1), n_ring - abs(ring_idx2 - ring_idx1))
        edge_angle_similarity = _compute_edge_angle_similarity(
            shared_ring_coords, ring_idx1, ring_idx2
        )
    else:
        # Points on different rings: use large separation to avoid
        # misclassifying as protrusion or near-self-intersection
        n_ring = n_exterior
        ring_idx1 = ring_idx2 = 0  # unused for cross-ring
        separation = max(n_exterior, 10)
        edge_angle_similarity = None

    narrow_extent = 0
    if same_ring and shared_ring_coords is not None and min_clearance > 0:
        narrow_extent = _compute_narrow_extent(
            shared_ring_coords, ring_idx1, ring_idx2, separation, min_clearance
        )

    tip_angle = None
    if same_ring and shared_ring_coords is not None and separation >= 2:
        tip_angle = _compute_wedge_tip_angle(shared_ring_coords, ring_idx1, ring_idx2, separation)

    return ClearanceContext(
        curvature=(curvature1, curvature2),
        separation=separation,
        vertex_count=n_exterior,
        ring_types=ring_types,
        edge_angle_similarity=edge_angle_similarity,
        narrow_extent=narrow_extent,
        tip_angle=tip_angle,
        hole_indices=(hole_idx1, hole_idx2),
    )


def _is_cross_ring(context: ClearanceContext) -> bool:
    """Return True if the clearance endpoints are on different rings.

    Same-ring cases (both exterior, or both on the *same* hole) return False
    so that the normal heuristic chain can diagnose them.  Cross-ring cases
    (exterior-to-hole, or different holes) return True.
    """
    if context.ring_types[0] == context.ring_types[1] == "exterior":
        return False
    if (
        context.ring_types[0] == context.ring_types[1] == "hole"
        and context.hole_indices[0] is not None
        and context.hole_indices[0] == context.hole_indices[1]
    ):
        return False
    # Different rings
    return True


# Minimum number of narrow vertex pairs beyond the clearance endpoints
# for a feature to be classified as a wedge rather than a simple spike.
_NARROW_WEDGE_MIN_EXTENT = 2


def _compute_narrow_extent(
    coords: np.ndarray,
    idx1: int,
    idx2: int,
    separation: int,
    min_clearance: float,
) -> int:
    """Count vertex pairs that are narrow when walking outward from the clearance bottleneck.

    Starting from the two endpoints of the minimum_clearance_line, walks
    outward along the ring in opposite directions (away from the short path
    between the endpoints). Counts consecutive vertex-pair steps where the
    distance between the walking indices remains below min_clearance.
    """
    n = len(coords) - 1  # closed ring
    if n < 4:
        return 0

    # Determine walk directions: away from the short path (toward the opening)
    fwd = (idx2 - idx1) % n
    if fwd == separation:
        dir1 = -1  # idx1 walks backward
        dir2 = +1  # idx2 walks forward
    else:
        dir1 = +1
        dir2 = -1

    max_steps = n // 2
    extent = 0

    for step in range(1, max_steps + 1):
        walk1 = (idx1 + dir1 * step) % n
        walk2 = (idx2 + dir2 * step) % n
        if walk1 == walk2:
            break
        dist = float(np.linalg.norm(coords[walk1][:2] - coords[walk2][:2]))
        if dist >= min_clearance:
            break
        extent += 1

    return extent


def _looks_like_narrow_wedge(
    context: ClearanceContext, _: float
) -> ClearanceIssue | None:
    """Detect narrow wedge intrusions (V-notches, tapered peninsulas).

    A narrow wedge is a protrusion-like feature where the narrowness extends
    well beyond the tip vertex.  Unlike a simple spike (1-2 narrow vertices),
    a wedge has a body of 2+ vertex pairs that are all narrower than
    min_clearance.  Detecting this allows removing the entire wedge in one
    operation instead of iteratively fixing vertex by vertex.

    Two detection paths:
    1. Width-based: close separation + moderate curvature + extended narrowness.
    2. Angle-based: acute tip angle, even when the opening is wider than
       min_clearance.  Catches long, gradually tapered features.
    """
    # Skip when endpoints are on different rings (cross-ring issues are
    # handled elsewhere). Same-hole cases are allowed through.
    if _is_cross_ring(context):
        return None

    # Path 1: Width-based detection (original)
    if (
        context.separation <= _CLOSE_VERTEX_SEPARATION
        and max(context.curvature) >= _PROTRUSION_MODERATE_ANGLE
        and context.narrow_extent >= _NARROW_WEDGE_MIN_EXTENT
    ):
        return ClearanceIssue.NARROW_WEDGE

    # Path 2: Angle-based detection for long wedges with acute tips
    if (
        context.tip_angle is not None
        and context.tip_angle < _ACUTE_TIP_ANGLE
        and context.separation >= 2
    ):
        return ClearanceIssue.NARROW_WEDGE

    return None


def _looks_like_protrusion(
    context: ClearanceContext, _: float
) -> ClearanceIssue | None:
    """Detect narrow spike-like protrusions.

    Protrusions are characterised by at least one endpoint of the clearance
    line sitting at a sharp turning angle (>135 degrees), indicating the
    polygon path reverses direction sharply.  A secondary check catches
    moderate turns (>90 degrees) when the two endpoints are very close along
    the ring, which is typical of narrow fingers.
    """
    if _is_cross_ring(context):
        return None
    curvature1, curvature2 = context.curvature
    if curvature1 > _PROTRUSION_SHARP_ANGLE or curvature2 > _PROTRUSION_SHARP_ANGLE:
        return ClearanceIssue.NARROW_PROTRUSION

    if context.separation <= _CLOSE_VERTEX_SEPARATION and (
        curvature1 > _PROTRUSION_MODERATE_ANGLE
        or curvature2 > _PROTRUSION_MODERATE_ANGLE
    ):
        return ClearanceIssue.NARROW_PROTRUSION
    return None


def _looks_like_near_self_intersection(
    context: ClearanceContext, _: float
) -> ClearanceIssue | None:
    """Detect near-self-intersection where the polygon path almost touches itself.

    When both clearance-line endpoints are close in ring distance *and* both
    have smooth turning angles (<=90 degrees), the geometry likely doubles
    back on itself without forming a sharp spike.  This differs from a
    protrusion, where at least one vertex has a sharp turn.
    """
    if _is_cross_ring(context):
        return None
    if context.separation <= _CLOSE_VERTEX_SEPARATION:
        curvature1, curvature2 = context.curvature
        if (
            curvature1 <= _SELF_INTERSECTION_MAX_ANGLE
            and curvature2 <= _SELF_INTERSECTION_MAX_ANGLE
        ):
            return ClearanceIssue.NEAR_SELF_INTERSECTION
    return None


def _looks_like_parallel_edges(
    context: ClearanceContext, _: float
) -> ClearanceIssue | None:
    """Detect parallel close edges (narrow channels, U-shapes, peninsulas).

    Parallel edges are characterised by:
    - Both endpoints on the exterior (not holes)
    - Endpoints far apart in ring distance (not a local feature)
    - Similar edge directions at both endpoints (edges run roughly parallel)

    When ``edge_angle_similarity`` is available, edges must be within
    ``_PARALLEL_EDGE_MAX_ANGLE`` degrees of each other.  Otherwise, falls back
    to a large ring-separation heuristic.
    """
    if _is_cross_ring(context):
        return None
    if context.vertex_count <= 0:
        return None
    if context.separation <= _CLOSE_VERTEX_SEPARATION:
        return None

    if context.edge_angle_similarity is not None:
        if context.edge_angle_similarity < _PARALLEL_EDGE_MAX_ANGLE:
            return ClearanceIssue.PARALLEL_CLOSE_EDGES
    elif context.separation >= context.vertex_count // 3:
        # Fallback when edge angles unavailable: use original separation heuristic
        return ClearanceIssue.PARALLEL_CLOSE_EDGES
    return None


def _default_clearance_issue(_: ClearanceContext, __: float) -> ClearanceIssue:
    """Fallback diagnosis when no specific pattern is detected.

    Returns ``NARROW_PASSAGE`` as the most general clearance issue type.
    The narrow-passage fix strategy (widen) is the safest default because
    it works on a broad range of clearance problems without making
    assumptions about the geometry's structure.
    """
    return ClearanceIssue.NARROW_PASSAGE


def diagnose_clearance(geometry: Polygon, min_clearance: float) -> ClearanceDiagnosis:
    """Diagnose clearance issues in a polygon without modifying it.

    Args:
        geometry: Input polygon to diagnose.
        min_clearance: Target minimum clearance value.

    Returns:
        ClearanceDiagnosis with issue type, current clearance, and recommended fix.

    Raises:
        TypeError: If geometry is not a Polygon.
    """
    if not isinstance(geometry, Polygon):
        raise TypeError(f"Expected Polygon, got {type(geometry).__name__}")

    current_clearance = geometry.minimum_clearance
    meets_requirement = current_clearance >= min_clearance - _CLEARANCE_TOLERANCE
    ratio = current_clearance / min_clearance if min_clearance > 0 else float("inf")

    try:
        clearance_line = shapely.minimum_clearance_line(geometry)
    except (GEOSException, ValueError):
        clearance_line = None

    if meets_requirement:
        return ClearanceDiagnosis(
            issue=ClearanceIssue.NONE,
            meets_requirement=True,
            current_clearance=current_clearance,
            clearance_ratio=ratio,
            clearance_line=clearance_line,
            recommended_fix=RECOMMENDED_FIXES[ClearanceIssue.NONE],
        )

    issue, context = _diagnose_clearance_issue(geometry, min_clearance)
    recommended = RECOMMENDED_FIXES.get(
        issue, RECOMMENDED_FIXES[ClearanceIssue.UNKNOWN]
    )
    return ClearanceDiagnosis(
        issue=issue,
        meets_requirement=False,
        current_clearance=current_clearance,
        clearance_ratio=ratio,
        clearance_line=clearance_line,
        recommended_fix=recommended,
        context=context,
    )


__all__ = [
    "fix_clearance",
    "diagnose_clearance",
    "ClearanceIssue",
    "ClearanceDiagnosis",
    "ClearanceFixSummary",
]
