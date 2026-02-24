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
# increase polygon area (e.g., by filling in concavities).
_AREA_GROWTH_TOLERANCE = 1.01


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
    """

    issue: ClearanceIssue
    meets_requirement: bool
    current_clearance: float
    clearance_ratio: float
    clearance_line: LineString | None
    recommended_fix: str

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

    # --- Phase 2: Point-based fixes (diagnosis-driven iteration) ---
    # For cases where erosion-dilation is too aggressive (would lose too much area)
    # or doesn't fully solve the problem (e.g., holes, single protrusions).
    best_valid = geometry

    def improve(poly: Polygon, target: float) -> Polygon | None:
        diagnosis = diagnose_clearance(poly, target)
        issue_history.append(diagnosis.issue)
        if diagnosis.issue == ClearanceIssue.NONE:
            return None
        candidate = _apply_clearance_strategy(poly, target, diagnosis)
        # If standard strategy failed, try hole ring simplification for
        # same-hole self-clearance issues.
        if candidate is None or not candidate.is_valid or candidate.is_empty:
            candidate = _try_hole_ring_fix(poly, target)
        if candidate is None or not candidate.is_valid or candidate.is_empty:
            return None
        nonlocal best_valid
        if candidate.area < min_area_ratio * original_area:
            return None
        # Reject candidates that grow the exterior ring beyond a small tolerance.
        # We check exterior area (not polygon area) so that hole fixes—which
        # legitimately increase polygon area by shrinking holes—are not blocked.
        if Polygon(candidate.exterior).area > original_exterior_area * _AREA_GROWTH_TOLERANCE:
            return None
        cand_clearance = _safe_clearance(candidate) or 0.0
        best_clearance = _safe_clearance(best_valid) or 0.0
        if cand_clearance > best_clearance:
            best_valid = candidate
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
        improved = best_valid
    if improved.area < min_area_ratio * original_area:
        improved = best_valid

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
    # Fallback: blunt vertex removal for micro-features that targeted fix can't resolve
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
    _: ClearanceDiagnosis,
) -> Polygon | None:
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


def _try_hole_ring_fix(geometry: Polygon, min_clearance: float) -> Polygon | None:
    """Try to fix clearance by simplifying the offending hole ring.

    Checks if the minimum_clearance bottleneck is on a single hole ring.
    If so, simplifies that hole to resolve the issue.
    """
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
    candidate = _simplify_hole_ring(geometry, context.hole_indices[0], min_clearance)
    if candidate is not None and candidate.is_valid:
        new_clearance = _safe_clearance(candidate) or 0.0
        if new_clearance > current_clearance:
            return candidate
    return None


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


def _normalize_polygon(candidate: BaseGeometry | None) -> Polygon | None:
    if candidate is None:
        return None
    if candidate.is_empty or not candidate.is_valid:
        return None
    polygon = to_single_polygon(candidate)
    return polygon if polygon.is_valid and not polygon.is_empty else None


def _diagnose_clearance_issue(
    geometry: Polygon, min_clearance: float
) -> ClearanceIssue:
    """Diagnose the type of clearance issue in a polygon.

    Examines the geometry's minimum_clearance_line and surrounding geometry
    to determine what type of issue is causing low clearance.

    Args:
        geometry: Input polygon
        min_clearance: Target minimum clearance

    Returns:
        A :class:`ClearanceIssue` describing the detected problem.
    """
    if _has_close_hole(geometry, min_clearance):
        return ClearanceIssue.HOLE_TOO_CLOSE

    context = _build_clearance_context(geometry, min_clearance)
    if context is None:
        return ClearanceIssue.UNKNOWN

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
            return ClearanceIssue.HOLE_TOO_CLOSE

    for heuristic in (
        _looks_like_narrow_wedge,
        _looks_like_protrusion,
        _looks_like_near_self_intersection,
        _looks_like_parallel_edges,
        _default_clearance_issue,
    ):
        issue = heuristic(context, min_clearance)
        if issue is not None:
            return issue
    return ClearanceIssue.UNKNOWN


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
        # Recompute indices on the shared ring
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
        separation = max(n_exterior, 10)
        edge_angle_similarity = None

    narrow_extent = 0
    if same_ring and shared_ring_coords is not None and min_clearance > 0:
        ring_idx1 = _find_nearest_vertex_index(shared_ring_coords, pt1)
        ring_idx2 = _find_nearest_vertex_index(shared_ring_coords, pt2)
        narrow_extent = _compute_narrow_extent(
            shared_ring_coords, ring_idx1, ring_idx2, separation, min_clearance
        )

    tip_angle = None
    if same_ring and shared_ring_coords is not None and separation >= 2:
        ring_idx1 = _find_nearest_vertex_index(shared_ring_coords, pt1)
        ring_idx2 = _find_nearest_vertex_index(shared_ring_coords, pt2)
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


@dataclass
class ClearanceContext:
    curvature: tuple[float, float]
    separation: int
    vertex_count: int
    ring_types: tuple[str, str]
    edge_angle_similarity: float | None = None
    narrow_extent: int = 0
    tip_angle: float | None = None
    hole_indices: tuple[int | None, int | None] = (None, None)


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

    issue = _diagnose_clearance_issue(geometry, min_clearance)
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
    )


__all__ = [
    "fix_clearance",
    "diagnose_clearance",
    "ClearanceIssue",
    "ClearanceDiagnosis",
    "ClearanceFixSummary",
]
