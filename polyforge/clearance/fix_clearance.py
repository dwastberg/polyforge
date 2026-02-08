"""Automatic clearance detection and fixing.

This module provides an intelligent function that automatically diagnoses
clearance issues and applies the most appropriate fixing strategy.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, LinearRing
from shapely.geometry.base import BaseGeometry

from polyforge.ops.clearance import (
    fix_hole_too_close,
    fix_narrow_protrusion,
    remove_narrow_protrusions,
    fix_narrow_passage,
    fix_near_self_intersection,
    fix_parallel_close_edges,
    _find_nearest_vertex_index,
    _calculate_curvature_at_vertex,
)
from polyforge.core.geometry_utils import to_single_polygon
from polyforge.core.types import HoleStrategy, PassageStrategy, IntersectionStrategy
from polyforge.core.iterative_utils import iterative_improve
from shapely.errors import GEOSException
from polyforge.metrics import _safe_clearance

# --- Clearance diagnosis thresholds ---
# Turning angle (degrees) above which a vertex is considered a sharp reversal,
# characteristic of spike-like protrusions.  135° means the path turns more
# than 45° past a right-angle bend.
_PROTRUSION_SHARP_ANGLE = 135.0

# Turning angle threshold for detecting protrusions when vertices are very
# close together (separation <= _CLOSE_VERTEX_SEPARATION).  A 90° turn
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


class ClearanceIssue(Enum):
    """Enumerates the types of clearance problems we can detect."""

    NONE = "none"
    HOLE_TOO_CLOSE = "hole_too_close"
    NARROW_PROTRUSION = "narrow_protrusion"
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
    clearance_line: Optional[LineString]
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
    history: List[ClearanceIssue]


def fix_clearance(
    geometry: Polygon,
    min_clearance: float,
    max_iterations: int = 10,
    min_area_ratio: float = 0.9,
    return_diagnosis: bool = False,
) -> Union[Polygon, Tuple[Polygon, ClearanceFixSummary]]:
    """Automatically diagnose and fix low minimum clearance in a polygon."""
    if not isinstance(geometry, Polygon):
        raise TypeError(f"Expected Polygon, got {type(geometry).__name__}")

    if min_area_ratio <= 0 or min_area_ratio > 1.0:
        raise ValueError("min_area_ratio must be in (0, 1].")

    initial_clearance = geometry.minimum_clearance
    original_area = geometry.area
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

    issue_history: List[ClearanceIssue] = []
    best_valid = geometry

    def improve(poly: Polygon, target: float) -> Optional[Polygon]:
        diagnosis = diagnose_clearance(poly, target)
        issue_history.append(diagnosis.issue)
        if diagnosis.issue == ClearanceIssue.NONE:
            return None
        candidate = _apply_clearance_strategy(poly, target, diagnosis)
        if candidate is None or not candidate.is_valid or candidate.is_empty:
            return None
        nonlocal best_valid
        if candidate.area < min_area_ratio * original_area:
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

    final_clearance = _safe_clearance(improved) or 0.0
    final_area_ratio = improved.area / original_area if original_area > 0 else float("inf")
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


StrategyFunc = Callable[[Polygon, float, ClearanceDiagnosis], Optional[Polygon]]

RECOMMENDED_FIXES: Dict[ClearanceIssue, str] = {
    ClearanceIssue.NONE: "none",
    ClearanceIssue.HOLE_TOO_CLOSE: "fix_hole_too_close",
    ClearanceIssue.NARROW_PROTRUSION: "remove_narrow_protrusions",
    ClearanceIssue.NARROW_PASSAGE: "fix_narrow_passage",
    ClearanceIssue.NEAR_SELF_INTERSECTION: "fix_near_self_intersection",
    ClearanceIssue.PARALLEL_CLOSE_EDGES: "fix_parallel_close_edges",
    ClearanceIssue.UNKNOWN: "fix_narrow_passage",
}

STRATEGY_REGISTRY: Dict[ClearanceIssue, StrategyFunc] = {}


def _strategy(issue: ClearanceIssue) -> StrategyFunc:
    return STRATEGY_REGISTRY.get(issue, _strategy_default)


def _apply_clearance_strategy(
    geometry: Polygon,
    min_clearance: float,
    diagnosis: ClearanceDiagnosis,
) -> Optional[Polygon]:
    handler = _strategy(diagnosis.issue)
    candidate = handler(geometry, min_clearance, diagnosis)
    return _normalize_polygon(candidate)


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
) -> Optional[Polygon]:
    fixed = fix_hole_too_close(geometry, min_clearance, strategy=HoleStrategy.REMOVE)
    return fixed


@_register_strategy(ClearanceIssue.NARROW_PROTRUSION)
def _strategy_narrow_protrusion(
    geometry: Polygon,
    min_clearance: float,
    diagnosis: ClearanceDiagnosis,
) -> Optional[Polygon]:
    baseline = diagnosis.current_clearance
    first_pass = remove_narrow_protrusions(geometry, aspect_ratio_threshold=10.0)
    if first_pass.is_valid and (_safe_clearance(first_pass) or 0.0) > baseline:
        return first_pass
    return fix_narrow_protrusion(geometry, min_clearance)


@_register_strategy(ClearanceIssue.NARROW_PASSAGE)
def _strategy_narrow_passage(
    geometry: Polygon,
    min_clearance: float,
    _: ClearanceDiagnosis,
) -> Optional[Polygon]:
    return fix_narrow_passage(geometry, min_clearance, strategy=PassageStrategy.WIDEN)


@_register_strategy(ClearanceIssue.NEAR_SELF_INTERSECTION)
def _strategy_near_self_intersection(
    geometry: Polygon,
    min_clearance: float,
    _: ClearanceDiagnosis,
) -> Optional[Polygon]:
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
) -> Optional[Polygon]:
    return fix_parallel_close_edges(
        geometry,
        min_clearance,
        strategy=IntersectionStrategy.SIMPLIFY,
    )


def _strategy_default(
    geometry: Polygon,
    min_clearance: float,
    _: ClearanceDiagnosis,
) -> Optional[Polygon]:
    return fix_narrow_passage(geometry, min_clearance, strategy=PassageStrategy.WIDEN)


def _normalize_polygon(candidate: Optional[BaseGeometry]) -> Optional[Polygon]:
    if candidate is None:
        return None
    if candidate.is_empty or not candidate.is_valid:
        return None
    polygon = to_single_polygon(candidate)
    return polygon if polygon.is_valid and not polygon.is_empty else None


def _diagnose_clearance_issue(
    geometry: Polygon,
    min_clearance: float
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

    context = _build_clearance_context(geometry)
    if context is None:
        return ClearanceIssue.UNKNOWN

    if "hole" in context.ring_types:
        return ClearanceIssue.HOLE_TOO_CLOSE

    for heuristic in (
        _looks_like_protrusion,
        _looks_like_near_self_intersection,
        _looks_like_parallel_edges,
        _default_clearance_issue,
    ):
        issue = heuristic(context, min_clearance)
        if issue is not None:
            return issue
    return ClearanceIssue.UNKNOWN


def _classify_ring_types(geometry: Polygon, pt1: Tuple[float, float], pt2: Tuple[float, float]) -> Tuple[str, str]:
    """Return ring type ('exterior' or 'hole') for each endpoint of the clearance line."""
    def classify(point: Tuple[float, float]) -> Tuple[str, Optional[int]]:
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

    t1, _ = classify(pt1)
    t2, _ = classify(pt2)
    return t1, t2


def _has_close_hole(geometry: Polygon, min_clearance: float) -> bool:
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
) -> Optional[float]:
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


def _build_clearance_context(geometry: Polygon) -> Optional["ClearanceContext"]:
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
    exterior_coords = np.array(geometry.exterior.coords)
    idx1 = _find_nearest_vertex_index(exterior_coords, pt1)
    idx2 = _find_nearest_vertex_index(exterior_coords, pt2)
    curvature1 = _calculate_curvature_at_vertex(exterior_coords, idx1)
    curvature2 = _calculate_curvature_at_vertex(exterior_coords, idx2)
    n = len(exterior_coords) - 1
    separation = min(abs(idx2 - idx1), n - abs(idx2 - idx1))
    ring_types = _classify_ring_types(geometry, pt1, pt2)
    edge_angle_similarity = _compute_edge_angle_similarity(exterior_coords, idx1, idx2)

    return ClearanceContext(
        curvature=(curvature1, curvature2),
        separation=separation,
        vertex_count=n,
        ring_types=ring_types,
        edge_angle_similarity=edge_angle_similarity,
    )


@dataclass
class ClearanceContext:
    curvature: Tuple[float, float]
    separation: int
    vertex_count: int
    ring_types: Tuple[str, str]
    edge_angle_similarity: Optional[float] = None


def _looks_like_protrusion(context: ClearanceContext, _: float) -> Optional[ClearanceIssue]:
    """Detect narrow spike-like protrusions.

    Protrusions are characterised by at least one endpoint of the clearance
    line sitting at a sharp turning angle (>135 degrees), indicating the
    polygon path reverses direction sharply.  A secondary check catches
    moderate turns (>90 degrees) when the two endpoints are very close along
    the ring, which is typical of narrow fingers.
    """
    if "hole" in context.ring_types:
        return None
    curvature1, curvature2 = context.curvature
    if curvature1 > _PROTRUSION_SHARP_ANGLE or curvature2 > _PROTRUSION_SHARP_ANGLE:
        return ClearanceIssue.NARROW_PROTRUSION

    if context.separation <= _CLOSE_VERTEX_SEPARATION and (
        curvature1 > _PROTRUSION_MODERATE_ANGLE or curvature2 > _PROTRUSION_MODERATE_ANGLE
    ):
        return ClearanceIssue.NARROW_PROTRUSION
    return None


def _looks_like_near_self_intersection(context: ClearanceContext, _: float) -> Optional[ClearanceIssue]:
    """Detect near-self-intersection where the polygon path almost touches itself.

    When both clearance-line endpoints are close in ring distance *and* both
    have smooth turning angles (<=90 degrees), the geometry likely doubles
    back on itself without forming a sharp spike.  This differs from a
    protrusion, where at least one vertex has a sharp turn.
    """
    if "hole" in context.ring_types:
        return None
    if context.separation <= _CLOSE_VERTEX_SEPARATION:
        curvature1, curvature2 = context.curvature
        if curvature1 <= _SELF_INTERSECTION_MAX_ANGLE and curvature2 <= _SELF_INTERSECTION_MAX_ANGLE:
            return ClearanceIssue.NEAR_SELF_INTERSECTION
    return None


def _looks_like_parallel_edges(context: ClearanceContext, _: float) -> Optional[ClearanceIssue]:
    """Detect parallel close edges (narrow channels, U-shapes, peninsulas).

    Parallel edges are characterised by:
    - Both endpoints on the exterior (not holes)
    - Endpoints far apart in ring distance (not a local feature)
    - Similar edge directions at both endpoints (edges run roughly parallel)

    When ``edge_angle_similarity`` is available, edges must be within
    ``_PARALLEL_EDGE_MAX_ANGLE`` degrees of each other.  Otherwise, falls back
    to a large ring-separation heuristic.
    """
    if "hole" in context.ring_types:
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


def diagnose_clearance(
    geometry: Polygon,
    min_clearance: float
) -> ClearanceDiagnosis:
    """Diagnose clearance issues without fixing them."""
    if not isinstance(geometry, Polygon):
        raise TypeError(f"Expected Polygon, got {type(geometry).__name__}")

    current_clearance = geometry.minimum_clearance
    meets_requirement = current_clearance >= min_clearance
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
    recommended = RECOMMENDED_FIXES.get(issue, RECOMMENDED_FIXES[ClearanceIssue.UNKNOWN])
    return ClearanceDiagnosis(
        issue=issue,
        meets_requirement=False,
        current_clearance=current_clearance,
        clearance_ratio=ratio,
        clearance_line=clearance_line,
        recommended_fix=recommended,
    )


__all__ = [
    'fix_clearance',
    'diagnose_clearance',
    'ClearanceIssue',
    'ClearanceDiagnosis',
    'ClearanceFixSummary',
]
