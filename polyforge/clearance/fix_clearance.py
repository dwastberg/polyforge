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

from .holes import fix_hole_too_close
from .protrusions import fix_narrow_protrusion
from .remove_protrusions import remove_narrow_protrusions
from .passages import fix_narrow_passage, fix_near_self_intersection, fix_parallel_close_edges
from .utils import _find_nearest_vertex_index, _calculate_curvature_at_vertex
from polyforge.core.types import HoleStrategy, PassageStrategy, IntersectionStrategy, EdgeStrategy
from polyforge.core.iterative_utils import iterative_improve


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
    iterations: int
    issue: ClearanceIssue
    fixed: bool
    history: List[ClearanceIssue]


def fix_clearance(
    geometry: Polygon,
    min_clearance: float,
    max_iterations: int = 10,
    return_diagnosis: bool = False,
) -> Union[Polygon, Tuple[Polygon, ClearanceFixSummary]]:
    """Automatically diagnose and fix low minimum clearance in a polygon."""
    if not isinstance(geometry, Polygon):
        raise TypeError(f"Expected Polygon, got {type(geometry).__name__}")

    initial_clearance = geometry.minimum_clearance
    summary = ClearanceFixSummary(
        initial_clearance=initial_clearance,
        final_clearance=initial_clearance,
        iterations=0,
        issue=ClearanceIssue.NONE,
        fixed=initial_clearance >= min_clearance,
        history=[ClearanceIssue.NONE],
    )

    if summary.fixed:
        return (geometry, summary) if return_diagnosis else geometry

    issue_history: List[ClearanceIssue] = []

    def improve(poly: Polygon, target: float) -> Optional[Polygon]:
        diagnosis = diagnose_clearance(poly, target)
        issue_history.append(diagnosis.issue)
        if diagnosis.issue == ClearanceIssue.NONE:
            return None
        return _apply_clearance_strategy(poly, target, diagnosis)

    metric = lambda poly: _safe_clearance(poly)  # noqa: E731
    improved = iterative_improve(
        geometry,
        target_value=min_clearance,
        improve_func=improve,
        metric_func=metric,
        max_iterations=max_iterations,
    )

    final_clearance = _safe_clearance(improved)
    final_diag = diagnose_clearance(improved, min_clearance)
    summary = ClearanceFixSummary(
        initial_clearance=initial_clearance,
        final_clearance=final_clearance,
        iterations=len(issue_history),
        issue=final_diag.issue,
        fixed=final_diag.meets_requirement,
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
    if first_pass.is_valid and _safe_clearance(first_pass) > baseline:
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
        strategy=EdgeStrategy.SIMPLIFY,
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
    if isinstance(candidate, Polygon):
        return candidate
    if isinstance(candidate, MultiPolygon) and candidate.geoms:
        largest = max(candidate.geoms, key=lambda g: g.area)
        return largest if isinstance(largest, Polygon) else None
    return None


def _safe_clearance(geometry: Polygon) -> float:
    try:
        return float(geometry.minimum_clearance)
    except Exception:
        return 0.0


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
    # Check for holes
    if geometry.interiors:
        # Check if any hole is too close to exterior
        exterior_ring = LinearRing(geometry.exterior.coords)

        for hole in geometry.interiors:
            hole_ring = LinearRing(hole.coords)
            distance = hole_ring.distance(exterior_ring)

            if distance < min_clearance:
                return ClearanceIssue.HOLE_TOO_CLOSE

    # Get clearance line to analyze the issue
    try:
        clearance_line = shapely.minimum_clearance_line(geometry)
    except Exception:
        return ClearanceIssue.UNKNOWN

    if clearance_line.is_empty:
        return ClearanceIssue.UNKNOWN

    # Get endpoints of clearance line
    coords_2d = np.array(clearance_line.coords)
    if len(coords_2d) < 2:
        return ClearanceIssue.UNKNOWN

    pt1 = coords_2d[0]
    pt2 = coords_2d[1]

    # Convert to array
    exterior_coords = np.array(geometry.exterior.coords)

    # Find which vertices the clearance points are near
    idx1 = _find_nearest_vertex_index(exterior_coords, pt1)
    idx2 = _find_nearest_vertex_index(exterior_coords, pt2)

    # Calculate curvature at these vertices
    curvature1 = _calculate_curvature_at_vertex(exterior_coords, idx1)
    curvature2 = _calculate_curvature_at_vertex(exterior_coords, idx2)

    # Determine vertex separation along boundary
    n = len(exterior_coords) - 1
    separation = min(abs(idx2 - idx1), n - abs(idx2 - idx1))

    # Decision logic:
    # 1. High curvature at one endpoint -> likely protrusion/spike
    # 2. Points very close along boundary (1-3 vertices) -> protrusion
    # 3. Points on opposite sides but close -> narrow passage
    # 4. Points adjacent with very close distance -> near self-intersection
    # 5. Points far apart along boundary -> parallel close edges

    # Threshold for "sharp" turn (likely protrusion)
    sharp_angle_threshold = 135.0  # degrees

    # Check for narrow protrusion/spike
    if curvature1 > sharp_angle_threshold or curvature2 > sharp_angle_threshold:
        return ClearanceIssue.NARROW_PROTRUSION

    # Check for very close vertices (protrusion or near-self-intersection)
    if separation <= 3:
        # Very close along boundary
        if curvature1 > 90 or curvature2 > 90:
            return ClearanceIssue.NARROW_PROTRUSION
        else:
            return ClearanceIssue.NEAR_SELF_INTERSECTION

    # Check for narrow passage (medium separation)
    if 3 < separation < n // 3:
        # Points are somewhat separated - likely narrow passage/neck
        return ClearanceIssue.NARROW_PASSAGE

    # Check for parallel close edges (large separation)
    if separation >= n // 3:
        return ClearanceIssue.PARALLEL_CLOSE_EDGES

    # Default to narrow passage (most common)
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
    except Exception:
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
