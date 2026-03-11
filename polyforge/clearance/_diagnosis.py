"""Clearance diagnosis: detecting and classifying clearance issues."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import shapely
from shapely.geometry import Polygon, Point, LineString, LinearRing
from shapely.errors import GEOSException

from polyforge.ops.clearance.utils import (
    _find_nearest_vertex_index,
    _calculate_curvature_at_vertex,
    _compute_wedge_tip_angle,
)

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

# Minimum number of narrow vertex pairs beyond the clearance endpoints
# for a feature to be classified as a wedge rather than a simple spike.
_NARROW_WEDGE_MIN_EXTENT = 2


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
