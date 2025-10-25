"""Automatic clearance detection and fixing.

This module provides an intelligent function that automatically diagnoses
clearance issues and applies the most appropriate fixing strategy.
"""

from typing import Union, Dict, List, Tuple
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, LinearRing

from .holes import fix_hole_too_close
from .protrusions import fix_narrow_protrusion
from .remove_protrusions import remove_narrow_protrusions
from .passages import fix_narrow_passage, fix_near_self_intersection, fix_parallel_close_edges
from .utils import _find_nearest_vertex_index, _calculate_curvature_at_vertex


def fix_clearance(
    geometry: Polygon,
    min_clearance: float,
    max_iterations: int = 10,
    return_diagnosis: bool = False
) -> Union[Polygon, MultiPolygon, Tuple[Union[Polygon, MultiPolygon], Dict]]:
    """Automatically diagnose and fix low minimum clearance in a polygon.

    This function examines the polygon's minimum_clearance and, if it's below
    the threshold, automatically determines the type of clearance issue and
    applies the most appropriate fixing strategy.

    Args:
        geometry: Input polygon to check and fix
        min_clearance: Target minimum clearance value
        max_iterations: Maximum number of fixing iterations (default: 10)
        return_diagnosis: If True, return (fixed_geometry, diagnosis_info)

    Returns:
        Fixed geometry, or (fixed_geometry, diagnosis_info) if return_diagnosis=True

    Diagnosis info dict contains:
        - 'initial_clearance': Original minimum clearance
        - 'final_clearance': Clearance after fixing
        - 'issue_type': Detected issue type
        - 'fix_applied': Name of fix function used
        - 'iterations': Number of iterations performed
        - 'fixed': Whether clearance was successfully improved

    Examples:
        >>> # Simple usage
        >>> poly = Polygon([(0, 0), (10, 0), (10, 1), (9.9, 1.5), (10, 3), (10, 10), (0, 10)])
        >>> fixed = fix_clearance(poly, min_clearance=1.0)
        >>> fixed.minimum_clearance >= 1.0
        True

        >>> # With diagnosis
        >>> fixed, info = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)
        >>> print(f"Issue: {info['issue_type']}, Fix: {info['fix_applied']}")
        Issue: narrow_passage, Fix: fix_narrow_passage

    Note:
        The function iterates until the clearance meets the threshold or
        max_iterations is reached. Different issue types are detected and
        the most appropriate fix is applied:
        - Holes too close to exterior
        - Narrow protrusions/spikes
        - Narrow passages/necks
        - Near self-intersections
        - Parallel close edges
    """
    if not isinstance(geometry, Polygon):
        raise TypeError(f"Expected Polygon, got {type(geometry).__name__}")

    # Check initial clearance
    initial_clearance = geometry.minimum_clearance

    if initial_clearance >= min_clearance:
        # Already meets requirement
        diagnosis = {
            'initial_clearance': initial_clearance,
            'final_clearance': initial_clearance,
            'issue_type': 'none',
            'fix_applied': 'none',
            'iterations': 0,
            'fixed': True
        }
        if return_diagnosis:
            return geometry, diagnosis
        return geometry

    # Iterative fixing
    current_geometry = geometry
    iteration = 0
    best_geometry = geometry
    best_clearance = initial_clearance
    issue_types_tried = []

    while iteration < max_iterations:
        iteration += 1
        current_clearance = current_geometry.minimum_clearance

        # Check if we've reached target
        if current_clearance >= min_clearance:
            diagnosis = {
                'initial_clearance': initial_clearance,
                'final_clearance': current_clearance,
                'issue_type': issue_types_tried[-1] if issue_types_tried else 'unknown',
                'fix_applied': issue_types_tried[-1] if issue_types_tried else 'none',
                'iterations': iteration,
                'fixed': True
            }
            if return_diagnosis:
                return current_geometry, diagnosis
            return current_geometry

        # Diagnose the issue
        issue_type = _diagnose_clearance_issue(current_geometry, min_clearance)
        issue_types_tried.append(issue_type)

        # Apply appropriate fix
        try:
            fixed_geometry = _apply_clearance_fix(
                current_geometry,
                min_clearance,
                issue_type
            )

            # Check if fix improved clearance
            if fixed_geometry.is_valid and not fixed_geometry.is_empty:
                fixed_clearance = fixed_geometry.minimum_clearance

                if fixed_clearance > current_clearance:
                    # Improvement - continue
                    current_geometry = fixed_geometry
                    best_geometry = fixed_geometry
                    best_clearance = fixed_clearance
                else:
                    # No improvement - try different strategy
                    # Fall back to more aggressive fixes
                    if issue_type != 'narrow_passage_widen':
                        fixed_geometry = _apply_clearance_fix(
                            current_geometry,
                            min_clearance,
                            'narrow_passage_widen'
                        )
                        if fixed_geometry.is_valid and not fixed_geometry.is_empty:
                            fixed_clearance = fixed_geometry.minimum_clearance
                            if fixed_clearance > current_clearance:
                                current_geometry = fixed_geometry
                                best_geometry = fixed_geometry
                                best_clearance = fixed_clearance
                                continue

                    # Still no improvement - stop iterating
                    break
            else:
                # Fix produced invalid geometry - stop
                break

        except Exception:
            # Fix failed - stop
            break

    # Return best result found
    diagnosis = {
        'initial_clearance': initial_clearance,
        'final_clearance': best_clearance,
        'issue_type': issue_types_tried[0] if issue_types_tried else 'unknown',
        'fix_applied': issue_types_tried[-1] if issue_types_tried else 'none',
        'iterations': iteration,
        'fixed': best_clearance >= min_clearance
    }

    if return_diagnosis:
        return best_geometry, diagnosis
    return best_geometry


def _diagnose_clearance_issue(
    geometry: Polygon,
    min_clearance: float
) -> str:
    """Diagnose the type of clearance issue in a polygon.

    Examines the geometry's minimum_clearance_line and surrounding geometry
    to determine what type of issue is causing low clearance.

    Args:
        geometry: Input polygon
        min_clearance: Target minimum clearance

    Returns:
        Issue type string, one of:
        - 'hole_too_close': Interior hole too close to exterior
        - 'narrow_protrusion': Narrow spike or protrusion
        - 'narrow_passage': Narrow neck/passage (hourglass shape)
        - 'near_self_intersection': Nearly self-intersecting
        - 'parallel_close_edges': Parallel edges running close together
        - 'unknown': Could not determine issue type
    """
    # Check for holes
    if geometry.interiors:
        # Check if any hole is too close to exterior
        exterior_ring = LinearRing(geometry.exterior.coords)

        for hole in geometry.interiors:
            hole_ring = LinearRing(hole.coords)
            distance = hole_ring.distance(exterior_ring)

            if distance < min_clearance:
                return 'hole_too_close'

    # Get clearance line to analyze the issue
    try:
        clearance_line = shapely.minimum_clearance_line(geometry)
    except Exception:
        return 'unknown'

    if clearance_line.is_empty:
        return 'unknown'

    # Get endpoints of clearance line
    coords_2d = np.array(clearance_line.coords)
    if len(coords_2d) < 2:
        return 'unknown'

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
        return 'narrow_protrusion'

    # Check for very close vertices (protrusion or near-self-intersection)
    if separation <= 3:
        # Very close along boundary
        if curvature1 > 90 or curvature2 > 90:
            return 'narrow_protrusion'
        else:
            return 'near_self_intersection'

    # Check for narrow passage (medium separation)
    if 3 < separation < n // 3:
        # Points are somewhat separated - likely narrow passage/neck
        return 'narrow_passage'

    # Check for parallel close edges (large separation)
    if separation >= n // 3:
        return 'parallel_close_edges'

    # Default to narrow passage (most common)
    return 'narrow_passage'


def _apply_clearance_fix(
    geometry: Polygon,
    min_clearance: float,
    issue_type: str
) -> Union[Polygon, MultiPolygon]:
    """Apply the appropriate fix for a diagnosed clearance issue.

    Args:
        geometry: Input polygon with clearance issue
        min_clearance: Target minimum clearance
        issue_type: Type of issue (from _diagnose_clearance_issue)

    Returns:
        Fixed geometry
    """
    if issue_type == 'hole_too_close':
        # Remove holes that are too close
        return fix_hole_too_close(geometry, min_clearance, strategy='remove')

    elif issue_type == 'narrow_protrusion':
        # Try to remove narrow protrusions
        # First try the iterative removal approach
        result = remove_narrow_protrusions(geometry, aspect_ratio_threshold=10.0)
        if result.is_valid and result.minimum_clearance > geometry.minimum_clearance:
            return result

        # Fall back to single protrusion fix
        return fix_narrow_protrusion(geometry, min_clearance)

    elif issue_type == 'narrow_passage':
        # Widen the narrow passage
        return fix_narrow_passage(geometry, min_clearance, strategy='widen')

    elif issue_type == 'narrow_passage_widen':
        # Explicit widen strategy (used as fallback)
        return fix_narrow_passage(geometry, min_clearance, strategy='widen')

    elif issue_type == 'near_self_intersection':
        # Fix near self-intersection
        return fix_near_self_intersection(geometry, min_clearance, strategy='simplify')

    elif issue_type == 'parallel_close_edges':
        # Fix parallel edges running close together
        return fix_parallel_close_edges(geometry, min_clearance)

    else:
        # Unknown issue - try general widening
        return fix_narrow_passage(geometry, min_clearance, strategy='widen')


def diagnose_clearance(
    geometry: Polygon,
    min_clearance: float
) -> Dict[str, any]:
    """Diagnose clearance issues without fixing them.

    This function provides detailed diagnostic information about a polygon's
    clearance without modifying the geometry.

    Args:
        geometry: Input polygon to diagnose
        min_clearance: Target minimum clearance for comparison

    Returns:
        Dictionary with diagnostic information:
        - 'current_clearance': Current minimum clearance value
        - 'meets_requirement': Whether clearance meets min_clearance
        - 'clearance_ratio': current_clearance / min_clearance
        - 'has_issues': Whether clearance is below threshold
        - 'issue_type': Detected issue type (if has_issues)
        - 'clearance_line': LineString showing location of minimum clearance
        - 'recommended_fix': Recommended fix function name

    Examples:
        >>> poly = Polygon([(0, 0), (10, 0), (10, 1), (9.9, 1.5), (10, 3), (10, 10), (0, 10)])
        >>> info = diagnose_clearance(poly, min_clearance=1.0)
        >>> print(f"Issue: {info['issue_type']}")
        >>> print(f"Recommended: {info['recommended_fix']}")
        Issue: narrow_passage
        Recommended: fix_narrow_passage
    """
    if not isinstance(geometry, Polygon):
        raise TypeError(f"Expected Polygon, got {type(geometry).__name__}")

    current_clearance = geometry.minimum_clearance
    meets_requirement = current_clearance >= min_clearance
    has_issues = not meets_requirement

    result = {
        'current_clearance': current_clearance,
        'meets_requirement': meets_requirement,
        'clearance_ratio': current_clearance / min_clearance if min_clearance > 0 else float('inf'),
        'has_issues': has_issues,
    }

    if has_issues:
        # Diagnose the issue
        issue_type = _diagnose_clearance_issue(geometry, min_clearance)
        result['issue_type'] = issue_type

        # Get clearance line location
        try:
            clearance_line = shapely.minimum_clearance_line(geometry)
            result['clearance_line'] = clearance_line
        except Exception:
            result['clearance_line'] = None

        # Recommend fix function
        fix_mapping = {
            'hole_too_close': 'fix_hole_too_close',
            'narrow_protrusion': 'remove_narrow_protrusions',
            'narrow_passage': 'fix_narrow_passage',
            'near_self_intersection': 'fix_near_self_intersection',
            'parallel_close_edges': 'fix_parallel_close_edges',
            'unknown': 'fix_narrow_passage'  # Safe default
        }
        result['recommended_fix'] = fix_mapping.get(issue_type, 'fix_narrow_passage')
    else:
        result['issue_type'] = 'none'
        result['clearance_line'] = None
        result['recommended_fix'] = 'none'

    return result
