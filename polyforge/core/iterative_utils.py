"""Common iterative improvement utilities.

This module provides reusable utilities for iterative geometry improvement
algorithms to eliminate code duplication.
"""

from typing import Callable, Optional
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


def iterative_improve(
    geometry: Polygon,
    target_value: float,
    improve_func: Callable[[Polygon, float], Optional[Polygon]],
    metric_func: Callable[[Polygon], float],
    max_iterations: int = 10,
    min_improvement: float = 1e-6
) -> Polygon:
    """Iteratively improve a polygon until target metric is reached.

    Generic function for iterative improvement algorithms that repeatedly
    apply a transformation until a target metric value is achieved or
    max iterations reached.

    Args:
        geometry: Input polygon to improve
        target_value: Target value for the metric
        improve_func: Function that attempts one improvement step
                     Takes (polygon, target_value) and returns improved polygon or None
        metric_func: Function that measures current metric value
                    Takes polygon and returns float
        max_iterations: Maximum number of iterations
        min_improvement: Minimum metric improvement to continue iterating

    Returns:
        Improved polygon (best result achieved)

    Examples:
        >>> def improve_clearance(poly, target):
        ...     # Try to improve minimum_clearance
        ...     return widen_narrow_passage(poly, target)
        >>>
        >>> def get_clearance(poly):
        ...     return poly.minimum_clearance
        >>>
        >>> result = iterative_improve(
        ...     polygon,
        ...     target_value=2.0,
        ...     improve_func=improve_clearance,
        ...     metric_func=get_clearance
        ... )
    """
    result = geometry
    best_result = geometry
    best_metric = metric_func(geometry)

    for iteration in range(max_iterations):
        current_metric = metric_func(result)

        # Check if target achieved
        if current_metric >= target_value:
            return result

        # Try to improve
        improved = improve_func(result, target_value)

        if improved is None:
            # Improvement failed
            return best_result

        new_metric = metric_func(improved)

        # Check if actually improved
        if new_metric > current_metric + min_improvement:
            result = improved
            if new_metric > best_metric:
                best_result = improved
                best_metric = new_metric
        else:
            # No improvement, return best so far
            return best_result

    return best_result


def iterative_clearance_fix(
    geometry: Polygon,
    min_clearance: float,
    fix_func: Callable[[Polygon, float], Optional[Polygon]],
    max_iterations: int = 10
) -> Polygon:
    """Specialized iterative improver for clearance fixing.

    Convenience wrapper around iterative_improve specifically for
    minimum_clearance improvements.

    Args:
        geometry: Input polygon
        min_clearance: Target minimum clearance
        fix_func: Function to improve clearance (polygon, target) -> fixed or None
        max_iterations: Maximum iterations

    Returns:
        Polygon with improved clearance

    Examples:
        >>> def widen_passage(poly, target_clearance):
        ...     # Implementation of passage widening
        ...     return widened_poly
        >>>
        >>> result = iterative_clearance_fix(
        ...     polygon,
        ...     min_clearance=2.0,
        ...     fix_func=widen_passage
        ... )
    """
    def get_clearance(poly: Polygon) -> float:
        return poly.minimum_clearance

    return iterative_improve(
        geometry,
        target_value=min_clearance,
        improve_func=fix_func,
        metric_func=get_clearance,
        max_iterations=max_iterations
    )


__all__ = [
    'iterative_improve',
    'iterative_clearance_fix',
]
