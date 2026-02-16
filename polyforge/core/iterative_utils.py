from typing import Callable

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


def iterative_improve(
    geometry: Polygon,
    target_value: float,
    improve_func: Callable[[Polygon, float], Polygon | None],
    metric_func: Callable[[Polygon], float],
    max_iterations: int = 10,
    min_improvement: float = 1e-6,
) -> Polygon:
    """Iteratively improve a polygon until target metric is reached.

    Args:
        geometry: Input polygon to improve
        target_value: Target metric value to reach
        improve_func: Function that takes current geometry and target value,
                     returns improved geometry or None if no improvement possible
        metric_func: Function that calculates current metric value
        max_iterations: Maximum number of improvement attempts
        min_improvement: Minimum required improvement per iteration

    Returns:
        Improved polygon that best meets the target metric
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


__all__ = [
    "iterative_improve",
]
