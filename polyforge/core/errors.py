"""Exception hierarchy for polyforge operations.

This module defines all custom exceptions used throughout polyforge.
All exceptions inherit from PolyforgeError, allowing users to catch
any polyforge-specific error with a single except clause.
"""

from typing import List, Optional, Any
from shapely.geometry.base import BaseGeometry


class PolyforgeError(Exception):
    """Base exception for all polyforge errors.

    All polyforge exceptions inherit from this, allowing users to catch
    any polyforge-specific error with a single except clause.

    Examples:
        >>> try:
        ...     result = some_polyforge_function()
        ... except PolyforgeError as e:
        ...     print(f"Polyforge error: {e}")
    """
    pass


class ValidationError(PolyforgeError):
    """Input geometry validation failed.

    Raised when input validation detects invalid parameters or geometries
    that cannot be processed.

    Attributes:
        geometry: The invalid input geometry (optional)
        issues: List of detected validation issues (optional)

    Examples:
        >>> try:
        ...     result = process_geometry(invalid_geom)
        ... except ValidationError as e:
        ...     print(f"Validation failed: {e}")
        ...     print(f"Issues: {e.issues}")
    """
    def __init__(self, message: str, geometry: Optional[BaseGeometry] = None,
                 issues: Optional[List[str]] = None):
        super().__init__(message)
        self.geometry = geometry
        self.issues = issues or []

    def __repr__(self):
        return f"ValidationError('{str(self)}', issues={self.issues})"

    def __str__(self):
        result = super().__str__()
        if self.issues:
            result += f"\nIssues found: {', '.join(self.issues)}"
        return result


class RepairError(PolyforgeError):
    """Geometry repair failed after trying all strategies.

    Raised when a geometry cannot be repaired using any available strategy.

    Attributes:
        geometry: The geometry that couldn't be repaired (optional)
        strategies_tried: List of strategies that were attempted (optional)
        last_error: The error from the last attempted strategy (optional)

    Examples:
        >>> try:
        ...     result = repair_geometry(broken_geom)
        ... except RepairError as e:
        ...     print(f"Repair failed: {e}")
        ...     print(f"Tried strategies: {e.strategies_tried}")
        ...     print(f"Last error: {e.last_error}")
    """
    def __init__(self, message: str, geometry: Optional[BaseGeometry] = None,
                 strategies_tried: Optional[List[str]] = None,
                 last_error: Optional[Exception] = None):
        super().__init__(message)
        self.geometry = geometry
        self.strategies_tried = strategies_tried or []
        self.last_error = last_error

    def __repr__(self):
        return f"RepairError('{str(self)}', strategies_tried={self.strategies_tried})"


class OverlapResolutionError(PolyforgeError):
    """Overlap resolution failed to converge or produced invalid result.

    Raised when overlap removal fails to complete successfully, either due
    to hitting max iterations or producing invalid geometries.

    Attributes:
        iterations: Number of iterations completed
        remaining_overlaps: Number of overlaps still present
        polygons: Current state of polygons (optional)

    Examples:
        >>> try:
        ...     result = remove_overlaps(polygons, max_iterations=10)
        ... except OverlapResolutionError as e:
        ...     print(f"Failed after {e.iterations} iterations")
        ...     print(f"Still have {e.remaining_overlaps} overlaps")
    """
    def __init__(self, message: str, iterations: int = 0,
                 remaining_overlaps: int = 0, polygons: Optional[List[Any]] = None):
        super().__init__(message)
        self.iterations = iterations
        self.remaining_overlaps = remaining_overlaps
        self.polygons = polygons

    def __repr__(self):
        return (f"OverlapResolutionError('{str(self)}', "
                f"iterations={self.iterations}, "
                f"remaining={self.remaining_overlaps})")


class MergeError(PolyforgeError):
    """Polygon merge operation failed.

    Raised when polygon merging fails due to invalid geometries or
    strategy-specific issues.

    Attributes:
        strategy: The merge strategy that failed (optional)
        group_indices: Indices of polygons being merged when error occurred (optional)

    Examples:
        >>> try:
        ...     result = merge_close_polygons(polygons, strategy='invalid')
        ... except MergeError as e:
        ...     print(f"Merge failed: {e}")
        ...     print(f"Strategy: {e.strategy}")
        ...     print(f"Group: {e.group_indices}")
    """
    def __init__(self, message: str, strategy: Optional[str] = None,
                 group_indices: Optional[List[int]] = None):
        super().__init__(message)
        self.strategy = strategy
        self.group_indices = group_indices

    def __repr__(self):
        return f"MergeError('{str(self)}', strategy={self.strategy})"


class ClearanceError(PolyforgeError):
    """Clearance improvement failed to meet target.

    Raised when clearance improvement operations fail to achieve the
    target minimum clearance value.

    Attributes:
        geometry: The geometry with low clearance (optional)
        target_clearance: Target minimum clearance value (optional)
        achieved_clearance: Best clearance achieved before failure (optional)
        issue_type: Type of clearance issue detected (optional)

    Examples:
        >>> try:
        ...     result = fix_clearance(polygon, target=1.0)
        ... except ClearanceError as e:
        ...     print(f"Clearance fix failed: {e}")
        ...     print(f"Target: {e.target_clearance}")
        ...     print(f"Achieved: {e.achieved_clearance}")
        ...     print(f"Issue type: {e.issue_type}")
    """
    def __init__(self, message: str, geometry: Optional[BaseGeometry] = None,
                 target_clearance: Optional[float] = None,
                 achieved_clearance: Optional[float] = None,
                 issue_type: Optional[str] = None):
        super().__init__(message)
        self.geometry = geometry
        self.target_clearance = target_clearance
        self.achieved_clearance = achieved_clearance
        self.issue_type = issue_type

    def __repr__(self):
        return (f"ClearanceError('{str(self)}', "
                f"target={self.target_clearance}, "
                f"achieved={self.achieved_clearance})")


class ConfigurationError(PolyforgeError):
    """Invalid configuration or parameter value.

    Raised when configuration parameters are invalid or incompatible.

    Examples:
        >>> try:
        ...     config.set_tolerance(-1.0)
        ... except ConfigurationError as e:
        ...     print(f"Invalid configuration: {e}")
    """
    pass


__all__ = [
    'PolyforgeError',
    'ValidationError',
    'RepairError',
    'OverlapResolutionError',
    'MergeError',
    'ClearanceError',
    'ConfigurationError',
]
