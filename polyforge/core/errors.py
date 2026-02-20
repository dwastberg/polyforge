from typing import Any

from shapely.geometry.base import BaseGeometry


class PolyforgeError(Exception):
    pass


class ValidationError(PolyforgeError):
    def __init__(
        self,
        message: str,
        geometry: BaseGeometry | None = None,
        issues: list[str] | None = None,
    ):
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
    def __init__(
        self,
        message: str,
        geometry: BaseGeometry | None = None,
        strategies_tried: list[str] | None = None,
        last_error: Exception | None = None,
    ):
        super().__init__(message)
        self.geometry = geometry
        self.strategies_tried = strategies_tried or []
        self.last_error = last_error

    def __repr__(self):
        return f"RepairError('{str(self)}', strategies_tried={self.strategies_tried})"


class OverlapResolutionError(PolyforgeError):
    def __init__(self, message: str, iterations: int = 0, remaining_overlaps: int = 0):
        super().__init__(message)
        self.iterations = iterations
        self.remaining_overlaps = remaining_overlaps

    def __repr__(self):
        return (
            f"OverlapResolutionError('{str(self)}', "
            f"iterations={self.iterations}, "
            f"remaining={self.remaining_overlaps})"
        )


class MergeError(PolyforgeError):
    def __init__(
        self,
        message: str,
        strategy: str | None = None,
        group_indices: list[int] | None = None,
    ):
        super().__init__(message)
        self.strategy = strategy
        self.group_indices = group_indices

    def __repr__(self):
        return f"MergeError('{str(self)}', strategy={self.strategy})"


class ClearanceError(PolyforgeError):
    def __init__(
        self,
        message: str,
        geometry: BaseGeometry | None = None,
        target_clearance: float | None = None,
        achieved_clearance: float | None = None,
        issue_type: str | None = None,
    ):
        super().__init__(message)
        self.geometry = geometry
        self.target_clearance = target_clearance
        self.achieved_clearance = achieved_clearance
        self.issue_type = issue_type

    def __repr__(self):
        return (
            f"ClearanceError('{str(self)}', "
            f"target={self.target_clearance}, "
            f"achieved={self.achieved_clearance})"
        )


class ConfigurationError(PolyforgeError):
    pass


class FixWarning(PolyforgeError):
    def __init__(
        self,
        message: str,
        geometry: BaseGeometry | None = None,
        status: Any | None = None,
        unmet_constraints: list[str] | None = None,
        history: list[str] | None = None,
    ):
        super().__init__(message)
        self.geometry = geometry
        self.status = status
        self.unmet_constraints = unmet_constraints or []
        self.history = history or []

    def __repr__(self):
        return f"FixWarning('{str(self)}', unmet={self.unmet_constraints})"

    def __str__(self):
        result = super().__str__()
        if self.unmet_constraints:
            result += f"\nUnmet constraints: {', '.join(self.unmet_constraints)}"
        if self.status and hasattr(self.status, "violations"):
            result += f"\nViolations: {len(self.status.violations)}"
        return result


__all__ = [
    "PolyforgeError",
    "ValidationError",
    "RepairError",
    "OverlapResolutionError",
    "MergeError",
    "ClearanceError",
    "ConfigurationError",
    "FixWarning",
]
