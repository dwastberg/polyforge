"""
Transactional fix system with rollback capability.

This module provides a transaction-based approach to applying geometry fixes
where each fix can be committed or rolled back based on constraint validation.
"""

from typing import List, Callable, Any, Optional, Tuple
from dataclasses import dataclass, field

from shapely.geometry.base import BaseGeometry

from ..core.constraints import GeometryConstraints, ConstraintStatus


@dataclass
class FixSnapshot:
    """
    A snapshot of geometry state at a point in time.

    Attributes:
        geometry: The geometry at this state
        status: Constraint validation status at this state
        fix_applied: Name of the fix that produced this state (None for initial)
        iteration: Which iteration this snapshot was created in
    """
    geometry: BaseGeometry
    status: ConstraintStatus
    fix_applied: Optional[str] = None
    iteration: int = 0


class FixTransaction:
    """
    Manages transactional application of geometry fixes with rollback capability.

    This class maintains a history of geometry states and allows trying fixes
    that are only committed if they improve or maintain constraint satisfaction.

    Example:
        ```python
        constraints = GeometryConstraints(min_clearance=2.0, min_area_ratio=0.9)
        transaction = FixTransaction(geometry, original_geometry, constraints)

        # Try a fix - will only commit if constraints don't regress
        success = transaction.try_fix(fix_clearance, min_clearance=2.0)

        if success:
            print("Fix applied successfully")
        else:
            print("Fix caused regression, rolled back")

        # Get the best result found
        best = transaction.get_best_result()
        ```

    Attributes:
        original: The original unmodified geometry
        constraints: The constraints to validate
        snapshots: History of all committed states
        current_iteration: Current iteration number
    """

    def __init__(
        self,
        geometry: BaseGeometry,
        original: BaseGeometry,
        constraints: GeometryConstraints
    ):
        """
        Initialize transaction with initial geometry.

        Args:
            geometry: Starting geometry (may be pre-processed)
            original: Original unmodified geometry (for area comparison)
            constraints: Constraints to enforce
        """
        self.original = original
        self.constraints = constraints
        self.current_iteration = 0

        # Create initial snapshot
        initial_status = constraints.check(geometry, original)
        self.snapshots: List[FixSnapshot] = [
            FixSnapshot(
                geometry=geometry,
                status=initial_status,
                fix_applied=None,
                iteration=0
            )
        ]

    @property
    def current_geometry(self) -> BaseGeometry:
        """Get the current geometry (last committed state)."""
        return self.snapshots[-1].geometry

    @property
    def current_status(self) -> ConstraintStatus:
        """Get the current constraint status."""
        return self.snapshots[-1].status

    def try_fix(
        self,
        fix_function: Callable[..., BaseGeometry],
        fix_name: Optional[str] = None,
        **fix_kwargs: Any
    ) -> bool:
        """
        Try applying a fix function, commit only if constraints don't regress.

        Args:
            fix_function: Function that takes geometry and returns fixed geometry
            fix_name: Name of the fix for logging (defaults to function name)
            **fix_kwargs: Additional arguments to pass to fix_function

        Returns:
            True if fix was committed, False if rolled back

        Example:
            ```python
            # Try fixing clearance
            success = transaction.try_fix(
                fix_clearance,
                fix_name="fix_clearance",
                min_clearance=2.0,
                max_iterations=10
            )
            ```
        """
        if fix_name is None:
            fix_name = fix_function.__name__

        self.current_iteration += 1

        # Apply the fix to current geometry
        try:
            fixed_geometry = fix_function(self.current_geometry, **fix_kwargs)
        except Exception as e:
            # Fix function raised an exception - don't commit
            return False

        # Validate all constraints on the fixed geometry
        new_status = self.constraints.check(fixed_geometry, self.original)

        # Check if this is an improvement or at least not a regression
        if new_status.is_better_or_equal(self.current_status):
            # Commit the fix
            self.commit(fixed_geometry, new_status, fix_name)
            return True
        else:
            # Rollback - don't add to snapshots
            return False

    def commit(
        self,
        geometry: BaseGeometry,
        status: ConstraintStatus,
        fix_name: str
    ) -> None:
        """
        Commit a geometry state to the transaction history.

        Args:
            geometry: The geometry to commit
            status: The constraint status of the geometry
            fix_name: Name of the fix that produced this geometry
        """
        snapshot = FixSnapshot(
            geometry=geometry,
            status=status,
            fix_applied=fix_name,
            iteration=self.current_iteration
        )
        self.snapshots.append(snapshot)

    def rollback(self, steps: int = 1) -> bool:
        """
        Rollback to a previous state.

        Args:
            steps: Number of steps to roll back (default: 1)

        Returns:
            True if rollback succeeded, False if not enough history

        Note:
            Cannot rollback past the initial state.
        """
        if steps >= len(self.snapshots):
            return False

        # Remove the last N snapshots
        self.snapshots = self.snapshots[:-steps]
        return True

    def get_best_result(self) -> Tuple[BaseGeometry, ConstraintStatus]:
        """
        Get the best geometry found based on constraint satisfaction.

        The best result is the one with:
        1. All constraints satisfied (if any exists)
        2. Fewest violations
        3. Lowest total severity

        Returns:
            Tuple of (best_geometry, best_status)
        """
        # Find snapshots with all constraints satisfied
        satisfied = [s for s in self.snapshots if s.status.all_satisfied()]

        if satisfied:
            # If any fully satisfy constraints, return the latest one
            # (latest is usually best as it has most fixes applied)
            return satisfied[-1].geometry, satisfied[-1].status

        # No perfect solution - return the one with fewest/least severe violations
        best = min(
            self.snapshots,
            key=lambda s: (len(s.status.violations), sum(v.severity for v in s.status.violations))
        )

        return best.geometry, best.status

    def get_history_summary(self) -> List[str]:
        """
        Get a human-readable summary of the transaction history.

        Returns:
            List of strings describing each snapshot

        Example output:
            ```
            [
                "Initial: 2 violations (severity: 150.0)",
                "fix_clearance: 1 violation (severity: 50.0)",
                "remove_overlaps: 0 violations"
            ]
            ```
        """
        summary = []

        for snapshot in self.snapshots:
            fix_name = snapshot.fix_applied or "Initial"
            violation_count = len(snapshot.status.violations)
            total_severity = sum(v.severity for v in snapshot.status.violations)

            if violation_count == 0:
                summary.append(f"{fix_name}: all constraints satisfied")
            else:
                summary.append(
                    f"{fix_name}: {violation_count} violation(s) "
                    f"(severity: {total_severity:.1f})"
                )

        return summary

    def has_improved(self) -> bool:
        """
        Check if any improvement has been made since initial state.

        Returns:
            True if current state is better than initial state
        """
        if len(self.snapshots) <= 1:
            return False

        initial = self.snapshots[0]
        current = self.snapshots[-1]

        return current.status.improved(initial.status)

    def get_initial_status(self) -> ConstraintStatus:
        """Get the constraint status of the initial geometry."""
        return self.snapshots[0].status

    def __repr__(self) -> str:
        return (
            f"FixTransaction({len(self.snapshots)} snapshots, "
            f"iteration {self.current_iteration}, "
            f"current violations: {len(self.current_status.violations)})"
        )


__all__ = [
    'FixSnapshot',
    'FixTransaction',
]
