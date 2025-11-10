"""
Unified constraint validation framework for geometry quality requirements.

This module provides a comprehensive system for validating multiple geometric
constraints simultaneously and detecting when fixes cause regressions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum, auto

from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry

# Import for MergeConstraints
try:
    from .types import MergeStrategy
except ImportError:
    # Handle case where types module not yet loaded
    MergeStrategy = None


class ConstraintType(Enum):
    """Types of geometric constraints that can be validated."""
    VALIDITY = auto()
    CLEARANCE = auto()
    OVERLAP = auto()
    AREA_PRESERVATION = auto()
    HOLE_VALIDITY = auto()


@dataclass
class ConstraintViolation:
    """
    Represents a single constraint violation.

    Attributes:
        constraint_type: Type of constraint that was violated
        severity: How severely the constraint is violated (0.0 = perfect, higher = worse)
        message: Human-readable description of the violation
        actual_value: The actual measured value (e.g., actual clearance)
        required_value: The required value (e.g., minimum clearance)
    """
    constraint_type: ConstraintType
    severity: float
    message: str
    actual_value: Optional[float] = None
    required_value: Optional[float] = None

    def __repr__(self) -> str:
        if self.actual_value is not None and self.required_value is not None:
            return (f"{self.constraint_type.name}: {self.message} "
                    f"(actual: {self.actual_value:.4f}, required: {self.required_value:.4f}, "
                    f"severity: {self.severity:.4f})")
        return f"{self.constraint_type.name}: {self.message} (severity: {self.severity:.4f})"


@dataclass
class ConstraintStatus:
    """
    Results of validating all constraints on a geometry.

    Attributes:
        geometry: The geometry that was validated
        violations: List of constraint violations (empty if all satisfied)
        validity: Whether geometry is topologically valid
        clearance: Measured minimum clearance (None if invalid or empty)
        overlap_area: Total overlap area with other geometries (for batch operations)
        area_ratio: Ratio of current area to original area
        original_area: Area of the original geometry
    """
    geometry: BaseGeometry
    violations: List[ConstraintViolation] = field(default_factory=list)
    validity: bool = False
    clearance: Optional[float] = None
    overlap_area: float = 0.0
    area_ratio: float = 1.0
    original_area: float = 0.0

    def all_satisfied(self) -> bool:
        """Check if all constraints are satisfied (no violations)."""
        return len(self.violations) == 0

    def is_better_or_equal(self, other: 'ConstraintStatus') -> bool:
        """
        Check if this status is better than or equal to another status.

        Better means:
        - Fewer violations
        - Lower total severity
        - No new violation types introduced

        Args:
            other: The status to compare against

        Returns:
            True if this status is better or equal, False if it regressed
        """
        # If this has more violations, it's worse
        if len(self.violations) > len(other.violations):
            return False

        # If this has fewer violations, it's better
        if len(self.violations) < len(other.violations):
            return True

        # Same number of violations - compare severity
        self_severity = sum(v.severity for v in self.violations)
        other_severity = sum(v.severity for v in other.violations)

        return self_severity <= other_severity

    def improved(self, other: 'ConstraintStatus') -> bool:
        """
        Check if this status is strictly better than another status.

        Args:
            other: The status to compare against

        Returns:
            True if this status is strictly better
        """
        # Fewer violations is always better
        if len(self.violations) < len(other.violations):
            return True

        # More violations is always worse
        if len(self.violations) > len(other.violations):
            return False

        # Same number of violations - check if severity decreased
        self_severity = sum(v.severity for v in self.violations)
        other_severity = sum(v.severity for v in other.violations)

        return self_severity < other_severity

    def worst_violation(self) -> Optional[ConstraintViolation]:
        """Get the most severe violation, if any."""
        if not self.violations:
            return None
        return max(self.violations, key=lambda v: v.severity)

    def get_violations_by_type(self, constraint_type: ConstraintType) -> List[ConstraintViolation]:
        """Get all violations of a specific type."""
        return [v for v in self.violations if v.constraint_type == constraint_type]

    def __repr__(self) -> str:
        if self.all_satisfied():
            return f"ConstraintStatus(all satisfied, clearance={self.clearance:.4f}, area_ratio={self.area_ratio:.2%})"
        violations_str = "; ".join(str(v) for v in self.violations)
        return f"ConstraintStatus({len(self.violations)} violations: {violations_str})"


@dataclass
class GeometryConstraints:
    """
    Defines quality constraints that geometry must satisfy.

    Constraints are optional - only non-None values are enforced.

    Attributes:
        min_clearance: Minimum required clearance (minimum_clearance property)
        max_overlap_area: Maximum allowed overlap area with other geometries
        min_area_ratio: Minimum ratio of area to preserve vs original (0.0 to 1.0)
        max_area_ratio: Maximum ratio of area vs original (allows limiting growth)
        must_be_valid: Whether geometry must be topologically valid
        allow_multipolygon: Whether MultiPolygon results are acceptable
        max_holes: Maximum number of interior holes allowed
        min_hole_area: Minimum area for holes (smaller holes removed). Zero-area holes always removed.
        max_hole_aspect_ratio: Maximum aspect ratio for holes (narrower holes removed). Default: None (no filtering).
        min_hole_width: Minimum width for holes based on OBB shorter dimension (narrower holes removed). Default: None (no filtering).

    Example:
        ```python
        # Require valid geometry with minimum 2.0 clearance and 90% area preservation
        constraints = GeometryConstraints(
            min_clearance=2.0,
            min_area_ratio=0.9,
            must_be_valid=True
        )

        status = constraints.check(geometry, original_geometry)
        if not status.all_satisfied():
            print(f"Violations: {status.violations}")
        ```
    """
    min_clearance: Optional[float] = None
    max_overlap_area: float = 0.0
    min_area_ratio: float = 0.0  # 0.0 = allow any area loss
    max_area_ratio: float = float('inf')  # Allow any area gain by default
    must_be_valid: bool = True
    allow_multipolygon: bool = True
    max_holes: Optional[int] = None
    min_hole_area: Optional[float] = None  # None = only remove zero-area holes
    max_hole_aspect_ratio: Optional[float] = None  # None = no aspect ratio filtering
    min_hole_width: Optional[float] = None  # None = no width filtering

    def check(self, geometry: BaseGeometry, original: BaseGeometry) -> ConstraintStatus:
        """
        Validate all constraints on a geometry.

        Args:
            geometry: The geometry to validate
            original: The original geometry (for area comparison)

        Returns:
            ConstraintStatus with validation results and any violations
        """
        violations = []

        # Check if None
        if geometry is None:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.VALIDITY,
                severity=float('inf'),
                message="Geometry is None",
                actual_value=0.0,
                required_value=1.0
            ))
            return ConstraintStatus(
                geometry=geometry,
                violations=violations,
                validity=False,
                original_area=original.area if original is not None else 0.0
            )

        # Check if empty
        if geometry.is_empty:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.VALIDITY,
                severity=float('inf'),
                message="Geometry is empty",
                actual_value=0.0,
                required_value=1.0
            ))
            return ConstraintStatus(
                geometry=geometry,
                violations=violations,
                validity=False,
                original_area=original.area
            )

        # Check validity
        is_valid = geometry.is_valid
        if self.must_be_valid and not is_valid:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.VALIDITY,
                severity=1000.0,  # High severity
                message=f"Geometry is invalid: {geometry.is_valid_reason if hasattr(geometry, 'is_valid_reason') else 'unknown'}",
            ))

        # Check MultiPolygon vs Polygon
        if not self.allow_multipolygon and geometry.geom_type == 'MultiPolygon':
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.VALIDITY,
                severity=100.0,
                message="MultiPolygon result not allowed",
            ))

        # Measure clearance (only for valid, non-empty geometries)
        measured_clearance = None
        if is_valid and not geometry.is_empty:
            try:
                measured_clearance = geometry.minimum_clearance

                # Check minimum clearance
                if self.min_clearance is not None and measured_clearance < self.min_clearance:
                    deficit = self.min_clearance - measured_clearance
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.CLEARANCE,
                        severity=deficit * 10.0,  # Scale by deficit
                        message=f"Clearance below minimum",
                        actual_value=measured_clearance,
                        required_value=self.min_clearance
                    ))
            except Exception:
                # Some geometries may not support minimum_clearance
                measured_clearance = None

        # Check area preservation
        original_area = original.area
        current_area = geometry.area
        area_ratio = current_area / original_area if original_area > 0 else 0.0

        if area_ratio < self.min_area_ratio:
            loss = (1.0 - area_ratio) * 100  # Convert to percentage
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.AREA_PRESERVATION,
                severity=loss,  # Severity proportional to loss
                message=f"Area loss exceeds maximum ({loss:.1f}% lost)",
                actual_value=area_ratio,
                required_value=self.min_area_ratio
            ))

        if area_ratio > self.max_area_ratio:
            gain = (area_ratio - 1.0) * 100  # Convert to percentage
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.AREA_PRESERVATION,
                severity=gain,
                message=f"Area gain exceeds maximum ({gain:.1f}% gained)",
                actual_value=area_ratio,
                required_value=self.max_area_ratio
            ))

        # Check hole count (only for Polygon)
        if self.max_holes is not None and isinstance(geometry, Polygon):
            hole_count = len(geometry.interiors)
            if hole_count > self.max_holes:
                excess = hole_count - self.max_holes
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.HOLE_VALIDITY,
                    severity=excess * 10.0,
                    message=f"Too many holes ({hole_count} > {self.max_holes})",
                    actual_value=float(hole_count),
                    required_value=float(self.max_holes)
                ))

        # Check hole properties (area, aspect ratio, width)
        # This applies to both Polygon and MultiPolygon
        if isinstance(geometry, (Polygon, MultiPolygon)):
            # Collect all holes from all polygons
            all_holes = []
            if isinstance(geometry, Polygon):
                all_holes = list(geometry.interiors)
            elif isinstance(geometry, MultiPolygon):
                for poly in geometry.geoms:
                    all_holes.extend(poly.interiors)

            # Check each hole against constraints
            violating_holes = []

            for hole in all_holes:
                hole_polygon = Polygon(hole)
                hole_area = hole_polygon.area

                # Check minimum hole area
                if self.min_hole_area is not None and hole_area < self.min_hole_area:
                    if hole_area > 1e-10:  # Don't report zero-area holes separately
                        violating_holes.append(('area', hole_area, self.min_hole_area))
                    continue

                # Check aspect ratio and width (requires OBB calculation)
                if self.max_hole_aspect_ratio is not None or self.min_hole_width is not None:
                    try:
                        from shapely.geometry import Point
                        obb = hole_polygon.minimum_rotated_rectangle
                        coords = list(obb.exterior.coords)

                        if len(coords) >= 4:
                            edge1 = Point(coords[0]).distance(Point(coords[1]))
                            edge2 = Point(coords[1]).distance(Point(coords[2]))

                            longer = max(edge1, edge2)
                            shorter = min(edge1, edge2)

                            if shorter > 0:
                                aspect_ratio = longer / shorter

                                # Check aspect ratio
                                if self.max_hole_aspect_ratio is not None and aspect_ratio > self.max_hole_aspect_ratio:
                                    violating_holes.append(('aspect', aspect_ratio, self.max_hole_aspect_ratio))
                                    continue

                                # Check minimum width
                                if self.min_hole_width is not None and shorter < self.min_hole_width:
                                    violating_holes.append(('width', shorter, self.min_hole_width))
                                    continue
                    except Exception:
                        # If we can't calculate OBB, skip this hole
                        pass

            # Add violations for holes that don't meet criteria
            if violating_holes:
                # Group by violation type for clearer reporting
                area_violations = [v for v in violating_holes if v[0] == 'area']
                aspect_violations = [v for v in violating_holes if v[0] == 'aspect']
                width_violations = [v for v in violating_holes if v[0] == 'width']

                if area_violations:
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.HOLE_VALIDITY,
                        severity=len(area_violations) * 5.0,
                        message=f"{len(area_violations)} hole(s) below minimum area",
                        actual_value=float(len(area_violations)),
                        required_value=0.0
                    ))

                if aspect_violations:
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.HOLE_VALIDITY,
                        severity=len(aspect_violations) * 5.0,
                        message=f"{len(aspect_violations)} hole(s) exceed maximum aspect ratio",
                        actual_value=float(len(aspect_violations)),
                        required_value=0.0
                    ))

                if width_violations:
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.HOLE_VALIDITY,
                        severity=len(width_violations) * 5.0,
                        message=f"{len(width_violations)} hole(s) below minimum width",
                        actual_value=float(len(width_violations)),
                        required_value=0.0
                    ))

        return ConstraintStatus(
            geometry=geometry,
            violations=violations,
            validity=is_valid,
            clearance=measured_clearance,
            overlap_area=0.0,  # Overlap checking requires multiple geometries
            area_ratio=area_ratio,
            original_area=original_area
        )

    def check_batch(
        self,
        geometries: List[BaseGeometry],
        originals: List[BaseGeometry]
    ) -> List[ConstraintStatus]:
        """
        Validate constraints on multiple geometries.

        This is useful for batch operations where you want to check all geometries
        at once, potentially including overlap detection between geometries.

        Args:
            geometries: List of geometries to validate
            originals: List of original geometries (for area comparison)

        Returns:
            List of ConstraintStatus, one per geometry
        """
        if len(geometries) != len(originals):
            raise ValueError("geometries and originals must have same length")

        # Check individual constraints for each geometry
        statuses = [self.check(geom, orig) for geom, orig in zip(geometries, originals)]

        # TODO: Add overlap detection between geometries if max_overlap_area is set
        # This would require spatial indexing similar to remove_overlaps()

        return statuses


@dataclass
class MergeConstraints:
    """
    Configuration for polygon merging during batch fixing.

    This class specifies parameters for merging close polygons as part of
    the robust fix pipeline. Merging is applied after individual geometry
    fixes and overlap resolution, with validation to ensure merged results
    still satisfy all quality constraints.

    Attributes:
        enabled: Whether to enable polygon merging
        margin: Maximum distance between polygons for merging
        merge_strategy: Strategy to use for merging (from MergeStrategy enum)
        preserve_holes: Whether to preserve holes when merging
        insert_vertices: Whether to insert vertices at boundaries
        validate_after_merge: Re-validate constraints after merging (recommended)
        fix_violations: Try to fix constraint violations in merged results
        rollback_on_failure: Rollback merge groups that fail validation after fixing

    Example:
        ```python
        from polyforge.core import MergeConstraints, MergeStrategy

        # Enable merging with selective buffer strategy
        merge_constraints = MergeConstraints(
            enabled=True,
            margin=0.5,
            merge_strategy=MergeStrategy.SELECTIVE_BUFFER,
            validate_after_merge=True
        )

        # Use in batch fixing
        fixed, warnings = robust_fix_batch(
            geometries,
            constraints=GeometryConstraints(min_clearance=2.0),
            merge_constraints=merge_constraints
        )
        ```

    See Also:
        - MergeStrategy: Available merge strategies
        - robust_fix_batch(): Batch fixing function that uses MergeConstraints
        - merge_close_polygons(): Underlying merge function
    """
    enabled: bool = False
    margin: float = 0.0
    merge_strategy: 'MergeStrategy' = None  # Will be set to default in __post_init__
    preserve_holes: bool = True
    insert_vertices: bool = False
    validate_after_merge: bool = True
    fix_violations: bool = True
    rollback_on_failure: bool = True

    def __post_init__(self):
        """Set default merge strategy if not specified."""
        if self.merge_strategy is None and MergeStrategy is not None:
            # Import here to get the actual enum
            from .types import MergeStrategy as MS
            self.merge_strategy = MS.SELECTIVE_BUFFER


__all__ = [
    'ConstraintType',
    'ConstraintViolation',
    'ConstraintStatus',
    'GeometryConstraints',
    'MergeConstraints',
]
