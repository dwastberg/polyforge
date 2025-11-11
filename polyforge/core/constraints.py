"""Lightweight constraint measurement built on top of :mod:`polyforge.metrics`."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from ..metrics import measure_geometry
from .types import MergeStrategy


class ConstraintType(Enum):
    VALIDITY = auto()
    CLEARANCE = auto()
    OVERLAP = auto()
    AREA_PRESERVATION = auto()
    HOLE_VALIDITY = auto()


@dataclass
class ConstraintViolation:
    constraint_type: ConstraintType
    severity: float
    message: str
    actual_value: Optional[float] = None
    required_value: Optional[float] = None


@dataclass
class ConstraintStatus:
    geometry: BaseGeometry
    violations: List[ConstraintViolation] = field(default_factory=list)
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    overlap_area: float = 0.0

    @property
    def validity(self) -> bool:
        return bool(self.metrics.get("is_valid"))

    @property
    def clearance(self) -> Optional[float]:
        return self.metrics.get("clearance")

    @property
    def area_ratio(self) -> Optional[float]:
        return self.metrics.get("area_ratio")

    def all_satisfied(self) -> bool:
        return not self.violations

    def is_better_or_equal(self, other: "ConstraintStatus") -> bool:
        if len(self.violations) != len(other.violations):
            return len(self.violations) < len(other.violations)
        return self._severity() <= other._severity()

    def improved(self, other: "ConstraintStatus") -> bool:
        if len(self.violations) != len(other.violations):
            return len(self.violations) < len(other.violations)
        return self._severity() < other._severity()

    def get_violations_by_type(self, constraint_type: ConstraintType) -> List[ConstraintViolation]:
        return [v for v in self.violations if v.constraint_type == constraint_type]

    def worst_violation(self) -> Optional[ConstraintViolation]:
        if not self.violations:
            return None
        return max(self.violations, key=lambda v: v.severity)

    def __repr__(self) -> str:
        if self.all_satisfied():
            return f"ConstraintStatus(all satisfied, clearance={self.clearance}, area_ratio={self.area_ratio})"
        violations_str = "; ".join(v.message for v in self.violations)
        return f"ConstraintStatus({len(self.violations)} violation(s): {violations_str})"

    def _severity(self) -> float:
        return sum(v.severity for v in self.violations)


@dataclass
class GeometryConstraints:
    min_clearance: Optional[float] = None
    max_overlap_area: float = 0.0
    min_area_ratio: float = 0.0
    max_area_ratio: float = float("inf")
    must_be_valid: bool = True
    allow_multipolygon: bool = True
    max_holes: Optional[int] = None
    min_hole_area: Optional[float] = None
    max_hole_aspect_ratio: Optional[float] = None
    min_hole_width: Optional[float] = None

    def check(self, geometry: BaseGeometry, original: BaseGeometry) -> ConstraintStatus:
        metrics = measure_geometry(geometry, original)
        violations: List[ConstraintViolation] = []

        if metrics.get("is_empty", False):
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.VALIDITY,
                    severity=float("inf"),
                    message="geometry is empty",
                )
            )

        if self.must_be_valid and not metrics.get("is_valid", False):
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.VALIDITY,
                    severity=1.0,
                    message="geometry is invalid",
                )
            )

        clearance = metrics.get("clearance")
        if self.min_clearance and (clearance is None or clearance + 1e-9 < self.min_clearance):
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.CLEARANCE,
                    severity=self.min_clearance - (clearance or 0.0),
                    message="minimum clearance not met",
                    actual_value=clearance,
                    required_value=self.min_clearance,
                )
            )

        area_ratio = metrics.get("area_ratio")
        if area_ratio is not None:
            if area_ratio < self.min_area_ratio:
                violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.AREA_PRESERVATION,
                        severity=self.min_area_ratio - area_ratio,
                        message="area ratio below threshold",
                        actual_value=area_ratio,
                        required_value=self.min_area_ratio,
                    )
                )
            if self.max_area_ratio != float("inf") and area_ratio > self.max_area_ratio:
                violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.AREA_PRESERVATION,
                        severity=area_ratio - self.max_area_ratio,
                        message="area ratio above threshold",
                        actual_value=area_ratio,
                        required_value=self.max_area_ratio,
                    )
                )

        self._check_holes(geometry, violations)

        return ConstraintStatus(
            geometry=geometry,
            violations=violations,
            metrics=metrics,
        )

    def _check_holes(self, geometry: BaseGeometry, violations: List[ConstraintViolation]) -> None:
        holes = _collect_holes(geometry)

        if self.max_holes is not None and len(holes) > self.max_holes:
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.HOLE_VALIDITY,
                    severity=len(holes) - self.max_holes,
                    message="too many holes",
                    actual_value=float(len(holes)),
                    required_value=float(self.max_holes),
                )
            )

        if self.min_hole_area:
            for hole in holes:
                area = Polygon(hole).area
                if area + 1e-9 < self.min_hole_area:
                    violations.append(
                        ConstraintViolation(
                            constraint_type=ConstraintType.HOLE_VALIDITY,
                            severity=self.min_hole_area - area,
                            message="hole below minimum area",
                            actual_value=area,
                            required_value=self.min_hole_area,
                        )
                    )

        if self.max_hole_aspect_ratio or self.min_hole_width:
            for hole in holes:
                try:
                    aspect_ratio, width = _hole_shape_metrics(Polygon(hole))
                except Exception:
                    continue

                if self.max_hole_aspect_ratio and aspect_ratio > self.max_hole_aspect_ratio:
                    violations.append(
                        ConstraintViolation(
                            constraint_type=ConstraintType.HOLE_VALIDITY,
                            severity=aspect_ratio - self.max_hole_aspect_ratio,
                            message="hole aspect ratio too high",
                            actual_value=aspect_ratio,
                            required_value=self.max_hole_aspect_ratio,
                        )
                    )

                if self.min_hole_width and width + 1e-9 < self.min_hole_width:
                    violations.append(
                        ConstraintViolation(
                            constraint_type=ConstraintType.HOLE_VALIDITY,
                            severity=self.min_hole_width - width,
                            message="hole too narrow",
                            actual_value=width,
                            required_value=self.min_hole_width,
                        )
                    )


@dataclass
class MergeConstraints:
    enabled: bool = False
    margin: float = 0.0
    merge_strategy: MergeStrategy = MergeStrategy.SELECTIVE_BUFFER
    preserve_holes: bool = True
    insert_vertices: bool = False
    validate_after_merge: bool = True
    fix_violations: bool = True
    rollback_on_failure: bool = True


def _collect_holes(geometry: BaseGeometry) -> List:
    if isinstance(geometry, Polygon):
        return list(geometry.interiors)
    if isinstance(geometry, MultiPolygon):
        rings = []
        for poly in geometry.geoms:
            rings.extend(poly.interiors)
        return rings
    return []


def _hole_shape_metrics(hole_polygon: Polygon) -> Tuple[float, float]:
    obb = hole_polygon.minimum_rotated_rectangle
    coords = list(obb.exterior.coords)
    if len(coords) < 4:
        raise ValueError("degenerate hole")
    from shapely.geometry import Point

    edge1 = Point(coords[0]).distance(Point(coords[1]))
    edge2 = Point(coords[1]).distance(Point(coords[2]))
    longer = max(edge1, edge2)
    shorter = min(edge1, edge2)
    if shorter <= 0:
        raise ValueError("degenerate width")
    aspect_ratio = longer / shorter
    width = shorter
    return aspect_ratio, width


__all__ = [
    "ConstraintType",
    "ConstraintViolation",
    "ConstraintStatus",
    "GeometryConstraints",
    "MergeConstraints",
]
